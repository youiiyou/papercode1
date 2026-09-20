# 05 model/clip_model.py —— 改造版 CLIP backbone(逐行精读)

文件:`model/clip_model.py`(601 行)。OpenAI CLIP 的完整拷贝 + **为 ReID 做的三处关键改造**(非方形输入、全 token 输出、位置编码插值加载)。IRRA 论文里"Feature Extraction Dual-Encoder"的实现载体。

对照阅读建议:先记住原版 CLIP 的行为,再看本文件改了哪几行——改动集中且都有注释痕迹(被注释掉的原版行还在),非常适合 diff 式学习。

形状约定(ViT-B/16,384×128,stride=16):视觉 width=768、12 层、12 头;文本 transformer width=512、12 层、8 头;embed_dim(joint 空间)=512。

---

## 1. 权重下载(21-65 行)
- `_MODELS`:8 个 OpenAI 官方 CLIP 的 URL(ViT-B/16 是 `5806e77c...pt`)。`--pretrain_choice` 直接查这个表。
- `_download`(36-65 行):下载到 `~/.cache/clip`(或 `download_root`),**URL 倒数第二段就是 sha256**,存在且校验通过就跳过;下载完再校验,不符就 RuntimeError。断点续传不支持,但幂等。
- `name` 也可以直接传本地 `.pt` 路径(532-533 行),离线环境用它。

## 2. ResNet 家族(68-213 行)—— 本项目实际不用,但有两个坑
`Bottleneck`(68-111)是 CLIP 式 ResNet 块(stride 用前置 AvgPool 实现 anti-aliasing 下采样);`ModifiedResNet`(152-213)三层 stem + 4 个 stage + `AttentionPool2d` 注意力池化。

两个坑(想跑 `RN50`/`RN101` 时会踩):
1. **118 行** `spacial_dim[0] * spacial_dim[1] + 1`:原版标量写法 `spacial_dim ** 2 + 1` 被注释,改成只支持 tuple;而 `ModifiedResNet.__init__` 183-186 行 `input_resolution[0] // 32` 同样要求 tuple。传 int 分辨率直接 TypeError。
2. **attnpool 的位置编码不会被插值**:`load_param` 只处理 `visual.positional_embedding`(ViT)和文本位置编码;224×224 预训练的 attnpool 位置编码形状 (50, 1024) 对不上 384×128 的 (49, 1024),copy 失败只打印 error 继续 → **ResNet 路径加载完是残缺的**。`model/build.py` 里 "for CLIP ResNet visual model" 的注释说明作者试过但主线是 ViT。

## 3. 通用积木(216-262 行)—— IRR 直接复用
- **LayerNorm**(216-222):子类重写 forward,**转 fp32 算再转回**——fp16 下 LayerNorm 的方差计算容易溢出/精度损失,这是 CLIP 的稳定性技巧。IRRA 的 IRR/MLM 头也用它。
- **QuickGELU**(225-227):`x·σ(1.702x)`,GELU 的快速近似,CLIP 全线用它(包括 mlm_head)。
- **ResidualAttentionBlock**(230-251):pre-LN 残差结构 `x = x + attn(ln_1(x)); x = x + mlp(ln_2(x))`。`attention()` 每次把 `attn_mask` 转到当前 dtype/device(245 行,fp16 必需)。`need_weights=False` 只要输出。
- **Transformer**(254-262):`nn.Sequential` 套 N 个 block;`attn_mask` 构造时传入、**所有 block 共享同一个 mask 张量**——文本塔传因果掩码,视觉塔与 cross_modal_transformer 传 None。

## 4. VisionTransformer(265-305 行)—— 改造点 ①②

### 构造(266-284 行)
```python
self.num_x = (input_resolution[1] - patch_size) // stride_size + 1   # 宽度方向:(128-16)//16+1 = 8
self.num_y = (input_resolution[0] - patch_size) // stride_size + 1   # 高度方向:(384-16)//16+1 = 24
num_patches = 24*8 = 192
self.conv1 = Conv2d(3, 768, kernel_size=16, stride=stride_size)      # patch embedding
self.positional_embedding = Parameter(randn(193, 768) * 768**-0.5)   # 193 = 192+CLS(与预训练 197 不同!)
self.proj = Parameter(768 → 512)                                     # joint 空间投影
```
- **改造①**:支持非方形输入与 `stride_size`。patch 之间可以重叠(stride < patch,TransReID 的滑窗技巧,可增密 token);默认 16/16 无重叠。
- conv1 输入 384×128 → 输出 768×24×8(注意 H 对应 num_y)。

### forward(287-305 行)
```python
x = conv1(x)                       # [B,768,24,8]
x = reshape → [B,768,192] → permute → [B,192,768]
x = cat([CLS广播, x], dim=1)       # [B,193,768],CLS 用 class_embedding + zeros 广播到 batch
x = x + positional_embedding; x = ln_pre(x)
x = LND → transformer(12层) → NLD
# x = self.ln_post(x[:, 0, :])     # ← 原版:只取 CLS
x = self.ln_post(x)                # ← 改造②:全部 token 过 ln_post
x = x @ self.proj                  # ← 改造②:全部 token 过投影
return x                           # [B,193,512]
```
- **改造②是 IRR 的前提**:原版 CLIP 只输出池化后的 [B,512];这里输出**全部 193 个 token 的联合空间表征**,IRRA 才能拿 192 个 patch token 当 cross-attention 的 K/V。全局图像特征则在 IRRA.encode_image 里自己取 `x[:,0,:]`。

## 5. CLIP 类(309-463 行)

### `__init__`(310-365 行)
- `vision_layers` 是 tuple → ResNet 分支;int → ViT 分支(339-348 行,`heads = width//64`:768/64=12 头)。实际走 ViT。
- 文本塔 `Transformer(width=512, layers=12, heads=8, attn_mask=build_attention_mask())`(350-355 行)。
- `token_embedding = Embedding(49408, 512)`;`positional_embedding = (77, 512)`;`ln_final`;`text_projection = (512, 512)`。
- **363 行 `logit_scale` 被注释掉**:对比学习头被整体移除,温度/相似度计算职责移交 IRRA 类 + objectives.py。

### initialize_parameters(367-394 行)
token_embedding std=0.02、位置编码 std=0.01;ResNet 分支的 bn3 零初始化;文本塔 resblock 的三档 scaled init(attn/proj/fc)——**IRRA `__init__` 里给 cross_modal_transformer/cross_attn 的初始化就是抄这里 384-391 行的**。这些初始化随后都会被预训练权重覆盖,只有形状变化的位置编码是真随机起步。

### build_attention_mask(396-402 行)
(77,77) 上三角填 −inf 的**因果掩码**——文本第 i 个 token 只能看到 ≤i 的位置。pad 的 -inf 加到注意力分数上,softmax 后未来位置权重为 0。

### encode_image / encode_text(408-425 行)
- `encode_image`:`visual(image.type(self.dtype))`,`dtype` property 取 conv1 权重 dtype(fp16 时自动把输入转 fp16)。
- `encode_text` 逐行:
```python
x = token_embedding(text)              # [B,77,512]
x = x + positional_embedding
x = NLD→LND → transformer(因果掩码) → NLD
x = ln_final(x)
# x = x[arange(B), text.argmax(-1)] @ text_projection   # ← 原版:只投影 EOT 行
x = x @ text_projection                # ← 改造②:77 个位置全部投影
return x                               # [B,77,512]
```
- 全 token 投影后,IRRA 在外层自己取 EOT 行(`argmax` 找 49407)当全局文本特征;MLM 分支用全部 77 个位置的表征。
- **因果掩码 × MLM 的语义**:第 i 个被掩 token 的表征只融合了**左侧文本 + 注入的图像信息**(图像经由 cross_former 无因果限制),预测被掩词 = "看左文+全图猜词"。

### forward(427-443 行)
直接 `return image_features, text_features`(两个 [B,N,512] 特征图)。原版的归一化/温度/相似度矩阵计算全部注释保留(431-441 行)——**CLIP 从"对比学习模型"退化成"双塔特征提取器"**,这就是论文说的"直接迁移完整 CLIP"的实现方式。

### load_param(446-463 行)—— 改造点③:预训练权重加载
```python
param_dict = {k:v for k,v in state_dict.items() if k in self.state_dict()}   # 只留名字能对上的
# 兼容嵌套 {'model': ...} / {'state_dict': ...} 的 checkpoint
for k, v in param_dict.items():
    if k == 'visual.positional_embedding' and 形状不同:
        v = resize_pos_embed(v, ..., num_y=24, num_x=8)      # 双线性插值
    elif k == 'positional_embedding' and 形状不同:
        v = resize_text_pos_embed(v, 77)                       # ← 未定义函数!
    try: self.state_dict()[k].copy_(v)
    except: 打印 error 继续                                     # 裸 except,失败静默跳过
```
- 位置编码 197→193 的插值在这里完成(见下)。
- ⚠️ **坑①**:`resize_text_pos_embed` **在整个仓库都没有定义**(grep 可证)。文本长度永远等于预训练的 77,该分支从不执行,所以没炸;一旦你想改 `--text_length` 就会 NameError——需要自己补这个函数。
- ⚠️ **坑②**:copy 失败只打印不抛异常,静默缺权重;调新 backbone 时务必检查日志里有没有 `ERROR occur in copy`。

## 6. resize_pos_embed(467-481 行)
```python
posemb_token, posemb_grid = posemb[:, :1], posemb[0, 1:]      # 拆出 CLS 行
gs_old = sqrt(196) = 14                                        # 预训练 224×224 → 14×14 网格
posemb_grid → reshape(1,14,14,768) → permute → [1,768,14,14]
F.interpolate(size=(24,8), mode='bilinear')                    # 双线性插到 24×8
→ 还原 → cat([CLS行, 新grid]) → (193,768)
```
来自 ViT(jax)的 checkpoint 工具。**假设预训练网格是正方形**(gs_old = √len),对 224×224 的 CLIP 成立。这一步就是"CLIP 迁移到 384×128 行人图"的核心:位置编码被空间重采样,CLS 位置向量原样保留。

## 7. convert_weights(484-505 行)
递归 `model.apply`:Linear/Conv 的 weight/bias 转 half;MultiheadAttention 的 in/q/k/v_proj_weight、in_proj_bias、bias_k/v 转 half;`text_projection / proj / mcq_proj` 参数转 half(`mcq_proj` 是别的项目残留名,本仓库无此类参数,无害)。**LayerNorm 参数不在列表里,保持 fp32**(配合 §3 的 fp32 计算)。注意:它只转"权重",不干预激活——激活 dtype 由输入决定。

## 8. build_CLIP_from_openai_pretrained(508-599 行)
| 步骤 | 行 | 逻辑 |
|---|---|---|
| 找权重 | 530-535 | 名字在 `_MODELS` → 下载;是本地文件 → 直接用;否则 RuntimeError |
| 读权重 | 537-548 | 先试 JIT 加载,失败退回 `torch.load` state_dict |
| **形状推断** | 550-572 | 不看任何配置文件,**全部从 state_dict 的张量形状反推**:`visual.proj` 在 → ViT;vision_width=conv1.weight.shape[0];层数=数 `visual.*.attn.in_proj_weight` 的个数;patch=conv1 核大小;分辨率=patch×网格宽;embed_dim=text_projection.shape[1];context_length/vocab_size/文本宽/头数/层数同理。这招让加载代码与具体型号完全解耦,值得借鉴 |
| 覆盖分辨率 | 588-590 | `model_cfg['image_resolution'] = image_size (384,128)`、`stride_size=16` —— **ReID 尺寸在此注入** |
| 建模型 | 592-598 | `CLIP(**model_cfg)`(新形状位置编码随机初始化)→ `model.load_param(state_dict)`(插值加载)。返回 `(model, model_cfg)`;fp16 转换**不在这里做**(注释掉了),而是最后在 `build_model` 里对整个 IRRA(含新头)统一做 |

---

## 9. 本模块要记住的五件事

1. 三处关键改造:①非方形输入 + stride 化 patch(24×8=192 token);②`ln_post/proj` 与 `text_projection` 作用到**全部 token**(IRR 的前提);③`resize_pos_embed` 双线性插值把 197 位置编码变成 193。
2. CLIP.forward 退化为纯特征提取器,对比头被注释/移除,温度在 IRRA 类里是固定值。
3. 文本塔是**因果掩码**自注意力;MLM 的"猜词"= 左文 + 全图。
4. `resize_text_pos_embed` 未定义(改 text_length 必炸);load_param 裸 except 静默跳过不匹配权重;ResNet 路径实际不可用(attnpool 位置编码不插值)。
5. 从 state_dict 形状反推模型结构(§8)是本文件最值得学的工程技巧之一。
