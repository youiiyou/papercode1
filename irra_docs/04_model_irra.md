# 04 model/build.py —— IRRA 模型主体(逐行精读)

文件:`model/build.py`(154 行,调试版;官方 151,差异只是中文注释,+4 行漂移见 11 §2.2)。这是**整篇论文的核心**:`IRRA` 类 = CLIP 双塔 backbone + 按需构建的任务头(ID 分类器、IRR 多模态交互编码器、MLM 头),forward 里按 `loss_names` 分发计算各项损失。**本文行号=当前调试版行号**。

约定形状(B=64,ViT-B/16,384×128,stride 16,embed_dim=512):
- `images` [B,3,384,128];`caption_ids`/`mlm_ids` [B,77];`pids` [B]
- `image_feats` [B,193,512](192 patch + 1 CLS);`text_feats`/`mlm_feats` [B,77,512]
- 全局特征 `i_feats`/`t_feats` [B,512]

---

## 1. import 区(1-6 行)
从 `clip_model.py` 复用 `Transformer`、`QuickGELU`、`LayerNorm`(IRR 的组件直接用 CLIP 同款积木,保证数值行为/初始化风格一致)和 `build_CLIP_from_openai_pretrained`、`convert_weights`;`objectives` 提供全部损失函数。

## 2. `__init__`(10-61 行)—— 按任务建头

| 行 | 代码 | 讲解 |
|---|---|---|
| 14 | `self._set_task()` | 先解析任务列表(forward 分发要用) |
| 16-17 | `base_model, base_cfg = build_CLIP_from_openai_pretrained('ViT-B/16', img_size, stride_size)` | 构建并加载预训练 CLIP(自动下载权重到 `~/.cache/clip`,非方形输入会做位置编码插值,详见 05 文档);`embed_dim=512` 从 checkpoint 的 cfg 读出 |
| 19 | `self.logit_scale = torch.ones([]) * (1/args.temperature)` | τ=0.02 → scale=50。**关键坑:这是普通 tensor,不是 `nn.Parameter` 也不是 buffer**——不训练、不进 state_dict、`model.to(device)` 不会搬它(留在 CPU)。能正常工作是因为 0 维 CPU 张量与 CUDA 张量相乘不报错(隐式按值广播)。后果:①温度永远是 0.02,想调温只能改参数重训;②checkpoint 里没有 logit_scale,加载不报缺 key。论文说 τ 是超参数(不是可学习参数),与实现一致,但 CLIP 原版是可学习的,移植时注意 |
| 21-24 | `'id' in loss_names` → `classifier = nn.Linear(512, num_classes)`,weight~N(0,0.001²)、bias=0 | ID 分类头,**图像文本共用同一个头**(式(7)的 L_id)。`num_classes` 由 train.py 从数据集传入(CUHK=11003;签名默认值 11003 只是占位) |
| 26-33 | `'mlm'` → `cross_attn = nn.MultiheadAttention(512, 8, batch_first=True)`;`cross_modal_transformer = Transformer(width=512, layers=cmt_depth(4), heads=8)` | **IRR / 多模态交互编码器**(论文 Fig.2 中部、Fig.4c):一层 MCA + 4 层自注意力 Transformer。`heads = 512//64 = 8`。**随机初始化**,所以 solver 里给它 5 倍 lr |
| 36-38 | `ln_pre_t / ln_pre_i / ln_post` | 交叉注意力前的两个 LN(分别作用于 query 侧文本、key/value 侧图像)+ 输出 LN。对应论文式(1)里的 `LN(·)` |
| 40-51 | 三档 std 的截断正态初始化(attn_std=width^-0.5,proj_std=attn_std·(2L)^-0.5,fc_std=(2·width)^-0.5) | CLIP/GPT-2 风格的 scaled init:输出投影除以 √(2×层数) 抑制残差累加方差。`cross_attn` 与 4 个 resblock 全部覆盖 |
| 53-58 | `mlm_head = Linear(512,512) → QuickGELU → LN → Linear(512, 49408)` | MLM 预测头(BERT 式 dense+gelu+LN+fc 结构,激活用 CLIP 的 QuickGELU 而非 GELU),输出全词表 logits |

`_set_task`(63-65 行):`loss_names.split('+')` 存成 `self.current_task`,并打印 "Training Model with ['sdm','mlm','id'] tasks"。

## 3. `cross_former(q, k, v)`(69-80 行)—— IRR 前向,论文式(1)(2)

```python
x = self.cross_attn(ln_pre_t(q), ln_pre_i(k), ln_pre_i(v), need_weights=False)[0]
   # q=掩码文本 [B,77,512], k=v=image_feats [B,193,512]
   # 论文式(2): MCA(Q,K,V) = softmax(QK^T/√d)V,多头版
x = x.permute(1,0,2)          # NLD → LND
x = self.cross_modal_transformer(x)   # 4 层自注意力 Transformer(残差块)
x = x.permute(1,0,2)          # LND → NLD
x = self.ln_post(x)           # 输出 LN
```

- **这就是论文的"隐式关系推理"全部内容**:文本 token 以图像 token 为 K/V 做一次交叉注意力(视觉信息注入文本),再过 4 层自注意力(文本内部在"看过图像"后再充分交互)。
- permute 的原因:`nn.MultiheadAttention(batch_first=True)` 吃 [N,L,D],而 `clip_model.Transformer` 是 CLIP 风格、吃 [L,N,D](继承自原始 nn.Transformer 约定)。
- 输出 [B,77,512]:每个(掩码)文本位置的融合表征。
- 与论文 Fig.4 的对照:Co-attention 要两套塔、Merged attention 拼接后进单塔,这里只单向 cross-attn 一次 + 4 层,故参数/时延最优(Tab.5:13.66M/6.42ms)。

## 4. `encode_image` / `encode_text`(82-89 行)—— 推理期用的全局特征
- `encode_image`:`base_model.encode_image(image)` 返回**全 token** [B,193,512],取 `x[:,0,:]`(CLS)→ float。
- `encode_text`:取 `x[arange(B), text.argmax(-1)]`(EOT 位置,argmax 找 49407)→ float。
- 这两个方法只在**评测**时被 `Evaluator` 调用;训练时 forward 里直接内联同样逻辑。`.float()` 把 fp16 转 fp32(模型整体被 convert_weights 转 fp16,损失计算用 fp32 更稳)。

## 5. `forward(batch)`(91-147 行)—— 训练前向,逐段

### 5.1 backbone 编码(95-100 行)
```python
image_feats, text_feats = self.base_model(images, caption_ids)
# [B,193,512], [B,77,512] —— 两个编码器都输出全 token 投影特征
i_feats = image_feats[:, 0, :].float()                       # CLS → 全局图像特征
t_feats = text_feats[arange(B), caption_ids.argmax(-1)].float()  # EOT → 全局文本特征
```
- `CLIP.forward` 被改造成返回全 token map(原版 CLIP 只返回池化后单向量;详见 05 文档 §5)——这是让 IRR 拿到 193 个图像 token 的前提。
- ⚠️ 结合 03 文档:开 MLM 时这里的 `caption_ids` 已被数据侧污染成掩码序列,所以 `t_feats`(SDM/ID 用)实际编码的是**掩码文本**。

### 5.2 日志(102-103 行)
`ret['temperature'] = 1/logit_scale`:只是把 0.02 放进返回 dict 方便 processor 记录。注意 `ret` 里 value 不含 "loss" 字样的项不会被求和进 total_loss。

### 5.3 损失分发(107-147 行)
| 任务 | 行 | 逻辑 | 论文对应 |
|---|---|---|---|
| `itc` | 107-108 | `compute_itc(i_feats, t_feats, logit_scale)` | **不在论文配置里**;这是消融的 CLIP baseline(InfoNCE),保留用于对照实验 |
| `sdm` | 110-111 | `compute_sdm(i_feats, t_feats, pids, logit_scale)` | 式(4)(5)(6),双向 KL,详见 06 文档 |
| `cmpm` | 113-114 | `compute_cmpm(i_feats, t_feats, pids)` | 消融对照损失 CMPM(无温度,不传 logit_scale) |
| `id` | 116-127 | `classifier(i_feats.half())` / `classifier(t_feats.half())` → CE 均值 ×`id_loss_weight`(1.0);顺带算 img/txt top1 准确率进 ret | 式(7)的 L_id。`.half()` 再转回 float:fp16 模型里 Linear 是 fp16 权重,输入必须 fp16(全局特征刚 .float() 过,这里转回去) |
| `mlm` | 129-145 | 见下 | 式(1)(2)(3),L_irr |

### 5.4 MLM 分支逐行(129-145 行)
```python
mlm_ids = batch['mlm_ids']                        # 130 行,[B,77] 掩码序列
mlm_feats = self.base_model.encode_text(mlm_ids)  # 132 行,[B,77,512] 用【共享的 CLIP 文本编码器】重新编码掩码文本
x = self.cross_former(mlm_feats, image_feats, image_feats)  # 134 行,Q=掩码文本, K=V=干净图像 token
x = self.mlm_head(x)                              # 136 行,[B,77,49408] 每个位置预测词表分布
scores = x.float().reshape(-1, vocab_size)        # 138 行,[B*77, 49408]
mlm_labels = batch['mlm_labels'].reshape(-1)      # 139 行,[B*77],非掩码位置为 0
ret['mlm_loss'] = compute_mlm(scores, mlm_labels) * mlm_loss_weight   # 140 行,CE(ignore_index=0)
pred = scores.max(1)[1]                           # 142 行
acc = (pred[nonzero(labels)] == labels[nonzero(labels)]).float().mean()  # 143-144 行
ret['mlm_acc'] = acc                              # 145 行
```
四个理解要点:
1. **掩码文本走的是同一个 CLIP 文本编码器**(权重共享,不额外建 MLM encoder);`<|mask|>` 是词表里的合法 token(49405),embedding 查得到。因为 CLIP 文本注意力是**因果掩码**,每个位置只能看到左侧上下文+图像(图像经由 cross_former 注入)——即"用左侧文本+全图预测被掩的词"。
2. `image_feats` 来自**原始图像**(没有掩码概念),是融合的 K/V。
3. mlm_acc 只在被掩位置(nonzero(labels),排除 0)上算,是训练曲线上观察 IRR 是否在学的直接指标。
4. ⚠️ 冗余:如 03 文档所述,`mlm_ids` 与 `caption_ids` 内容相同,`encode_text` 被调用两次(浪费约一次文本塔前向);若修复数据侧别名问题,这里天然就是"干净文本一次、掩码文本一次"的正确结构。

### 5.5 返回值约定
forward 返回 dict:`sdm_loss / id_loss / mlm_loss / mlm_loss_weight 加权后`、`itc_loss / cmpm_loss`(如开)、`temperature / img_acc / txt_acc / mlm_acc`(日志用)。processor 里 `total_loss = sum(v for k,v in ret.items() if 'loss' in k)`——**任何 key 含 "loss" 的项都会被加进总损失**,加新损失时命名要小心(比如叫 `xxx_metric` 就不会被误加;反之想被自动求和就带 `loss` 后缀)。这就是论文式(7) `L = L_irr + L_sdm + L_id` 的实现。

## 6. `build_model`(150-154 行)
```python
model = IRRA(args, num_classes)
convert_weights(model)   # 整个模型(含 IRR/MLM 头)转 fp16
return model
```
fp16 策略:权重 fp16、前向 fp16,只在算损失/评测前 `.float()`。无 loss scaler(processor 里直接 backward)——lr=1e-5 很小,训练能稳。换更大 lr 或复现遇 NaN 时优先怀疑这里。

---

## 7. 本模块要记住的五件事

1. IRRA = 复用 CLIP 全部积木(Transformer 块、QuickGELU、LayerNorm、初始化风格)+ 三个新增模块:`classifier`(ID)、`cross_attn + cross_modal_transformer + 3 个 LN`(IRR)、`mlm_head`。新增部分全部随机初始化、5 倍 lr。
2. IRR 的信息流:掩码文本 token(Q)× 图像 token(K/V) → 1 层 MCA → 4 层自注意力 → 每 token 预测原词。**只在训练时存在,推理零开销**(评测只调 encode_image/encode_text)。
3. `logit_scale` 是不可学习的固定 CPU tensor,不进 checkpoint;温度是纯超参数。
4. `loss_names` 字符串同时决定**结构**(建哪些头)和**损失**(算哪些项);返回 dict 里 key 含 "loss" 即被自动求和。
5. 训练时 text 全局特征与 MLM 分支编码的是同一份(被污染的)掩码序列;`encode_text` 重复调用一次是可优化点。
