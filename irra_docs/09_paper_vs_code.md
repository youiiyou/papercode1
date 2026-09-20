# 09 论文 ↔ 代码 系统对照

目的:逐项核对论文(CVPR 2023, arXiv 2303.12501)声明的方法与开源代码的一致性,给"复现是否忠实""哪里有出入"一个权威答案。结论先行:**三个核心组件(SDM/IRR/ID)与式(1)-(7)全部忠实实现,连消融实验的对照损失(ITC/CMPM)都保留在代码里;存在一个影响训练语义的隐蔽副作用(caption 污染)和一批不影响主线的死代码/坑**。

## 1. 方法组件对照总表

| 论文概念 | 论文位置 | 代码位置 | 一致性 |
|---|---|---|---|
| CLIP ViT-B/16 双塔初始化 | Sec.3.1 | `clip_model.py::build_CLIP_from_openai_pretrained` | ✅ 完整 CLIP(图像+文本塔都加载预训练权重) |
| 图像 384×128、patch 16、N=192 patch | Sec.3.1/4 | `VisionTransformer.__init__` num_x=8, num_y=24 | ✅(还支持 stride<patch 重叠,论文未提,来自 TransReID) |
| 位置编码适配非方形输入 | 隐含 | `resize_pos_embed` 双线性插值 197→193 | ✅(论文未展开,代码做了) |
| 文本 BPE、L=77、[SOS]/[EOS] | Sec.3.1 | `bases.py::tokenize` + `SimpleTokenizer` | ✅(论文写 vocab 49152 指 BPE 合并数;代码词表 49408=256+256+48894+3,数学上是同一套 BPE) |
| MLM 掩码 15%/80/10/10 | Sec.3.2 | `bases.py::_build_random_masked_tokens_and_labels` | ✅ 硬编码;**--masked_token_rate 两个参数是死代码** |
| MCA + 4 层交互编码器(Q=掩文,K/V=图) | 式(1)(2)、Fig.4c | `model/build.py::cross_former`(cross_attn + cross_modal_transformer + 3 LN) | ✅ hidden 512、8 头、4 层 |
| MLM 头 + L_irr | 式(3) | `mlm_head` + `objectives.compute_mlm`(CE ignore_index=0) | ✅ 式(3)就是 CE 的矩阵写法;mean 归约=按有效 token 平均 |
| SDM:softmax 匹配概率 | 式(4) | `compute_sdm` 35/37 行 `F.softmax(logit_scale*cos)` | ✅ |
| SDM:KL 双向求和 | 式(5)(6) | `compute_sdm` 36-40 行 | ✅ `p·(log p−log(q+ε))` 对行求和恰为 KL;ε 只护 log |
| 温度 τ=0.02 | Sec.4 | `logit_scale=1/temperature` 固定 tensor | ✅ 是超参数非可学习(论文如此设计;但注意它不进 checkpoint,见 §3) |
| ID loss | 式(7) | `classifier` + `compute_id` | ✅ 图文共一个分类头,CE 均值 |
| 总损失 L_irr+L_sdm+L_id | 式(7) | processor `sum("loss" in key)`,权重 1:1:1 | ✅ |
| 推理只算一次全局相似度 | Sec.3/4 | `Evaluator` 只调 encode_image/encode_text | ✅ cross_former/mlm_head 推理不跑 |
| 数据增强:翻转/pad 裁剪/擦除 | Sec.4 | `build_transforms` aug 分支 | ✅ |
| Adam、60ep、1e-5、cosine、warmup 1e-6→1e-5、新模块 5e-5 | Sec.4 | solver(分组 5 倍 + LRSchedulerWithWarmup) | ✅(Adam eps=1e-3 论文未提) |
| 消融 Baseline(CLIP+InfoNCE) | Tab.4 No.0 | `loss_names 'itc'` | ✅ 保留 |
| 消融 +CMPM | Tab.4 No.1 | `loss_names 'cmpm'` | ✅ 保留 |
| 指标 R1/R5/R10/mAP/mINP | Sec.4 | `metrics.py::rank` | ✅ 语义逐行核对过(08 文档 §1) |
| 复现数字(CUHK 73.38/66.13/50.24) | Tab.1 | README | ✅ 一致 |

**代码里有、论文里没有的**:
- `compute_sdm` 的 `image_id` 软标签分支(同图 1.0 / 同 pid 异图 0.3):写了但未启用,model 调用时没传 `image_id`;
- `--sampler identity`(PK 采样)与 DDP 版采样器:论文未讨论采样策略;
- IRR 模块结构按 `embed_dim` 自适应(`embed_dim//64` 头数),换 ViT-L/14 也能直接建;
- visualize.py 定性可视化脚本(对应论文 Fig.5,但已失修)。

## 2. 论文表述与代码行为的出入(按严重度排序)

### ① caption 污染(最重要,影响训练语义)
- **论文**:全局分支编码原文 T(SDM/ID 用),IRR 分支编码掩码文 T̂ —— 两条通路输入不同。
- **代码**:03 文档 §3 详述的共享内存别名使 `caption_ids` 也变成掩码序列 → **全局分支与 MLM 分支输入相同**;干净文本从未进入模型。`encode_text` 对同一序列跑了两次。
- **判断**:更像无意 bug 而非设计(EOT 不掩码,池化特征仍有效,故指标正常)。按论文意图修复 = 一行 deepcopy。**改代码前必须知道这一点,否则你复现的"IRRA"和作者跑出来的不是同一个东西。**

### ② "模型选择在 test 集上"
- 论文没说模型选择策略;代码 `--val_dataset test` + 每_epoch 评测 + 按 test R1 存 best。领域惯例,但严格的评测口径下应改用 val(RSTPReid 有 val;CUHK/ICFG 没有独立 val,只能如此)。

### ③ 可学习性相关的三个"非注册"细节
- `logit_scale` 不是 Parameter/buffer(不训练、不存档、不随 `.to(device)` 搬迁——靠 0 维 CPU 张量与 CUDA 张量相乘的隐式兼容);
- `masked_token_rate/unchanged_rate` 选项与实现脱钩;
- optimizer 分组的 if 覆盖顺序造成 cross 的 bias 是 2 倍 lr 而 classifier 的 bias 是 5 倍(07 文档 §2 的表)。
这些不影响默认复现,但改超参/换结构时容易踩。

## 3. 死代码与坑清单(复现/移植前过一遍)

> 状态标注以**当前调试版**为准(官方版行为差异见 11 文档);✅已修复 = 用户调试时已改。

| # | 位置 | 问题 | 触发条件 | 建议 | 状态 |
|---|---|---|---|---|---|
| 1 | `bases.py:157` | caption_ids 被掩码原地污染(§2①) | 开 `--MLM` | deepcopy 修复或保持原样但知情 | 未修(复现保持原样) |
| 2 | `visualize.py` | 解包 2 值/缺 num_classes/读不存在的 `gt_img_paths` | 运行该脚本 | 按 02 文档 §4 修三处 | 未修 |
| 3 | `datasets/build.py:95-103` | DDP+identity 分支不给 train_loader 赋值(官方) | DDP 且 identity 采样 | 补 DataLoader 或避开 | ✅已修复(104-110 行补上了 DataLoader) |
| 4 | `clip_model.py:458` | `resize_text_pos_embed` 未定义 | 改 `--text_length` | 自己实现文本位置编码插值 | 未修 |
| 5 | `solver/build.py:47` | `else: NotImplementedError` 没 raise | 优化器名拼错 | 改 raise | 未修 |
| 6 | `train.py:29-30` | cudnn deterministic 与 benchmark 同开 | 总是 | 想确定性别开 benchmark | 未修(官方 26-27 行) |
| 7 | `clip_model.py:118` 等 | ResNet 分支:标量分辨率崩、attnpool 位置编码不插值 | `--pretrain_choice RN*` | 只用 ViT 系列 | 未修 |
| 8 | `options.py:34` | `--img_size` `type=tuple` 从命令行传会得到字符元组 | 命令行改尺寸 | 改默认值再跑 | 未修 |
| 9 | `iotools.py:62` | `get_text_embedding` 空壳 | 调用即错 | 删除 | 未修 |
| 10 | `preprocessing.py` | 未 import 的自定义 RandomErasing | 无 | 忽略 | — |
| 11 | `processor.py:80` | `scheduler.get_lr()` 弃用 API | torch≥2.0 出 warning | 换 `get_last_lr()` | 未修(官方 75 行) |
| 12 | `metrics.py:31` | mINP 循环假设每 query 至少一个同 pid | 稀疏 pid 数据集 | 加保护 | 未修 |
| 13 | 全局 | `device` 硬编码(官方 `"cuda"`)、fp16 无 loss scaler | CPU/MPS 或大 lr 实验 | 按需改造 | 半修:调试版改为 `torch.device("cuda", args.local_rank)`(多卡正确),仍无 CPU 回退 |
| 14 | `comm.py:90` | `reduce_dict` 无人调用;`all_gather` 与 sampler_ddp 重复 | 无 | 了解即可 | — |

## 4. 参数保真度核对(论文 Sec.4 ↔ 默认值)

| 参数 | 论文 | 代码默认 | 备注 |
|---|---|---|---|
| backbone | CLIP ViT-B/16 | `ViT-B/16` | ✅ |
| 输入尺寸 | 384×128 | `(384,128)` | ✅ |
| 文本长度 L | 77 | `77` | ✅ |
| 交互编码器 | hidden 512 / 8 heads / 4 层 | 由 embed_dim 与 cmt_depth 推出 | ✅ |
| 优化器 | Adam | Adam(betas 默认,eps=1e-3) | ✅(eps 未提) |
| epoch | 60 | 60 | ✅ |
| lr | 1e-5(新模块 5e-5) | 1e-5 × lr_factor 5 | ✅ |
| 调度 | cosine + 5ep warmup(1e-6→1e-5) | warmup_factor 0.1 → 1.0 线性 | ✅ |
| τ | 0.02 | 0.02 | ✅ |
| batch size | 未写明 | 默认 128,run_irra.sh 64 | README 脚本 64 应为实际值 |
| 增强三件套 | 翻转/裁剪/擦除 | `--img_aug` | ✅ |
| 硬件 | RTX3090 24G 单卡 | — | 单卡即可复现 |

## 5. 一句话总结
代码是论文的**忠实且略超集**的实现(多出的 itc/cmpm/软标签/采样器都是消融或预留),唯一实质性行为偏差是 ①caption 污染;它不阻止复现论文数字,但任何"基于 IRRA 改进"的工作都应先决定:保持该行为(与已发表结果对齐)还是修复它(与论文表述对齐),并在论文里说明。
