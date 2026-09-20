# 01 论文速览:IRRA

> 本文档是对论文的**简要**了解(用户要求),重点是吃透代码,代码逐行精读见后续文档。
> 论文原文:`E:\papercode\project\IRRA.pdf`(全文文本已提取,可随时深挖)。

## 1. 论文信息

| 项目 | 内容 |
|---|---|
| 标题 | Cross-Modal Implicit Relation Reasoning and Aligning for Text-to-Image Person Retrieval |
| 作者 | Ding Jiang, Mang Ye(武汉大学;通讯作者 Mang Ye) |
| 发表 | CVPR 2023(arXiv: 2303.12501,2023-03-22) |
| 官方代码 | https://github.com/anosorae/IRRA ;本仓库 `E:\papercode\project\IRRA`(调试复现版,与官方差异见 11 文档) |
| 任务 | 文本到行人图像检索(text-to-image person retrieval) |

## 2. 任务定义

给一段自然语言行人描述(如 "A woman in a gray pair of shorts, a pair of gray shoes and a white purse around her waist"),从图像库中检索出同一身份的行人图像。本质是**跨模态匹配**:学习映射,把视觉和文本模态嵌入同一联合空间。

两大挑战:
1. **类内差异大**(intra-identity variation):同一人姿态/视角/光照变化,文本描述顺序任意、有歧义;
2. **模态异构**(modality heterogeneity):视觉与语言的表征差异。

## 3. 动机(论文 Fig.1 的三代范式)

1. **早期全局匹配**:单塔各自编码,只在网络末端用匹配损失对齐全局特征 → 中间层缺乏模态交互;
2. **显式局部匹配**:用先验(人体分割、部位、颜色、名词短语)建立身体部位↔文本实体的显式对应 → 引入噪声与不确定性,推理时要存储/计算多个局部表征的成对相似度,代价大;
3. **IRRA(本文)隐式关系推理**:不依赖任何先验监督,通过 MLM 范式隐式挖掘局部 token 间的关系来**增强全局对齐**,推理时只算一次全局相似度,**零额外推理开销**。

另外两个动机:
- 之前方法用**单模态分别预训练**的 backbone(RN50+BERT 等),缺乏底层跨模态对齐能力 → IRRA 直接用**完整 CLIP**(ViT-B/16 图像编码器 + 文本 Transformer)初始化;已有用 CLIP 的工作要么冻结部分参数、要么只用图像编码器,没有完整迁移 CLIP 双塔。
- 常用匹配损失 CMPM 的投影相当于变权温度,无法精确控制概率分布、难以聚焦困难负样本 → 提出 SDM 损失。

## 4. 方法(论文 Sec.3,对应代码核心)

整体 = **双塔特征提取 + 三个训练分支**(IRR / SDM / ID),见论文 Fig.2。

### 4.1 双塔编码器(Sec.3.1)
- **图像编码器**:CLIP 预训练 ViT。图像切成 N=H×W/P² 个 patch,线性投影成 token,加位置编码和 [CLS] token,过 L 层 Transformer;[CLS] 经线性投影到联合空间,作为全局图像特征。
  - 本代码具体设定:H×W = 384×128,P=16(patch=stride=16)→ 24×8=192 patch + 1 CLS = **193 个 token**。
- **文本编码器**:CLIP 文本 Transformer。小写 BPE 分词,加 [SOS]/[EOS];**因果掩码**自注意力;最后一层 [EOS] 位置经 text_projection 得全局文本特征。
  - 本代码具体设定:序列长度 L=**77**,CLIP 词表(vocab_size=49408,其中含 <|mask|> 等特殊 token)。

### 4.2 IRR:隐式关系推理(Sec.3.2,MLM 范式)
- **随机掩码**:按 BERT 方式 mask 文本 token——15% 被选中,其中 80% 换 [MASK]、10% 换随机 token、10% 保持不变。
- **多模态交互编码器**(论文 Fig.4 (c)):掩码文本特征作为 **Query**,图像 token 作为 **Key/Value**,先过一层多头交叉注意力(MCA,式2),再过 4 层 Transformer block(式1,内为自注意力)。论文对比了 Co-attention(Fig.4a,两套独立塔)和 Merged attention(Fig.4b,拼接后单塔),本设计参数量和时延最优(消融 Tab.5:13.66M / 6.42ms vs 33.62M / 24.30ms)。
- **MLM 头**:对每个被 mask 的位置,用 MLP 分类器在全词表上预测原 token,交叉熵损失(式3,`L_irr`)。
- **直觉**(论文 Fig.3):被 mask token 的**静态词向量**作为局部细粒度锚点,迫使图像 token 与上下文化文本表征在同一语境中对齐——这就是"隐式"利用细粒度信息增强全局特征的方式。
- 论文明确说这是**首次把 MLM 从 VLP 预训练阶段用到下游微调任务**。

### 4.3 SDM:相似度分布匹配损失(Sec.3.3)
- batch 内 N 个图文对,`sim(u,v)` 为 L2 归一化后的余弦相似度;匹配概率经带温度 τ 的 softmax(式4):
  `p_ij = exp(sim(f_i^v, f_j^t)/τ) / Σ_k exp(sim(f_i^v, f_k^t)/τ)`
- 真实匹配分布:`q_ij = y_ij / Σ_k y_ik`(同身份标签归一化,batch 内同 pid 的都算正样本)。
- 损失 = KL 散度(式5),i2t 与 t2i 两个方向相加(式6)。
- 温度 τ=0.02 可精确控制分布尖锐度,使更新聚焦困难负样本:拉大非匹配对方差、提高匹配对相关性。

### 4.4 ID 损失与总目标(Sec.3.3 末)
- ID loss:softmax 分类损失,把图像/文本分类到身份类别,显式压缩类内距离(同一身份的图表征聚拢)。
- 总损失(式7):`L = L_irr + L_sdm + L_id`(代码里即 `--loss_names 'sdm+mlm+id'`,三项等权)。
- 推理时 IRR/MLM 头等虚线模块全部移除,只用双塔全局特征算一次余弦相似度。

## 5. 实验设置

**三个基准数据集**(论文 Sec.4):

| 数据集 | 身份数 | 图像数 | 描述数 | 划分 |
|---|---|---|---|---|
| CUHK-PEDES | 13,003 | 40,206 | 80,412(每图2条) | train 11,003 id / 34,054 img / 68,108 txt;val 与 test 各 1,000 id / 3,078 与 3,074 img |
| ICFG-PEDES | 4,102 | 54,522 | 54,522(每图1条) | train 3,102 id / 34,674 对;test 1,000 id / 19,848 对 |
| RSTPReid | 4,101 | 20,505(15相机) | 41,010(每图2条) | train/val/test = 3,701 / 200 / 200 id |

**评测指标**:Rank-k(k=1,5,10)、mAP、mINP。查询=文本,库=图像。

**实现细节**(论文 Sec.4,与代码 `utils/options.py` 默认值一致):
- CLIP-ViT-B/16 图像编码器 + CLIP 文本编码器(预训练);多模态交互编码器**随机初始化**,hidden 512、8 heads、4 层。
- 数据增强:随机水平翻转、pad+随机裁剪、随机擦除(`--img_aug`);图像 resize 到 384×128;文本最大 77 token。
- Adam,60 epoch,lr=1e-5,cosine 衰减,前 5 个 epoch 从 1e-6 线性 warmup;随机初始化模块 lr=5e-5(即 5 倍)。
- SDM 温度 τ=0.02。单卡 RTX3090 24G。

## 6. 主要结果

**SOTA 对比**(CUHK-PEDES,Tab.1):IRRA R1 **73.38** / R5 89.93 / R10 93.71 / mAP 66.13 / mINP 50.24,比此前最好方法领先约 3%~9% R1。值得注意:**直接用 InfoNCE 微调的 CLIP baseline 就有 R1 68.19**(超过 CFine 69.57 之前的多数方法),IRRA 在其上再 +5.19。

| 数据集 | CLIP baseline R1 | IRRA R1 | 提升 |
|---|---|---|---|
| CUHK-PEDES | 68.19 | **73.38** | +5.19 |
| ICFG-PEDES | 56.74 | **63.46** | +6.72 |
| RSTPReid | 54.05 | **60.20** | +6.15 |

**组件消融**(CUHK-PEDES,Tab.4,R1):

| 配置 | R1 | 说明 |
|---|---|---|
| Baseline(CLIP + InfoNCE) | 68.19 | 代码中的 `itc` 损失 |
| +CMPM | 59.31 | 换 CMPM 反而降;代码中的 `cmpm` |
| +SDM | 70.42 | SDM 比 CMPM 高 11.11 |
| +SDM+ID | 70.52 | ID loss 小幅提升 |
| +IRR | 71.23 | 只加 IRR 也 +3.04 |
| +SDM+IRR | 72.81 | |
| IRRA(SDM+ID+IRR) | **73.38** | 完整版 |

**交互编码器对比**(Tab.5):Co-attn 73.28 R1/33.62M/24.30ms;Merged 73.21/12.61M/19.20ms;Ours 73.38/13.66M/**6.42ms**。

**定性分析发现**(论文 Sec.4.3,对二次开发有启发):模型只学到**词级**语义,不理解**短语级**语义(因为 MLM 只随机 mask 单个 token,没有短语级掩码)——作者留作 future work,这是一个明确的可改进点。

## 7. 论文概念 ↔ 代码位置速查

| 论文概念 | 代码位置 |
|---|---|
| 图像编码器(ViT,384×128,193 token) | `model/clip_model.py::VisionTransformer` |
| 文本编码器(因果掩码,BPE,77 token) | `model/clip_model.py::CLIP.encode_text` |
| 全 token 投影改造(区别于原版 CLIP 只投影 CLS/EOT) | `clip_model.py` 中 `ln_post`/`proj` 与 `text_projection` 的应用处 |
| IRR / 多模态交互编码器(MCA + 4 层) | `model/build.py::IRRA.cross_former`(cross_attn + cross_modal_transformer) |
| MLM 随机掩码(15%/80/10/10) | `datasets/bases.py::_build_random_masked_tokens_and_labels` |
| MLM 头 + L_irr | `model/build.py::IRRA.mlm_head`、`model/objectives.py::compute_mlm` |
| SDM(式4-6) | `model/objectives.py::compute_sdm` |
| ID loss | `model/build.py::IRRA.classifier`、`objectives.py::compute_id` |
| 温度 τ(0.02) | `model/build.py` 的 `logit_scale = 1/temperature` |
| 训练总损失求和 | `processor/processor.py::do_train`(`sum of "loss" in key`) |
| 评测(Rank-k/mAP/mINP) | `utils/metrics.py::rank` + `Evaluator` |

> 注:论文正文只包含 IRR/SDM/ID 三个分支,**没有**额外的 GLS(global-local similarity)模块;代码与论文方法一一对应,`itc`/`cmpm` 两个损失是论文消融实验用的对照项,也保留在代码里。详见 `09_paper_vs_code.md`。
