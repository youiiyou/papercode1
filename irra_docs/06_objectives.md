# 06 model/objectives.py —— 损失函数(逐行精读 + 公式映射)

文件:`model/objectives.py`(119 行),5 个纯函数,无状态。论文式(3)(4)(5)(6)(7)全部在这里落地。

调用关系:`model/build.py::IRRA.forward` 按 `loss_names` 调用;输入都是**已 `.float()` 的全局特征**(fp16→fp32 后再算损失)。

---

## 1. compute_sdm(6-42 行)—— 论文式(4)(5)(6),核心损失

### 逐行
```python
batch_size = image_fetures.shape[0]                # B(注意:函数签名里 fetures 是原作者拼写错误)
pid = pid.reshape((batch_size, 1))                 # [B,1]
pid_dist = pid - pid.t()                           # [B,B] 两两 pid 之差
labels = (pid_dist == 0).float()                   # 同身份矩阵 y_ij(对角线恒为 1)
```
**标签构造**:batch 内任意两条样本 pid 相同 → 正样本。对角线恒 1,所以每行至少有一个正样本(自己)。

```python
if image_id != None:                               # ← 默认不走(model/build.py 只传了 pids)
    image_id_mask = (两两 image_id 相同)
    labels = (labels - image_id_mask) * factor + image_id_mask
```
**软标签扩展(未启用)**:启用后权重变为——同图=1.0、同 pid 不同图=`factor`(0.3)、不同 pid=0。意图是"同图的描述最该匹配,同身份不同图次之"。`image_ids` 其实在 batch 里一直有,只是没接进来——**这是现成的消融/改进入口**(见 10 文档)。

```python
image_norm = image_fetures / norm(...)             # L2 归一化
text_norm  = text_fetures  / norm(...)
t2i_cosine_theta = text_norm @ image_norm.t()      # [B,B] 余弦相似度矩阵(sim(u,v)=uᵀv/||u||||v||)
i2t_cosine_theta = t2i_cosine_theta.t()            # 转置即另一方向
text_proj_image = logit_scale * t2i_cosine_theta   # ×50(1/τ),式(4)的分子指数
image_proj_text = logit_scale * i2t_cosine_theta

labels_distribute = labels / labels.sum(dim=1)     # q_ij = y_ij / Σ_k y_ik,式(5)的真实分布
```

```python
i2t_pred = F.softmax(image_proj_text, dim=1)                       # p_ij,式(4)
i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1)
                       - torch.log(labels_distribute + epsilon))   # p·(log p − log q)
...t2i 同理...
loss = mean(sum(i2t_loss, dim=1)) + mean(sum(t2i_loss, dim=1))     # 式(6):双向相加
```

### 公式 ↔ 代码对照(式(4)(5)(6))
| 论文 | 代码 |
|---|---|
| `p_ij = exp(sim/τ)/Σexp(sim/τ)` | `F.softmax(logit_scale * cosine, dim=1)` |
| `q_ij = y_ij/Σy_ik` | `labels / labels.sum(dim=1)` |
| `L_i2t = 1/N Σ_i Σ_j p log(p/(q+ε))` | `mean(sum(pred * (log_softmax − log(q+ε))))` |
| `L_sdm = L_i2t + L_t2i` | 两个方向 `+` 相加(**不是平均**,数值≈单方向 2 倍) |

数学上 `p·(log p − log q)` 对行求和恰好是 **KL(p‖q)**(log_softmax 就是 log p),所以代码就是论文说的 KL 散度,ε=1e-8 只护 `log 0`。
**梯度的直觉**:交叉熵项被 p 加权——模型越自信的错误配对(p 大而 q 小)惩罚越大,这就是论文说的"聚焦困难负样本";τ=0.02 把相似度×50 再 softmax,分布足够尖,梯度信号才显著(CMPM 无温度时 logits 量级不可控,正是论文动机)。

### 与评测的呼应
推理时 `Evaluator` 用的是同样的 `L2 归一化 + 余弦相似度`,与 SDM 优化的对象完全一致——训练目标与评测协议同构。

## 2. compute_mlm(45-47 行)—— 论文式(3)
```python
ce = nn.CrossEntropyLoss(ignore_index=0)
return ce(scores, labels)
```
- `scores` [B·77, 49408](全词表 logits),`labels` [B·77](被掩位置=原 token id,其余=0)。
- `ignore_index=0`:位置 0(padding 与未掩码)整行不进损失——与数据侧 "labels.append(0)" 约定闭环。
- 论文式(3)是标准 softmax CE 的矩阵写法,`reduction='mean'` 默认按**有效 token 数**平均(不是 B·77)。

## 3. compute_itc(50-71 行)—— CLIP 原版 InfoNCE(消融 Baseline,默认关闭)
```python
labels = torch.arange(batch_size)                  # 对角线为正:第 i 图配第 i 文
logits_per_image = logit_scale * image_norm @ text_norm.t()
loss = (CE(logits_per_image, labels) + CE(logits_per_text, labels)) / 2
```
与 SDM 的区别只有一点:**标签是"位置相等"而非"pid 相同"**——batch 内同 pid 的其他样本被 InfoNCE 当负样本(伪负样本),SDM 把它们并进正样本分布。消融表里 Baseline(InfoNCE)68.19 → +SDM 70.42 的差距正来源于此(+温度可控)。这也提示:**同 pid 伪负样本问题在随机采样器下更严重**,identity 采样 + SDM 理论上更配。

## 4. compute_id(74-82 行)—— Instance loss(arXiv 1711.05535)
```python
criterion = nn.CrossEntropyLoss(reduction="mean")
loss = criterion(image_logits, labels) + criterion(text_logits, labels)
return loss / 2
```
- 输入是 IRRA.classifier 的输出(图像/文本过**同一个**分类头),labels=pid。
- 就是普通多分类 CE 的图文双份平均;作用是把每个身份的图表征和文表征都拉向各自的类中心(类内聚拢)。论文式(7)的 L_id。
- CUHK 11003 类、每类样本少,这个头容易过拟合,所以权重固定 1.0 且 lr 是 backbone 的 5 倍(快速收敛)。

## 5. compute_cmpm(85-118 行)—— CMPM 对照损失(arXiv 1905.06525,默认关闭)
结构与 SDM 几乎逐行同构,只差三处,而这三处正是论文的改进点:
| | CMPM(本函数) | SDM |
|---|---|---|
| 相似度 | **原始内积** `image_embeddings @ text_norm.t()`(无温度;一个模态归一化、另一个不归一化 → 投影长度可变) | `logit_scale × 余弦`(τ=0.02 固定) |
| 目标分布 | `labels_mask / labels_mask.norm(dim=1)`(**L2 归一化**,不是概率分布,k 个正样本时每项 1/√k) | `labels / labels.sum(dim=1)`(行和为 1 的真分布) |
| ε 保护 | 同 | 同 |

论文动机原文:CMPM 的投影相当于"变权温度",无法精确控制概率分布、难以聚焦困难负样本。对照读这两个函数即可完全理解 SDM 的贡献点。

## 6. 损失组合总账(论文式(7)的落地)
默认配置 `sdm+mlm+id`、权重 sdm=1(隐含)/ mlm=1.0 / id=1.0:

```
total = KL_i2t + KL_t2i          (compute_sdm, ≈2×KL)
      + CE_掩码词                 (compute_mlm × 1.0)
      + (CE_id_图 + CE_id_文)/2   (compute_id × 1.0)
```
processor 里 `sum(v for k,v in ret.items() if 'loss' in k)` 完成求和;三项量级都在几左右,无需手动调权也能平衡(论文也没调)。

---

## 7. 本模块要记住的四件事

1. SDM = 带固定温度(×50)的对称 KL,标签按 **pid** 建(同身份皆正样本),行归一化成真分布;`image_id` 软标签分支是**已写好但未接线**的功能。
2. MLM 的 ignore_index=0 与数据侧 labels=0 约定是闭环,改动任一侧都要同步。
3. ITC 与 SDM 唯一本质差别是伪负样本(同 pid 当负例);CMPM 与 SDM 的差别是"无固定温度 + L2 归一化目标"——这两组对照把论文动机变成了可运行的代码。
4. 所有损失输入均为 fp32 全局特征;函数名里 `fetures` 拼写错误、无任何状态(每次 new CrossEntropyLoss,等价于函数式调用)。
