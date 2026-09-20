# 08 utils/ 工具模块(逐行精读)

覆盖文件:`metrics.py`(102 行)、`simple_tokenizer.py`(136 行)、`checkpoint.py`(149 行)、`logger.py`(32 行)、`meter.py`(20 行)、`comm.py`(117 行)。(`options.py`、`iotools.py` 已在 02 文档精读。)

---

## 1. metrics.py —— 评测协议(论文 Sec.4 的指标定义)

### rank(similarity, q_pids, g_pids, max_rank=10, get_mAP=True)(9-39 行)

输入:相似度矩阵 [Q,G](query=文本,gallery=图像)、两侧 pid。

```python
if get_mAP: indices = torch.argsort(similarity, dim=1, descending=True)   # 全排序
else:       _, indices = torch.topk(similarity, k=max_rank, …)             # 只要前 10,加速
pred_labels = g_pids[indices.cpu()]         # 每个 query 的检索结果按相似度降序的 pid
matches = pred_labels.eq(q_pids.view(-1, 1))# [Q,G] 布尔:检索到的图与 query 同 pid 即正确
```
- **多正样本语义**:gallery 里所有同 pid 图都算命中(不是只认配对的那张)。
- 注意 `indices.cpu()` 把索引搬回 CPU 再取 pid——G 大时(Q≈6156×G≈3074 for CUHK)矩阵乘在 GPU、排序在 CPU/GPU 混合。

**CMC(Rank-k)**(20-22 行):
```python
all_cmc = matches[:, :10].cumsum(1)   # 前 k 里命中个数的累计
all_cmc[all_cmc > 1] = 1              # clamp:只关心"至少命中一个"
all_cmc = all_cmc.float().mean(0)*100 # 对所有 query 取平均 → [10]
# R1=cmc[0], R5=cmc[4], R10=cmc[9]
```
这就是论文"Rank-k = 前 k 里至少找到一个正确样本的概率"的严格实现。

**mINP**(28-32 行):
```python
inp = [ tmp_cmc[i][match_row.nonzero()[-1]] / (match_row.nonzero()[-1] + 1)  for … ]
```
对每个 query:找**最后一个**命中位置的精确率 = (到该位置的累计命中数)/(该位置下标+1)。含义:把最难的正确样本挖出来时的 precision——mINP 越高说明最难样本排得越靠前。逐 query 的 Python 循环(O(Q) 次 nonzero),大数据集上是最慢的一步。⚠️ 隐含假设:每个 query 在 gallery 至少有一个同 pid(这些数据集都满足;否则 `nonzero()[-1]` 直接 IndexError)。

**mAP**(34-37 行):
```python
tmp_cmc = [ tmp_cmc[:, i] / (i+1) for i … ]   # precision@k 矩阵
tmp_cmc = stack * matches                       # 只保留命中位置的 precision
AP = tmp_cmc.sum(1) / num_rel                   # 每 query 的 AP = 命中处 precision 的平均
mAP = AP.mean() * 100
```

### Evaluator(42-101 行)
- `_compute_embedding`(48-73 行):先遍历**文本 loader**(query),再**图像 loader**(gallery);`no_grad` 下调 `model.encode_text/encode_image`——**只走双塔全局特征,cross_former/mlm_head 完全不跑**。论文"零额外推理开销"的代码证据就在这两行。`pid.view(-1)` 展平拼接。
- `eval`(75-101 行):双侧 L2 归一化 → `similarity = qfeats @ gfeats.t()`(余弦=归一化内积,与 SDM 优化目标同构)→ `rank()` → PrettyTable 打 t2i 行(R1/R5/R10/mAP/mINP,`custom_format` 保留三位小数);可选 `i2t_metric=True` 时对转置矩阵再算一遍图→文。**返回 t2i R1**(numpy float),供 do_train 选 best。
- 读数对照:CUHK-PEDES 复现目标 R1 73.38 / mAP 66.13 / mINP 50.24。

## 2. simple_tokenizer.py —— CLIP BPE 分词器 + `<|mask|>` 改造

GPT-2 式 byte-level BPE,与 OpenAI CLIP 的区别只在词表改造(见下)。

- `default_bpe`(10-12 行):词表路径 = `utils/../data/bpe_simple_vocab_16e6.txt.gz`,**即仓库根的 `data/` 目录**——挪动仓库结构会断。`@lru_cache()` 只解析一次路径。
- `bytes_to_unicode`(15-35 行):GPT-2 的字节→可见 unicode 映射表(256 项),让 BPE 能在任意 utf-8 文本上可逆工作。
- `get_pairs` / `basic_clean`(ftfy 修乱码 + html 反转义) / `whitespace_clean`(折叠空白):标准预处理。
- **`__init__` 的词表拼装(62-81 行),逐行算 ID**:
  ```
  merges = 文件第 1~48894 行(跳过首行版本注释),共 48894 条合并规则
  vocab = 256 个字节符 + 256 个 '字节</w>' + 48894 个合并结果 = 49406
  vocab.pop(-1)                      # 删掉最后一个合并 token("jekyll")→ 49405
  vocab.extend(['<|mask|>', '<|startoftext|>', '<|endoftext|>'])  → 49408
  ```
  于是 **`<|mask|>`=49405、`<|startoftext|>`=49406、`<|endoftext|>`=49407**——词表大小保持 CLIP 的 49408 不变,embedding 矩阵形状与预训练完全兼容(mask 槽位顶掉了几乎不会用到的 "jekyll")。这三个 ID 与 `bases.py` 的掩码条件 `0 < token < 49405`、`token_range = range(1, len-3)` 严格互锁,改动词表必须同步。
  - `self.cache` 预置三个特殊 token 直通;`self.pat` 正则把特殊 token 列在最前(普通描述里不含它们,但保证万一出现时不被切碎)。
- `bpe()`(83-122 行):标准贪心合并——反复取 rank 最小的相邻符号对合并,直到不可再合;词尾加 `</w>` 标记。
- `encode()`(124-130 行):`whitespace_clean(basic_clean(text)).lower()` → 按 `pat` 切词(字母串/单个数字/标点串)→ 字节编码 → BPE → 查 ID。**全程小写**(论文 Sec.3.1 "lower-cased BPE")。
- `decode()`(132-135 行):逆过程,`</w>` 还原成空格。

## 3. checkpoint.py —— detectron2 式存档器

- `save(name, **kwargs)`(28-45 行):`save_dir` 与 `save_to_disk`(=is_master)都真才写;`{"model": state_dict, "optimizer": …, "scheduler": …, **kwargs(含 epoch)}` → `save_dir/name.pth`。全程只存 `best.pth` 一种。
- `load(f)`(47-54 行):只加载模型权重(test.py 用);`resume(f)`(56-71 行):模型 + optimizer + scheduler 全恢复,pop 掉后返回剩余 dict(train.py 从中取 `checkpoint['epoch']`)。
- **模糊匹配加载**(90-148 行,detectron2 同款):
  - `strip_prefix_if_present`:若所有 key 都有 `module.` 前缀(DDP 存档)则整体剥掉;
  - `align_and_update_state_dicts`:构造"当前 key 是否以加载 key 结尾"的匹配矩阵,每个当前 key 取**最长后缀匹配**的加载张量塞进 state_dict,再 `strict` 加载。作用:跨版本/跨包装加载时自动对齐名字。本仓库自己存自己载时是恒等映射,属于安全网。
  - 残留细节:logger 名字还是 `"PersonSearch.checkpoint"`(拷贝来源项目);匹配日志逐 key 打印很长,但能当加载审计用。
- ⚠️ 与 05 文档呼应:`logit_scale` 不在 state_dict(普通 tensor),best.pth 里也没有它——加载后温度仍由 args 决定,不冲突。

## 4. logger.py —— 日志装配(7-32 行)
`setup_logger(name, save_dir, if_train, distributed_rank)`:
- rank>0 直接返回空 logger(非主进程不记日志);
- 控制台 StreamHandler(stdout) + 文件 FileHandler:`train_log.txt` 用 **mode='w'(每次训练覆盖)**,`test_log.txt` 用 **mode='a'(追加)**;
- save_dir 不存在则创建(train.py 先建了目录,这里兜底);调试版 23 行是 `os.makedirs(save_dir, exist_ok=True)`(官方不带 exist_ok,同目录二次启动会 FileExistsError);
- 格式:`时间 名称 级别: 消息`。所有模块用 `logging.getLogger("IRRA.xxx")` 共享这套体系。

## 5. meter.py —— AverageMeter(1-20 行)
标准实现:`update(val, n)` 累加 `val*n`,avg=sum/count。训练循环里损失按 batch_size 加权、mlm_acc 按 1 加权(每 batch 一个均值)。

## 6. comm.py —— DDP 通信原语
- `get_world_size/get_rank`:dist 未初始化时安全返回 1/0(所以单卡代码不用改)。
- `synchronize`:world_size>1 时 `dist.barrier()`(processor 每 iter 调一次)。
- `all_gather`(47-87 行):任意可 pickle 对象的跨卡收集(pickle→ByteTensor→padding 对齐→all_gather→反序列化),NCCL/CUDA 版。与 `sampler_ddp.py` 里的 gloo 版重复实现了一份。
- `reduce_dict`(90-117 行):按键排序 stack 后 `dist.reduce` 到 rank 0 求均值——**本仓库没有任何地方调用**(DDP 下指标平均本该用它),属于"备好没用上"的工具;单卡训练无影响。

## 7. `__init__.py` 们
`model/__init__.py` 导出 `build_model`(调试版加了包机制注释);`solver/__init__.py` 导出 `build_optimizer/build_lr_scheduler`;`datasets/__init__.py` 导出 `build_dataloader`;`processor/__init__.py` 导出 `do_train/do_inference`;只有 `utils/__init__.py` 是空文件。各包对外只暴露工厂函数,内部模块互 import 走相对路径——这就是"从 XX import build_YY"能工作的原因。

---

## 8. 本模块要记住的四件事

1. 评测只用双塔全局特征 + 余弦相似度;CMC=前 k 至少一中、mAP=命中处 precision 均值、mINP=最后命中位置的 precision;同 pid 多图全算正样本;返回 R1 做模型选择。
2. tokenizer 靠"删 jekyll + 加 3 个特殊 token"保持词表 49408;mask=49405/sot=49406/eot=49407 与掩码逻辑互锁;词表路径写死为仓库 `data/`。
3. checkpoint 是 detectron2 式后缀模糊匹配加载,兼容 `module.` 前缀;只存 best.pth;logit_scale 永不进存档。
4. `reduce_dict` 无人调用、`comm.all_gather` 与 sampler_ddp 里的重复;logger 的 train_log 每次覆盖、test_log 追加。
