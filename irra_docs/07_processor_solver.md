# 07 processor/ 与 solver/(逐行精读)

覆盖文件:`processor/processor.py`(122 行,调试版;官方 117)、`solver/build.py`(64 行,与官方一致)、`solver/lr_scheduler.py`(87 行,与官方一致)。训练循环、优化器分组、学习率调度的全部实现。**processor 行号=当前调试版**;solver 未改动,行号对官方同样有效。

---

## 1. processor/processor.py

### do_train(11-113 行)—— 训练主循环

**准备段(14-39 行)**
- `arguments = {"num_epoch":…, "iteration": 0}`:之后 `checkpointer.save("best", **arguments)` 会把它存进 checkpoint,resume 时取回 `epoch`。
- `device = torch.device("cuda", args.local_rank)`(16 行;官方硬编码 `"cuda"`,多卡下所有进程用 0 号卡)。
- `meters`(25-34 行):AverageMeter × 8——总 loss、四个损失(sdm/itc/id/mlm)、三个准确率(img/txt/mlm)。**未启用的损失也能 update(值=0)**,日志里 `avg > 0` 过滤掉。
- `tb_writer`(35-37 行):**仅 rank 0 创建** SummaryWriter(`if get_rank() == 0`)。官方每 rank 都建、写同一目录,多卡会互相踩;调试版修掉(11 §2.1-4)。
- `best_top1 = 0.0`:选模型标准 = t2i Rank-1。

**每个 epoch(42-50 行)**:
```python
if args.distributed:                                   # 44-47 行,调试版新增
    sampler = train_loader.batch_sampler.sampler       # identity 采样分支才有 batch_sampler
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)                       # DDP PK 采样器每 epoch 重洗牌(11 §2.1-5)
```
→ 重置 meters → `model.train()`(注意:上一个 epoch 末 eval 把模型设成了 eval,这里切回)。random sampler 分支 RandomSampler 没有 set_epoch,hasattr 自动跳过,不影响默认配置。

**每个 iteration(52-72 行)**
```python
batch = {k: v.to(device) for k, v in batch.items()}   # dict batch 整体搬 cuda
ret = model(batch)                                     # IRRA.forward,返回损失+指标 dict
total_loss = sum([v for k, v in ret.items() if "loss" in k])   # 56 行 ← 论文式(7)的实现
meters[...].update(...)                                # 记账(mlm_acc 用 n=1,其余按 batch_size 加权)
optimizer.zero_grad(); total_loss.backward(); optimizer.step()
#synchronize()                                         # 72 行:官方这里有显式 barrier,调试版注释掉(梯度 allreduce 本身就是同步点)
```
- `total_loss` 的求和规则:key 含 "loss" 子串就加(`sdm_loss + id_loss + mlm_loss`)。`temperature/img_acc/...` 不含 "loss",天然排除。**加新指标时别把名字起成带 "loss" 的,除非想被求和**。
- 细节:`meters['sdm_loss'].update(ret.get('sdm_loss', 0), …)` 传入的是 **0 维 CUDA tensor**(或 int 0),AverageMeter 的 sum 会变成 GPU 张量——能跑(打印/比较都正常),只是每次日志都有隐式 GPU 同步,小低效。
- **没有梯度裁剪、没有 loss scaler**:fp16 模型直接 backward。lr=1e-5 量级下数值稳定;放大 lr 做实验时若出现 NaN,先怀疑这里。

**日志与调度(74-98 行)**
- 每 `log_period=100` iter 打一行:`Epoch[e] Iteration[i/N], loss: …, Base Lr: scheduler.get_lr()[0]`。
- epoch 末:`if tb_writer is not None`(83-88 行,rank0 守卫)TensorBoard 记 lr/temperature/各 meter(`ret['temperature']` 恒 0.02,画出来是直线,可当 sanity check);`scheduler.step()`(**按 epoch 步进**,不是按 iter);rank 0 打印吞吐(samples/s)。
- ⚠️ `scheduler.get_lr()` 在新版 PyTorch(≥2.0)已弃用(建议 `get_last_lr()`),升级 torch 版本时会看到 warning,功能尚在。

**验证与存档(99-113 行)**
```python
if epoch % eval_period == 0 and rank==0:
    top1 = evaluator.eval(model.module.eval() if 分布式 else model.eval())
    torch.cuda.empty_cache()
    if best_top1 < top1:
        best_top1 = top1; arguments["epoch"] = epoch
        checkpointer.save("best", **arguments)     # 存 logs/<...>/best.pth
最后: logger.info(f"best R1: {best_top1} at epoch {arguments['epoch']}")   # 112-113 行
```
- `eval_period=1`:每个 epoch 都在(默认)test 集上评测——再次强调"用测试集选模型"的惯例。
- DDP 下只有 rank 0 评测/存档;其他 rank 直接进下个 epoch,在 backward 的梯度 allreduce 处等 rank 0 追上(官方靠 iter 末 synchronize() 显式同步,调试版注释掉后依赖 allreduce 隐式同步,不死锁)。
- 只存 `best.pth`(R1 最优),**不存 last.pth**;resume 只能从 best 断。
- eval 后**没有**显式 `model.train()`——由下个 epoch 开头的 `model.train()` 兜住,但 resume 后第一件事也是 train(),OK。

### do_inference(116-122 行)
new 一个 `Evaluator`,跑一次 `eval`,返回 top1。test.py 的全部逻辑。

---

## 2. solver/build.py —— 优化器分组

### build_optimizer(6-49 行)—— 论文"随机初始化模块 5 倍 lr"的实现
```python
for key, value in model.named_parameters():
    lr, weight_decay = args.lr, args.weight_decay
    if "cross" in key:        lr = args.lr * args.lr_factor        # ① cross_attn / cross_modal_transformer → 5e-5
    if "bias" in key:         lr = args.lr * args.bias_lr_factor   # ② 偏置 → 2e-5(覆盖①)
                              weight_decay = 0
    if "classifier" in key or "mlm_head" in key:                   # ③ 分类/MLM 头 → 5e-5(覆盖②的 lr)
        lr = args.lr * args.lr_factor
    params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
```
- **三个 if 是顺序覆盖,不是互斥 elif**,精确结果:
  | 参数 | lr | weight_decay |
  |---|---|---|
  | CLIP backbone 权重 | 1e-5 | 4e-5 |
  | CLIP backbone 偏置 | 2e-5 | **0** |
  | cross_attn / cross_modal_transformer 权重 | **5e-5** | 4e-5 |
  | cross_* 的偏置 | 2e-5(①被②覆盖) | 0 |
  | classifier / mlm_head 权重 | 5e-5 | 4e-5 |
  | classifier / mlm_head 偏置 | **5e-5**(②被③覆盖) | 0 |
  - 所以"cross 的 bias 是 2 倍、classifier 的 bias 是 5 倍"——不一致但作者显然没在意;改分组逻辑时要知道这个现状。
- 每个参数单独一个 param group(一对一),PyTorch 完全支持,只是 group 数量大。
- Adam:`eps=1e-3`——**远大于默认 1e-8**,fp16 训练的稳态技巧(避免小二阶矩下步长爆炸);AdamW 才用 1e-8。SGD 分支不传 momentum 外的东西(per-group wd 仍然生效)。
- ⚠️ 46-47 行 `else: NotImplementedError` **没有 raise**——传未知优化器名不会报这个错,而是在 `return optimizer` 处 UnboundLocalError。

### build_lr_scheduler(52-64 行)
把 args 原样灌进 `LRSchedulerWithWarmup`:milestones=(20,50)、gamma=0.1、warmup_factor=0.1、warmup_epochs=5、warmup_method=linear、total_epochs=60、mode=cosine、target_lr=0。

---

## 3. solver/lr_scheduler.py —— LRSchedulerWithWarmup

继承 `_LRScheduler`,自定义 `get_lr()`;**按 epoch 计数**(`last_epoch` 由 processor 的 `scheduler.step()` 每 epoch +1)。

**构造校验(22-36 行)**:milestones 必须递增;mode ∈ step/exp/poly/cosine/linear;warmup ∈ constant/linear。

**get_lr 的分段逻辑(48-87 行)**:
```
epoch < 5(warmup):
    linear: warmup_factor = 0.1*(1-e/5) + e/5        # 0.1 → 1.0 线性
    返回 base_lr * warmup_factor                      # backbone: 1e-6→1e-5;新模块: 5e-6→5e-5(论文原文)
epoch ≥ 5:
    ratio = (epoch-5)/(60-5)
    cosine: factor = 0.5*(1+cos(π·ratio))             # 1 → 0
    返回 target_lr + (base_lr-target_lr)*factor       # 余弦衰减到 0
```
- 其他 mode:step(`gamma^bisect(milestones)` 阶梯)、exp、poly、linear——本配置用不到,是留的通用件。
- 关键性质:**所有 param group 按 base_lr 等比例缩放**(5 倍组全程保持 5 倍),warmup/衰减对分组透明。
- 实际 lr 曲线:1e-6 线性升 5 个 epoch 到 1e-5,再余弦降到 0(第 60 epoch)。

---

## 4. 本模块要记住的四件事

1. 总损失 = forward 返回 dict 中 key 含 "loss" 的项之和;无裁剪、无 fp16 loss scaler。
2. 优化器分组是三个顺序 if 的覆盖关系(cross→5x,bias→2x 覆盖,classifier/mlm_head→5x 再覆盖);Adam eps=1e-3。
3. lr 调度按 epoch:5 epoch 线性 warmup(1e-6→1e-5)→ cosine 衰减到 0;分组倍率全程保持。
4. 每 epoch 在(默认)test 集上评测、按 R1 存唯一的 best.pth;`scheduler.get_lr()` 是弃用 API;未知优化器名的 else 分支忘了 raise。
