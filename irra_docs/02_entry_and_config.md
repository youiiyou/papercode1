# 02 入口脚本与配置系统(逐行精读)

覆盖文件:`train.py`(97 行,调试版)、`test.py`(34 行)、`visualize.py`(96 行)、`run_irra_cu/ic/rstp.sh`(各 14 行)、`dist_test.py`(34 行,新增)、`utils/options.py`(73 行)、`utils/iotools.py`(75 行)。

约定:以下讲解均以论文默认配置为例 —— batch_size=64(运行脚本的值)、图 384×128、文本 77 token、`--loss_names 'sdm+mlm+id'`。
**注意**:本仓库是用户调试版,`train.py` 相对官方 77 行版重构过(DDP 初始化);官方原版行为以"官方对照"标注,全部差异汇总见 11 文档 §2。

---

## 1. run_irra_*.sh —— 启动命令(官方 run_irra.sh 的三份数据集版)

官方只有一份 `run_irra.sh`,本仓库拆成 `run_irra_cu.sh / run_irra_ic.sh / run_irra_rstp.sh`(CUHK-PEDES / ICFG-PEDES / RSTPReid),内容除数据集名与显卡号外一致:

```bash
DATASET_NAME="CUHK-PEDES"
DATASET_ROOT="/data/ljx/ydl/project/IRRA/IRRA-main/my_dataset_root"   # 服务器路径,本机运行要改
CUDA_VISIBLE_DEVICES=3 \
python train.py \
--name irra --img_aug --batch_size 64 --MLM \
--dataset_name $DATASET_NAME --root_dir $DATASET_ROOT \
--loss_names 'sdm+mlm+id' --num_epoch 60
```

- `--name irra` 决定输出目录名;`--img_aug` 开启训练增强(翻转/裁剪/擦除);`--MLM` 让 dataloader 使用带掩码的数据集(`ImageTextMLMDataset`);`--loss_names` 决定模型建哪些头、算哪些损失。
- 显式传了 `--root_dir`(官方脚本不传,用默认 `./data`);lr 仍用默认 1e-5。
- **三个"总开关"的关系**:`--MLM` 管数据侧(是否生成 `mlm_ids/mlm_labels`),`loss_names` 里的 `mlm` 管模型侧(是否建 cross_former/mlm_head 并计算 MLM 损失)。两者必须同时开,单独开 `--MLM` 而损失里没有 `mlm` 只是白白多算掩码。
- **dist_test.py(新增,35 行)**:多卡环境自检脚本。`torchrun --nproc_per_node=2 dist_test.py` → 检查 RANK/LOCAL_RANK 环境变量、初始化 NCCL 进程组、all_gather 一个 rank 张量再打印。练手 DDP 或排查多卡环境时先用它。

---

## 2. train.py —— 训练入口(逐行,97 行调试版)

### import 区(1-21 行)
- 官方 17 个 import 基础上**新增 `import torch.distributed as dist`**(配合重构后的 DDP 初始化),其余一致:`build_dataloader`(datasets)、`do_train`(processor)、`Checkpointer`、`save_train_configs`、`setup_logger`、`build_optimizer/build_lr_scheduler`(solver)、`build_model`(model)、`Evaluator`、`get_args`、`get_rank/synchronize`(comm)。
- 这份 import 列表其实就是整个工程的模块地图:`数据 → 模型 → 求解器 → 训练循环 → 评测 → 存档`。

### set_seed(23-30 行)
```python
def set_seed(seed=0):
    torch.manual_seed(seed); torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    np.random.seed(seed); random.seed(seed)
    torch.backends.cudnn.deterministic = True   # 29 行
    torch.backends.cudnn.benchmark = True       # 30 行
```
- **坑**:`deterministic=True` 与 `benchmark=True` 同开,后者会自动选最快卷积算法、引入不确定性,"确定性"实际不成立。复现实验想严格对齐就注释掉 `benchmark`(官方对照:26-27 行)。

### init_distributed_mode(32-46 行)—— 调试版重构,官方没有这个函数
```python
def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:   # torchrun/launch 启动时注入
        args.distributed = True
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        synchronize()
    else:                                     # 单卡直接 python train.py:显式走非分布式
        args.distributed = False
        args.rank = 0; args.world_size = 1; args.local_rank = 0
```
- **官方对照**:官方在 `__main__` 里用 `num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1; args.distributed = num_gpus > 1` 探测,且从不给 `args.rank/world_size` 赋值(靠后续 get_rank() 现查)。
- **为什么改**:torchrun 的标准姿势就是读这三个环境变量;非分布式分支显式赋值让后面 `set_seed(1+args.rank)`、`torch.device("cuda", args.local_rank)` 都有确定的值可拿。

### 主流程(48-97 行)
| 行 | 代码 | 说明 |
|---|---|---|
| 49 | `args = get_args()` | 解析全部命令行参数(见下文 options.py) |
| 50 | `init_distributed_mode(args)` | DDP 初始化(见上);官方在 35-41 行内联做 |
| 52 | `set_seed(1+args.rank)` | 主进程 seed=1,DDP 各卡不同 seed。**官方对照**:`set_seed(1+get_rank())` 在初始化**之前**调用,`get_rank()` 未初始化恒返 0 → 官方多卡下所有卡 seed 相同(同序重复数据),调试版才是真正的"各卡不同种子" |
| 57 | `device = torch.device("cuda", args.local_rank)` | 每进程用自己那张卡。**官方硬编码 `device = "cuda"`**,多卡下全挤 0 号卡 |
| 59-60 | `cur_time = ...; args.output_dir = op.join(output_dir, dataset_name, f'{cur_time}_{name}')` | 输出目录形如 `logs/CUHK-PEDES/20251217_145346_irra/` |
| 61-63 | `setup_logger(...)`;`logger.info(参数逐行打印)` | 日志同时写 `train_log.txt`;`str(args).replace(',', '\n')` 让参数一行一个,方便看日志 |
| 64 | `save_train_configs(args.output_dir, args)` | **把本次全部参数 dump 成 `configs.yaml`**,test.py 靠它复现训练配置(这就是本仓库的"配置文件"机制) |
| 67-72 | `train_loader, val_img_loader, val_txt_loader, num_classes = build_dataloader(args)` | 一次拿齐:训练 loader + 评测用的图像/文本 loader + 训练集身份数(`num_classes`,ID loss 分类头要用)。70-72 有三行被注释掉的"其他 rank 再建 loader"实验痕迹,已废弃勿学 |
| 73-75 | `model = build_model(args, num_classes)`;打印参数量;`.to(device)` | build_model 内部最后会把整个模型转 fp16(见 04 文档) |
| 77-84 | DDP 包装 | `broadcast_buffers=False`:不同步 BN 统计;模型里其实没有 BN(都是 LayerNorm),此设置影响不大 |
| 85-86 | `optimizer = build_optimizer(...)`;`scheduler = build_lr_scheduler(...)` | 分组学习率:随机初始化模块 5 倍 lr(见 07 文档) |
| 88-90 | `checkpointer`、`evaluator` | `is_master` 只有 0 号卡存 checkpoint |
| 92-95 | `if args.resume: checkpoint = checkpointer.resume(...); start_epoch = checkpoint['epoch']` | 断点续训 |
| 97 | `do_train(start_epoch, args, model, train_loader, evaluator, optimizer, scheduler, checkpointer)` | 进入训练循环(07 文档) |

**数据流总览**(把整个工程串起来):
```
train.py
 ├─ build_dataloader ─→ datasets/(json标注 → Dataset → 掩码 → collate 成 dict batch)
 ├─ build_model ─→ model/build.py(IRRA) ─→ model/clip_model.py(CLIP backbone)
 ├─ build_optimizer / build_lr_scheduler ─→ solver/
 ├─ do_train ─→ processor/processor.py(循环: model(batch) → sum(loss) → backward → Evaluator.eval)
 └─ Checkpointer ─→ logs/<dataset>/<time>_<name>/{configs.yaml, train_log.txt, best.pth, tensorboard}
```

---

## 3. test.py —— 测试入口(逐行,19-35 行)

| 行 | 代码 | 说明 |
|---|---|---|
| 20-22 | 只有 `--config_file` 参数,默认 `logs/CUHK-PEDES/iira/configs.yaml` | **不重新传超参**,全部从训练时 dump 的 yaml 恢复 |
| 23 | `args = load_train_configs(args.config_file)` | yaml → EasyDict,属性可用 `args.xxx` 访问 |
| 25 | `args.training = False` | 让 `build_dataloader` 走测试分支(返回 test loader、无增强) |
| 30 | `test_img_loader, test_txt_loader, num_classes = build_dataloader(args)` | 测试分支返回 3 个值(训练分支返回 4 个,多了 train_loader)——同一函数两种返回形状,见 03 文档 |
| 31 | `model = build_model(args, num_classes=num_classes)` | 重建同结构模型 |
| 32-33 | `Checkpointer(model).load(f=op.join(args.output_dir, 'best.pth'))` | 只加载模型权重(load 不含 optimizer/scheduler);注意 `best.pth` 路径由 yaml 里存的 `output_dir` 拼出来 |
| 34-35 | `model.to(device)`;`do_inference(model, 两个 loader)` | 纯评测,无梯度 |

坑:因为 `load_train_configs` 恢复的是训练时的 `output_dir`(含时间戳),换机器/挪目录后 test.py 会找不到 `best.pth`,需要手改 yaml 里的 `output_dir`。

---

## 4. visualize.py —— 检索结果可视化(定性图)

用途:取一条测试集文本 query,画 top-10 检索结果,绿框=同 pid 正确、红框=错误(对应论文 Fig.5)。

**这个文件已经和主代码脱节,直接跑会报错**,三处硬伤:
1. 23 行 `test_img_loader, test_txt_loader = build_dataloader(args)` —— 测试分支实际返回 **3** 个值(还有 `num_classes`),解包会崩。
2. 24 行 `model = build_model(args)` —— 没传 `num_classes`,若 `loss_names` 含 `id` 会用默认 11003(CUHK-PEDES 训练集恰好 11003,所以 CUHK 能蒙对,其他数据集就错)。
3. 44 行 `gt_img_paths = test_dataset['gt_img_paths']` —— 现在的 `CUHKPEDES.test` 字典里**没有** `gt_img_paths` 键(只有 `image_pids/img_paths/caption_pids/captions`)。

修法(二次开发时):解包改成 3 个值、`build_model` 传 `num_classes`、`gt_img_path` 改为按 `caption_pids[idx]` 从 `img_paths` 里找一张同 pid 的图。注释里还有中文注释("边框宽度设置为2"),是作者自己的调试脚本。

其余逻辑没问题:31-37 行 `_compute_embedding` → L2 归一化 → 相似度矩阵 → `topk(k=10)`;54-94 行 matplotlib 画图,`ax.spines` 染色区分对错。

---

## 5. utils/options.py —— 全部命令行参数(逐段)

本仓库**没有 yaml 配置目录**,所有超参都在这个 argparse 里,默认值即论文设定。

### general(7-14 行)
| 参数 | 默认 | 说明 |
|---|---|---|
| `--local_rank` | 0 | DDP 进程内卡号(torchrun 自动注入) |
| `--name` | baseline | 实验名,进输出目录路径 |
| `--output_dir` | logs | 日志/ckpt 根目录 |
| `--log_period` | 100 | 每 100 个 iter 打印一次训练 meter |
| `--eval_period` | 1 | 每个 epoch 评测一次 |
| `--val_dataset` | test | **用测试集当验证集**(选 best 的依据);RSTPReid 有真 val 集时可改成 val |
| `--resume` / `--resume_ckpt_file` | False / "" | 断点续训 |

### model(17-26 行)
| 参数 | 默认 | 说明 |
|---|---|---|
| `--pretrain_choice` | ViT-B/16 | CLIP backbone 型号(clip_model.py 的 `_MODELS` 里有 RN50/RN101/ViT-B-16/ViT-B-32/ViT-L-14) |
| `--temperature` | 0.02 | SDM/ITC 的 τ;注意模型里是 `logit_scale=1/τ` **固定值**,不参与训练(见 04 文档) |
| `--img_aug` | False | 训练增强开关 |
| `--cmt_depth` | 4 | cross-modal transformer 层数(论文的 4 层) |
| `--masked_token_rate` | 0.8 | **死代码**:数据集掩码实际硬编码 15%/80/10/10(见 03 文档),这两个参数从未被引用 |
| `--masked_token_unchanged_rate` | 0.1 | 同上 |
| `--lr_factor` | 5.0 | 随机初始化模块(cross*/classifier/mlm_head)的 lr 放大倍数,对应论文的 5e-5 |
| `--MLM` | False | 数据侧使用 MLM 掩码数据集 |

### loss(29-31 行)
- `--loss_names` 默认 `'sdm+id+mlm'`:可选 `mlm/cmpm/id/itc/sdm`,`+` 连接。**这个字符串决定模型结构**(建不建 classifier/cross_former/mlm_head)与损失计算,是二次开发加损失的第一入口。
- `--mlm_loss_weight` / `--id_loss_weight` 默认 1.0:在模型 forward 里乘。SDM 没有权重参数(隐含 1.0),要加得自己写。

### vision / text(34-39 行)
- `--img_size (384,128)`、`--stride_size 16`:注意 argparse 里 `type=tuple` 传参得写 `--img_size 384 128`?实际上 `type=tuple` 对命令行字符串会得到 `(3,8,4)` 这样的字符元组——**不要从命令行改它**,改默认值才行(这是原代码的坑)。patch=16、stride=16 时 token 网格 24×8=192+CLS=193。
- `--text_length 77`、`--vocab_size 49408`:77=CLIP 上下文长度;49408=49152 BPE 合并 + 特殊 token(`<|mask|>` 替换了 "jekyll",见 08 文档 tokenizer)。

### solver(42-49 行)
- `--optimizer Adam`(可选 SGD/Adamw);`--lr 1e-5`;`--bias_lr_factor 2.0`(偏置 2 倍 lr);`--weight_decay 4e-5`;`--weight_decay_bias 0`;`--alpha/--beta` 是 Adam 的 β1/β2。Adam 的 eps 在 solver 里硬编码 1e-3(见 07 文档)。

### scheduler(52-60 行)
- `--num_epoch 60`;`--milestones (20,50)` 与 `--gamma 0.1` 只有 step 调度才用;`--warmup_factor 0.1`、`--warmup_epochs 5`、`--warmup_method linear`(论文:5 epoch 从 1e-6 线性升到 1e-5);`--lrscheduler cosine`;`--target_lr 0`;`--power 0.9` 是 poly 衰减的指数。

### dataset(63-70 行)
- `--dataset_name CUHK-PEDES`(三选一);`--sampler random`(可选 `identity`,PK 采样;默认随机采样,batch 内同 pid 对的分布随机);`--num_instance 4`(identity 采样时每 id 取 4 张);`--root_dir ./data`(**数据根目录,必须准备**);`--batch_size 128`(脚本用 64);`--test_batch_size 512`;`--num_workers 8`;`--test`(store_false 翻转 `args.training`)。

---

## 6. utils/iotools.py —— IO 小工具(逐行)

- 15 行 `ImageFile.LOAD_TRUNCATED_IMAGES = True`:容错加载截断图片(大规模数据集常见)。
- **read_image(18-31 行)**:路径不存在直接 raise;否则循环 try 打开直到成功(`while not got_img`),只对 IOError 重试,`.convert('RGB')` 统一三通道。这是密集 IO 下防数据损坏的常见写法(来自 ReID 代码库传统)。
- mkdir_if_missing(34-41)/check_isfile(43-47):常规。
- read_json / write_json(50-59):数据集标注的读写。
- **get_text_embedding(62-65 行):空壳函数**,只有 `open`+`pkl.load` 两行,没有任何返回——遗留死代码,别调用。
- save_train_configs / load_train_configs(67-76 行):`vars(args)` → `yaml.dump` 存配置;`yaml.load(FullLoader)` → `EasyDict` 恢复。test.py 的配置机制全靠它。注意 tuples 存进 yaml 再读出会变 list(`img_size` 会从 `(384,128)` 变 `[384,128]`)——好在下游都是解包使用,不比较类型,不会出错,但要知道有这回事。调试版把 69 行的 `os.makedirs(path)` 改成了 `os.makedirs(path, exist_ok=True)`(官方版同目录二次启动会 FileExistsError;logger.py:23 同修)。

---

## 7. 本模块要记住的五件事

1. 配置=argparse+训练时 dump 的 `configs.yaml`,test.py 靠 yaml 复现配置;没有独立配置目录。
2. `--MLM`(数据侧)与 `loss_names` 含 `mlm`(模型侧)是两个独立开关,要一起开。
3. `val_dataset=test`:模型选择直接看测试集 R1(该领域惯例,但严格说是"在测试集上选模")。
4. `img_size` 的 `type=tuple` 坑 + visualize.py 失修是两个已知坑;调试版已把 DDP 初始化重构成 `init_distributed_mode`、device 按 local_rank 指定(官方原版的行为见"官方对照"标注)。
5. 训练入口的组装顺序就是理解全仓库的阅读顺序:dataloader → model → optimizer/scheduler → do_train → evaluator/checkpointer。
