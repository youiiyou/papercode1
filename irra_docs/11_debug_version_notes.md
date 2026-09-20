# 11 本仓库(调试版)与官方版的差异全清单 · 复现证据(2026-09-19 核对)

> **为什么需要这一篇**:01-10 文档描述的是**官方代码**(github.com/anosorae/IRRA)。当前 `E:\papercode\project\IRRA` 是用户在 Linux 服务器上**调试、训练并复现成功**的版本,相对官方有一批改动。本文档列出全部差异,是"当前代码 = 01-10 文档描述 + 本文差异"的差集。
>
> **核对方法**:克隆官方仓库与本地逐文件 `diff --strip-trailing-cr`(排除 Windows CRLF 干扰)。注意本仓库 git 已被重置(只有一个 "first commit",全部文件未跟踪),**不能再用 git 对比官方**,要对比就用本文档。

---

## 1. 复现结果(最重要的结论:三个数据集全部复现成功)

训练在 Linux 服务器上完成(run 脚本里 root_dir=`/data/ljx/ydl/project/IRRA/IRRA-main/my_dataset_root`),logs/ 已拷回本地。每个 epoch 都在 test 集上评测(`--val_dataset test` 默认行为),按 R1 存 best.pth。

| 数据集 | 日志目录(logs/ 下) | 硬件 | 最后 epoch 评测 R1 / R5 / R10 / mAP / mINP | best R1 @ epoch | 论文数字 R1 / mAP / mINP |
|---|---|---|---|---|---|
| CUHK-PEDES | `20251217_145346_irra` | 1×GPU | **73.376** / 89.506 / 93.827 / **66.161** / **50.321** | 73.52 @ 48 | 73.38 / 66.13 / 50.24 ✅ 基本逐位一致 |
| CUHK-PEDES | `20251213_163923_irra` | 2×GPU | — | 72.17 @ 50 | 略低 1.2(见 §2.1 第 3 条的 DDP 行为差异) |
| ICFG-PEDES | `20251109_154938_irra` | 1×GPU | **63.598** / 80.139 / 85.580 / **38.189** / **7.949** | 63.70 @ 53 | 63.46 / 38.06 / 7.93 ✅ |
| RSTPReid | `20251111_130052_irra` | 1×GPU | 58.600 / 80.450 / 86.550 / **47.349** / 26.512 | 60.00 @ 13 | 60.20 / 47.17 / 26.09 ✅(数据集小、波动大,best 与 last 差 1.4 属正常) |

- 四份 `best.pth` 都在对应日志目录里,可直接 `test.py` 加载(注意 configs.yaml 里的 output_dir 是服务器路径,本机要手改,见 02 文档 §3 的坑)。
- 其余日志目录(`20251217_14*` 系列)是多次秒级失败启动的残迹(看 train_log.txt 只有几行),可忽略。
- 结论:**caption 污染(09 文档 §2①)原样保留、未修复**的情况下复现成功——再次印证"污染不影响复现已发表数字,但改代码前必须知情"。

## 2. 相对官方版的全部改动

### 2.1 功能性改动(7 处,按重要性排序)

1. **train.py:DDP 初始化重构成 `init_distributed_mode(args)`**(32-46 行)。
   官方在 `__main__` 里用 `num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1` 探测,`args.rank/world_size/local_rank` 从不显式赋值(local_rank 只有 argparse 默认 0)。调试版:读 RANK/WORLD_SIZE/LOCAL_RANK 三个环境变量,`args.distributed` 为 True/False 两分支都把 rank/world_size/local_rank 显式设好。这是配合 torchrun 启动的标准写法,也是**官方代码在多卡下会踩的第一个坑**。
2. **train.py:16 与 processor.py:16:`device = torch.device("cuda", args.local_rank)`**(官方硬编码 `device = "cuda"`)。
   官方写法在多卡下所有进程都用 cuda:0,调试版让每个进程用自己那张卡。单卡行为不变。
3. **train.py:52:`set_seed(1+args.rank)` 且移到 DDP 初始化之后**(官方 `set_seed(1+get_rank())` 在初始化之前调用,`get_rank()` 尚未初始化返回 0,**所有 rank 都拿到 seed=1** → 两卡洗牌完全相同,等于每张卡重复处理同一份数据)。调试版各卡 seed 不同,各卡看不同数据。⚠️ 但注意:随机采样分支没有 DistributedSampler,各 rank 仍是**独立**从全量训练集随机采样(有重叠、无覆盖保证),不是严格切分——这就是 2×GPU 那次 best R1 72.17 略低于单卡 73.38 的可能原因之一。
4. **processor.py:35-37, 83-88:`tb_writer` 仅 rank 0 创建**,全部 `add_scalar` 加 `if tb_writer is not None` 保护(官方每个 rank 都开 SummaryWriter 写同一目录,多卡会互相干扰)。官方那次 2×GPU 训练能出事件文件,是 20251213 的日志——注意它是在打补丁**之前**还是之后跑的无法分辨,以当前代码为准。
5. **processor.py:44-47 + datasets/sampler_ddp.py:148, 197-199:DDP 采样器每 epoch 重洗牌**。
   官方 `RandomIdentitySampler_DDP` 的 seed 固定(`shared_random_seed()`),60 个 epoch 每 epoch 采出**完全相同**的 batch 序列;且 processor 里从不调 set_epoch。调试版:seed 改为 `shared_random_seed() + self.epoch`,新增 `set_epoch(epoch)` 方法,processor 每个 epoch 开头调用(仅 `--sampler identity` 时 train_loader 的采样器才有该属性,random 分支是 RandomSampler 没有 set_epoch,`hasattr` 判断天然跳过)。
6. **processor.py:72:iter 循环里的 `synchronize()` 被注释掉**。梯度 allreduce 本身就是同步点,去掉显式 barrier 省一点时间;评测分支的 rank0-only 逻辑不受影响(非 rank0 在 backward 的 allreduce 处等待)。
7. **datasets/build.py:104-110:官方"TODO 未完成"的 DDP+identity 分支补上了 DataLoader 构建**。官方该分支构造了 batch_sampler 却不建 train_loader,跑到 return 直接 UnboundLocalError(09 文档坑 #3);调试版补齐,现在 `--sampler identity` + 多卡可用。

另有两处小修:**iotools.py:69 与 logger.py:23 的 `os.makedirs(..., exist_ok=True)`**(官方不带 exist_ok,同目录重复启动会 FileExistsError)。

### 2.2 非功能性改动(不影响行为,影响行号)

- **中文学习注释**加在:`train.py`(几乎每行)、`datasets/bases.py`、`cuhkpedes.py`、`preprocessing.py`、`sampler.py`、`model/build.py`、`model/__init__.py`、`utils/comm.py`、`iotools.py`、`logger.py`。
- **行号漂移表**(01-10 文档中引用行号时以此为准):
  | 文件 | 行数 | 漂移规则 |
  |---|---|---|
  | train.py | 官方 77 → **97** | 结构重排(见 §2.1-1),整段重写,无简单偏移 |
  | processor/processor.py | 官方 117 → **122** | 35 起 +1(实际上 tb_writer 块 35-37),44-47 是新增 set_epoch 块,46 行之后累计 +5 |
  | model/build.py | 官方 151 → **154** | ≤35 行不变;36-92 行 +1;93-100 行 +2;**101 行起 +4** |
  | datasets/build.py | 官方 170 → **176** | ≤103 不变;**104 起 +7** |
  | datasets/sampler_ddp.py | 官方 197 → **200** | ≤148 不变;148 行处 0,196 起 +3 |
  | 其余被注释文件 | 不变 | 注释加在原行行尾,行号不变 |

### 2.3 新增文件(官方没有)

| 文件 | 作用 | 备注 |
|---|---|---|
| `datasets/cuhkpedes_badvision.py` | **下错的数据集格式**的适配器:读 `caption_all.json`(而非 `reid_raw.json`)、按 `file_path` 所在文件夹名(`train_query`/`test_query`)切分、train pid 需 sorted 重映射、`_check_before_run` 额外检查 `cam_a/cam_b/CUHK01/CUHK03/Market/test_query/train_query` 七个子目录 | **未注册进 `__factory`,不参与训练**;留作"当初数据集下错了"的教训记录(见 §3) |
| `dist_test.py` | DDP 环境自检:`torchrun --nproc_per_node=2 dist_test.py`,验证 NCCL 初始化与 all_gather 通不通 | 34 行,调服务器多卡环境用 |
| `run_irra_cu.sh` / `run_irra_ic.sh` / `run_irra_rstp.sh` | 三个数据集各一份训练脚本(bs64, sdm+mlm+id, 60ep, `--root_dir` 指向服务器路径) | 各 14 行;官方只有一份 `run_irra.sh`(本地已无此文件);**本机使用须改 DATASET_ROOT** |
| `environment_full.yml` | 服务器环境全量导出(376 行,pip freeze 风格) | `environment.yml` 是精简 conda 侧:name=irra, python 3.9, pytorch-cuda 12.1 |
| `my_dataset_root/` | 三个数据集(见 §3)+ `RSTPReid.zip` 压缩包残留 | |
| `logs/` | 训练输出(§1 表) | |
| `images/architecture.png` | README 用的论文架构图 | 官方也有,非新增 |

### 2.4 与官方完全一致的文件(只动了换行符)

`model/clip_model.py`、`model/objectives.py`、`solver/build.py`、`solver/lr_scheduler.py`、`utils/metrics.py`、`utils/simple_tokenizer.py`、`utils/checkpoint.py`、`utils/meter.py`、`utils/options.py`、`test.py`、`visualize.py`、`datasets/icfgpedes.py`、`datasets/rstpreid.py`、`utils/__init__.py`、`datasets/__init__.py`(1 行 import)、`processor/__init__.py`、`data/bpe_simple_vocab_16e6.txt.gz`。→ **05/06/08 三个文档的行号引用全部仍然有效**(除 08 §4 logger 的 exist_ok 一处)。

## 3. 数据集现状(2026-09-19)

三个数据集**全部就位且验证可训**(§1 即证据),根目录 `IRRA/my_dataset_root/`:

```
my_dataset_root/
├── CUHK-PEDES/   imgs/(40,206 张裁剪行人图) + reid_raw.json(正确标注) + caption_all.json(下错版本残留) + readme.txt
├── ICFG-PEDES/   imgs/ + ICFG-PEDES.json + processed_data/
└── RSTPReid/     imgs/ + data_captions.json(+ RSTPReid.zip 残留)
```

- 训练读的是 `reid_raw.json`(cuhkpedes.py),`caption_all.json` 只有 badvision 适配器会用。
- HANDOFF/10 文档旧版说"CUHK-SYSU 原始数据集不能直接用、还需下载"——**已过时**:错的 CUHK-SYSU 已删除,换成了正确的三个发布包。
- CLIP 权重首次运行自动下载到 `~/.cache/clip`(ViT-B/16)。

## 4. 把这套东西跑起来

```bash
# 服务器(脚本现成):
bash run_irra_cu.sh        # CUHK-PEDES;同理 _ic(ICFG) / _rstp(RSTPReid)

# 本机(有 CUDA 时;无 RANK/WORLD_SIZE 环境变量自动走非分布式分支):
python train.py --name irra --img_aug --batch_size 64 --MLM \
  --dataset_name CUHK-PEDES --root_dir ./my_dataset_root \
  --loss_names 'sdm+mlm+id' --num_epoch 60

# 用已有 best.pth 评测:
python test.py --config_file logs/CUHK-PEDES/20251217_145346_irra/configs.yaml
#   ⚠️ yaml 里 output_dir 是服务器路径,先手改成实际日志目录(02 文档 §3 的坑)

# 多卡(调试版的重构就是为这个):
torchrun --nproc_per_node=2 train.py ...(参数同上)
torchrun --nproc_per_node=2 dist_test.py   # 先验环境
```

## 5. 一句话总结

当前仓库 = 官方 IRRA + **6 处多卡/健壮性修复**(DDP 初始化、按卡 device、按 rank 种子、tb_writer 单例、采样器 per-epoch 洗牌、补全 DDP+identity 分支)+ 2 处目录创建小修 + 大量中文注释 + 1 个错误数据集适配器留档;**三个数据集均已用官方超参复现到论文水平**,学习与二次开发请以 01-10 文档 + 本差异清单为准。
