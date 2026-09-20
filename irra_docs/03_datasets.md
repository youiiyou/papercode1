# 03 datasets/ 数据管线(逐行精读)

覆盖文件:`build.py`(176 行)、`bases.py`(209 行)、`cuhkpedes.py`(114 行)、`icfgpedes.py`(104 行)、`rstpreid.py`(103 行)、`preprocessing.py`(50 行)、`sampler.py`(67 行)、`sampler_ddp.py`(200 行),另有 `cuhkpedes_badvision.py`(129 行,用户留档,见 §4.4)。

> 行号说明:调试版在多个文件里加了中文注释(行号不变)并改了 build.py/sampler_ddp.py(行号有漂移,见 11 §2.2)。本文行号均为当前代码行号。

数据管线职责一句话:**json 标注 → (pid, image_id, img_path, caption) 元组 → Dataset 包装(读图/增强/BPE 分词/MLM 掩码)→ collate 成 dict batch → DataLoader**。

---

## 1. 数据目录约定(已就绪 ✅)

代码期望 `--root_dir` 下有三个数据集目录。**当前实际位置是 `IRRA/my_dataset_root/`**(运行时把 root_dir 指过去即可;run 脚本里的服务器路径是训练时的值):

```
my_dataset_root/
├── CUHK-PEDES/  ├── imgs/ └── reid_raw.json(另有 caption_all.json,是当初下错格式的残留,训练不读它,见 §4.4)
├── ICFG-PEDES/  ├── imgs/ └── ICFG-PEDES.json
└── RSTPReid/    ├── imgs/ └── data_captions.json
```

三个数据集都已在服务器上完整训练复现成功(见 11 §1),目录格式正确性已被实践验证。

标注 json 是"每图一条记录"的列表:`[{'split': 'train'/'test'/'val', 'captions': [描述...], 'file_path' 或 'img_path': 相对路径, 'id': 身份 int}]`。

---

## 2. build.py —— DataLoader 组装工厂

### `__factory`(17 行)
`{'CUHK-PEDES': CUHKPEDES, 'ICFG-PEDES': ICFGPEDES, 'RSTPReid': RSTPReid}`,`--dataset_name` 的字符串直接映射到数据集类。**加新数据集的三步**:写解析器类 → 注册进 `__factory` → `utils/options.py` 里 help 文本加名字(见 10 文档)。

### build_transforms(20-52 行)
- 归一化参数是 **CLIP 的 mean/std**(`0.48145466,...`),不是 ImageNet 的——因为 backbone 是 CLIP,必须保持预训练时的统计量。
- **测试/验证**(26-32 行):Resize(384,128) → ToTensor → Normalize。无任何随机性。
- **训练 + `--img_aug`**(35-44 行):Resize → RandomHorizontalFlip(0.5) → **Pad(10)** → RandomCrop(384,128) → ToTensor → Normalize → **RandomErasing(scale=(0.02,0.4), value=mean)**。
  - Pad+Crop:四周填 10 像素再随机裁回原尺寸 = 平移扰动;
  - `T.RandomErasing` 未传 `p`,默认 **p=1(必擦除)**;擦除区域填 CLIP mean 而不是 0。论文说的三种增强(翻转/裁剪/擦除)就在这里。
- **训练无 aug**(46-51 行):只有翻转。即不开 `--img_aug` 也始终有水平翻转。

### collate(55-69 行)—— 自定义批组装
- 输入是 `list[dict]`(每个样本一个 dict),先转成 `dict[list]`(56-58 行,取所有样本 key 的并集,缺的填 None——兼容有无 mlm 字段的数据集混用)。
- 再按类型组 batch:int → `torch.tensor`(pids、image_ids);tensor → `torch.stack`(images [B,3,384,128]、caption_ids [B,77]、mlm_ids、mlm_labels);其他类型直接 raise。
- **这就是模型 forward 收到的 `batch` dict 的来源**:`batch['images'/'caption_ids'/'pids'/'image_ids'/'mlm_ids'/'mlm_labels']`。

### build_dataloader(71-176 行)—— 主函数
| 行 | 逻辑 |
|---|---|
| 75-76 | 实例化数据集类;`num_classes = len(dataset.train_id_container)`(训练集身份数,ID loss 分类头的输出维度) |
| 78 | `args.training=True` 走训练分支(train.py),False 走测试分支(test.py) |
| 85-92 | `--MLM` 开 → `ImageTextMLMDataset`(多产出 mlm_ids/mlm_labels);否则 `ImageTextDataset` |
| 94-110 | `--sampler identity`:PK 采样,`RandomIdentitySampler`(见 §6)。其中分布式子分支(95-110)构造 `RandomIdentitySampler_DDP` + `BatchSampler` + `train_loader`(104-110 行)——**这段 train_loader 构建是用户补的**(官方 TODO 未完成分支,只建了 batch_sampler 就撒手,return 时必崩;见 11 §2.1-7) |
| 112-122 | identity 非分布式分支:DataLoader + `RandomIdentitySampler` 作 sampler |
| 123-130 | `--sampler random`(默认):普通 shuffle=True 的 DataLoader。**论文官方命令就是随机采样**——batch 内同 pid 正样本数是随机的,SDM 的 batch 内标签分布随 batch 波动 |
| 131-132 | 其他值只打 error 日志但**不抛异常**,train_loader 未定义会在后续崩 |
| 135 | 验证集选择:`args.val_dataset == 'val'` 用 val,否则用 **test**(默认,RSTPReid 之外两数据集没有 val 划分) |
| 136-149 | 评测用 `ImageDataset`(返回 (pid, img))与 `TextDataset`(返回 (pid, tokens));两个 loader 都不 shuffle(顺序即 gallery/query 顺序,评测依赖 pid 对齐) |
| 151 | 训练分支返回 **4** 个值:`(train_loader, val_img_loader, val_txt_loader, num_classes)`;val loader 用 `args.batch_size` |
| 153-176 | 测试分支:无增强 transforms、`test_batch_size=512`、只用 test 集,返回 **3** 个值(无 train_loader)。签名里 `tranforms` 是拼写错误,且只在测试分支生效 |

**原已知 bug 已修复**:官方代码 DDP + identity sampler 分支(95-103 行)只构造 `batch_sampler` 不建 train_loader(代码里 `# TODO wait to fix bugs`),走到 return 时 UnboundLocalError;用户在 104-110 行补上了 DataLoader 构建。注:默认 random sampler + 单卡训练完全用不到这个分支。

---

## 3. bases.py —— 数据集基类与四个 Dataset 包装

### BaseDataset.show_dataset_info(14-39 行)
用 PrettyTable 打印 train/test/val 的 ids/images/captions 统计。注意 train 的 captions 数 = `len(self.train)`(每条 caption 一个元组),test 的 = `len(test['captions'])`。

### tokenize(42-57 行)—— 文本 → 定长 token 序列
```python
sot = encoder["<|startoftext|>"]; eot = encoder["<|endoftext|>"]
tokens = [sot] + tokenizer.encode(caption) + [eot]        # 前后加特殊 token
result = torch.zeros(text_length)                          # 77,补零
if 超长: 截断到 77 且强制最后一位是 eot
result[:len(tokens)] = tensor(tokens)
```
- 输出定长 77 的 LongTensor,**padding 是 0**——这也是 MLM label 用 0 表示"不预测"的原因(0 永远不会是真实 token)。
- 模型取全局文本特征用 `caption_ids.argmax(-1)` 找 EOT 位置(49407 是词表最大 id,argmax 必然落在 EOT 上,包括截断时强制写入的 EOT)。

### ImageTextDataset(60-90 行)
训练用(不开 MLM 时)。`__getitem__`:`dataset[index]` 解包 `(pid, image_id, img_path, caption)` → `read_image`(带 IO 重试)→ transform → tokenize,返回 4 个字段的 dict。

### ImageDataset / TextDataset(93-130 行)
评测用,极其简单:图像侧返回 `(pid, img)`,文本侧返回 `(pid, tokens)`。默认 collate(stack 成 tensor)即可,无需自定义。

### ImageTextMLMDataset(133-210 行)—— MLM 数据侧核心

`__getitem__`(149-168 行):
```python
caption_tokens = tokenize(caption, ...)                                    # 干净序列
mlm_tokens, mlm_labels = self._build_random_masked_tokens_and_labels(
    caption_tokens.cpu().numpy())                                          # ← 注意传的是共享内存的 numpy 视图
ret = {'pids':…, 'image_ids':…, 'images':…,
       'caption_ids': caption_tokens, 'mlm_ids': mlm_tokens, 'mlm_labels': mlm_labels}
```

`_build_random_masked_tokens_and_labels`(170-210 行),BERT 式掩码:
- `mask` = `<|mask|>` 的 id(49405);`token_range = range(1, len(encoder)-3)` = 1~49404(即所有真实词 token,排除 0 padding 和 49405/49406/49407 三个特殊 token)。
- 逐 token 判断:`0 < token < 49405` 才可能被选中(所以 SOT/EOT/padding 永不被掩码);
- 15% 概率被选中(`prob < 0.15`),选中后再除以 0.15 归一:`<0.8` 换 [MASK](80%)、`<0.9` 换随机 token(10%)、否则保持(10%);
- **被选中的位置无论替换与否,labels 都记原 token**;未选中位置 label=0(损失里 ignore)。
- 205-208 行兜底:一条描述一个 token 都没掩到时,强制掩位置 1(第一个真实 token)。
- 选项里的 `--masked_token_rate(0.8)`、`--masked_token_unchanged_rate(0.1)` **没有接线**,15%/80/10/10 全部硬编码。

**⚠️ 最重要的隐蔽行为(共享内存别名)**:`.cpu().numpy()` 返回与 `caption_tokens` **共享存储**的数组,而掩码函数在原地改 `tokens[i]`。因此返回 dict 时:
- `mlm_ids` = 掩码后序列(末尾 `torch.tensor(tokens)` 是拷贝);
- `caption_ids` = **同一个掩码后序列**(张量被原地污染,干净的原文丢失)!

后果(结合 04 文档的 forward 看):
1. 全局分支(SDM/ID 用的 text_feats)编码的**也是掩码文本**,与论文"全局分支用原文、MLM 分支用掩码文"的表述不完全一致——这更像是无意的副作用而非设计;
2. 模型 forward 里 `encode_text(caption_ids)` 与 `encode_text(mlm_ids)` 输入完全相同,第二次编码是冗余计算(可以顺手优化);
3. 想按论文意图修复:在调用掩码函数前 `copy.deepcopy(caption_tokens.numpy())` 或先 `tokens.copy()`。EOT 池化不受影响(EOT 不会被掩码),所以模型照样能训、指标也没崩,这就是它长期没被发现的原因。

---

## 4. 三个数据集解析器

### cuhkpedes.py(逐行)
- 类 docstring 给了统计:13,003 ids / 40,206 imgs / 80,412 captions,9 张图有超过 2 条描述,4 个身份只有 1 张图。
- 31-38 行:拼路径 `root/CUHK-PEDES/`、`imgs/`、`reid_raw.json`,然后 `_check_before_run()`(107-114 行,目录/标注不存在就 RuntimeError,是最常见的报错入口)。
- 41-45 行:`_split_anno` 按 `anno['split']` 分三堆 → `_process_anno(training=True/False)` 生成 train/val/test。
- `_process_anno`(65-104 行)两种输出形态:
  - **training=True**:扁平元组列表 `[(pid, image_id, img_path, caption)]`,**每条 caption 一条记录**(同一张图的多条描述共享同一 image_id;image_id 是按图顺序 0,1,2... 的计数器)。**`pid = int(anno['id']) - 1` 让 pid 从 0 开始**(CUHK 原始 id 从 1 开始)。78-80 行断言 `idx == pid`:pid 必须从 0 连续,否则 assert 报错(所以 train_id_container 虽然是 set,遍历等价于 0..N-1,`num_classes = len(container)`,分类头维度刚好覆盖)。
  - **training=False**(test/val):dict `{'image_pids': […], 'img_paths': […], 'caption_pids': […], 'captions': […]}`,图像侧每图一项,文本侧每描述一项,**此处 pid 不减 1**(原始 id)。两侧 pid 只用于评测时判同身份,不对称无影响。
- 图片字段名:`anno['file_path']`。

### icfgpedes.py / rstpreid.py —— 与 cuhkpedes 的全部差异

| 差异点 | CUHK-PEDES | ICFG-PEDES | RSTPReid |
|---|---|---|---|
| 目录/标注名 | `CUHK-PEDES/reid_raw.json` | `ICFG-PEDES/ICFG-PEDES.json` | `RSTPReid/data_captions.json` |
| 图片路径字段 | `anno['file_path']` | `anno['file_path']` | **`anno['img_path']`** |
| train pid 处理 | **`id - 1`** | `int(id)` | `int(id)` |
| 划分 | train/test/val | train/test(json 无 val,val 为空) | train/test/val |
| 规模 | 13,003 id | 4,102 id | 4,101 id |

其余逐行相同(包括 `idx == pid` 断言——所以 ICFG/RSTPReid 的训练 id 也必须从 0 连续,否则 assert 失败;这是换数据集时最容易踩的雷)。

### cuhkpedes_badvision.py(129 行)—— 下错数据集格式的留档适配器(用户新增,不参与训练)

未注册进 `build.py` 的 `__factory`,纯留档。它适配的是**当初误下载的另一版 CUHK-PEDES 组织格式**,与 cuhkpedes.py 的全部差异:
1. 标注读 `caption_all.json`(官方发布包是 `reid_raw.json`);
2. 切分不看 `anno['split']`,而是看 `file_path` 的**上级文件夹名**(`train_query`→train / `test_query`→test / 其余→val);
3. train 的 pid 需要**先收集全部 raw id、sorted 后建 pid2label 重映射**(因为这种格式的 id 不保证 `id-1` 后从 0 连续,原版 `assert idx == pid` 会炸;对应代码注释掉了断言);
4. `_check_before_run` 额外检查 `imgs/` 下必须有 `cam_a/cam_b/CUHK01/CUHK03/Market/test_query/train_query` 七个子目录(那是错误数据集的目录结构)。

**学习价值**:对照读它和 cuhkpedes.py,正好覆盖"拿到一个新标注格式怎么写解析器"的全部考点——切分依据、pid 重映射、前置校验。

---

## 5. preprocessing.py —— 死代码
自定义 `RandomErasing` 类(Zhong et al. 论文实现,带 `probability=0.5` 概率门)。`build_transforms` 实际用的是 `torchvision.transforms.RandomErasing`(p=1)。**本文件未被任何地方 import**,保留自 ReID 代码库传统,可忽略。

## 6. sampler.py —— RandomIdentitySampler(PK 采样,非默认)

仅在 `--sampler identity` 时使用。算法(`batch_size=64, num_instance=4` 为例,每 batch = 16 个 id × 每 id 4 个实例):
- `__init__`(17-35 行):建 `pid → 样本下标列表` 的倒排 `index_dic`;估算 epoch 长度:每 pid 的样本数向下取整到 K 的倍数(不足 K 的按 K 计,即过采样补齐)。
- `__iter__`(37-63 行):每个 pid 的下标洗牌后切成 K 个一组;循环随机抽 `num_pids_per_batch` 个 pid,各取一组,拼成 batch 下标;pid 的组用完就移出候选。
- 作用:保证 batch 内**必有**同 id 正样本(对照 SDM 的 batch 内标签分布)。默认 random sampler 则无此保证(CUHK 训练集平均每 id 约 6.2 条描述,batch 64 里撞出同 id 对的概率不高,多数样本的 SDM 目标退化为只对齐自己)。

## 7. sampler_ddp.py —— DDP 版 PK 采样器(用户已修复,identity+多卡可用)

- 12-36 行 `_get_global_gloo_group`:NCCL 训练下另建一个 gloo 组(CPU 通信,传任意对象)。
- 22-97 行 `all_gather`(**通用对象版本**):pickle 序列化 → ByteTensor → 按最大长度 padding → `dist.all_gather` → 反序列化。这是 detectron2 的经典实现,与 `utils/comm.py` 里同名函数重复了一份。
- 99-109 行 `shared_random_seed`:各 rank 各自随机,all_gather 后**都取第 0 个** → 全体一致(所有 rank 必须都调用,否则死锁)。
- `RandomIdentitySampler_DDP`(111-199 行):与单机版同样的 PK 逻辑,但用共享 seed 保证各卡采样出**同一份**下标列表,再由 `__fetch_current_node_idxs`(159-168 行)按"块交错"方式切给各 rank——第 i 个全局 batch 的第 rank 份分给该 rank,使每卡的 mini-batch 仍是完整 PK 结构。
- **用户的修复**(11 §2.1-5):官方版 seed 固定(`seed = shared_random_seed()`,148 行),60 个 epoch 每 epoch 采出**完全相同**的 batch 序列,且没人调 set_epoch;调试版改为 `seed = shared_random_seed() + self.epoch`(148 行)并新增 `set_epoch(epoch)`(197-199 行),processor.py:44-47 在每个 epoch 开头调用 → 每个 epoch 重新洗牌。配合 build.py 分支补全(§2),`--sampler identity` + 多卡现在是完整可用的。
- 默认 `--sampler random` 时这套东西完全不参与(RandomSampler 没有 set_epoch,processor 的 hasattr 判断自动跳过)。

---

## 8. 本模块要记住的五件事

1. batch 是 dict:images[B,3,384,128] / caption_ids[B,77] / pids[B] / image_ids[B] / mlm_ids[B,77] / mlm_labels[B,77]。
2. **共享内存别名**:开 MLM 时 caption_ids 也会变成掩码序列(§3.3 的 ⚠️),全局分支编码的是掩码文本;这是理解/复现/改进本代码必须知道的一点(复现已证实不影响论文数字,11 §1)。
3. MLM 掩码参数(15%/80/10/10)硬编码,`--masked_token_rate*` 两个选项是死代码;padding=0 同时充当 mlm_labels 的 ignore 标记。
4. 数据集三件套仅路径字段、pid 偏移、json 文件名不同;换新数据集照抄一个解析器 + 注册 `__factory` 即可,但训练 id 必须从 0 连续(assert 强制,不满足就学 badvision 适配器建 pid2label 重映射)。
5. DDP+identity 分支用户已补全(此前是官方 TODO);preprocessing.py 仍是死代码;默认 random sampler + shuffle。
