# 10 二次开发指南(扩展点 · 改动位置 · 实验建议)

面向"在 IRRA 上改进方法/做实验"的场景。每个条目给出**精确到文件与行的改动位置**、注意事项(坑清单见 09 文档 §3,不重复)。

---

## 0. 环境与跑通(前置)

- 依赖(无 requirements.txt,按 import 整理):`pytorch(1.9)+torchvision、prettytable、easydict、pyyaml、numpy、ftfy、regex、tqdm、Pillow、matplotlib(仅 visualize)、tensorboard`。需要 **CUDA**(`device="cuda"` 硬编码)。
- CLIP 权重首次训练自动下载到 `~/.cache/clip`;离线可先放好 `ViT-B-16.pt`。
- 数据:CUHK-PEDES / ICFG-PEDES / RSTPReid 三个数据集已全部就位在 `IRRA/my_dataset_root/`(目录布局见 03 文档 §1),且已在服务器上完整训练复现成功(见 11 文档 §1)。历史教训:2026-09-16 时曾误下 CUHK-SYSU(Person Search 原始数据集,无裁剪图无文本),现已删除。
- 训练:`bash run_irra.sh`(或 02 文档的等价命令);测试:`python test.py --config_file logs/<...>/configs.yaml`。
- 复现目标:CUHK R1 73.38 / mAP 66.13;ICFG 63.46 / 38.05;RSTPReid 60.20 / 47.17。

## 1. 加一个新损失(最常见需求)

四处改动,顺序即依赖链:
1. `utils/options.py`:加权重参数(如 `--xxx_loss_weight`)、把名字加进 `--loss_names` 的 help;
2. `model/objectives.py`:写 `compute_xxx(...)`;
3. `model/build.py`:
   - `__init__` 里 `'xxx' in args.loss_names` 时建需要的头(照抄 classifier/mlm 的条件构建模式);
   - `forward` 里加分支 `ret['xxx_loss'] = compute_xxx(...) * weight`;
4. `processor/processor.py`:meters 加 `xxx_loss`(可选,纯日志)。

**命名铁律**:想被自动求和,名字必须含 `loss`;不想被求和的指标(准确率之类)千万别带。总损失在 `processor.py:51` 按子串求和。

## 2. 现成但未接线的实验开关(零成本消融)

| 实验 | 改动 | 位置 |
|---|---|---|
| SDM 软标签(同图 1.0 / 同 pid 异图 0.3) | 调用时传 `image_id=batch['image_ids']`(可再调 factor) | `model/build.py:107`(`compute_sdm(..., image_id=None)`);`compute_sdm` 内部 15-21 行已写好 |
| 修 caption 污染(全局分支用干净文本) | 掩码前拷贝:`self._build_...(caption_tokens.clone().numpy())` 或函数内先 `tokens = tokens.copy()` | `datasets/bases.py:157`。**注意:这会改变训练行为,结果不再与已发表数字对齐;消融时两组都要跑** |
| 接通掩码比例参数 | 把硬编码 0.15/0.8 换成 args | `datasets/bases.py:184-195`;args 已存在(`masked_token_rate` 等,注意语义:rate 是"选中后换 MASK 的比例"0.8,不是总掩码率) |
| MLM 冗余编码优化 | `mlm_feats` 直接复用 `text_feats`(内容相同) | `model/build.py:128`(省一次文本塔前向;若同时修了污染则不能复用) |
| identity 采样 vs random | 命令行 `--sampler identity` | 无需改码;单卡可用,DDP 分支是坏的 |
| 换对照损失 | `--loss_names 'itc'` / `'cmpm'` / 组合 | 论文 Tab.4 的所有消融行都能直接复现 |

## 3. 论文指明的改进方向(自带动机)

- **短语级掩码**:论文 Sec.4.3 明确说当前只掩单个 token、学不到短语语义,"plan to address in future work"。落点:`bases.py` 的掩码函数——先按 `pat` 正则或依存句法把连续 token 分组成短语,再整短语掩。这是最"名正言顺"的 follow-up。
- 交互编码器结构(Fig.4 的 co-attn/merged 变体):落点 `model/build.py::cross_former`。
- mINP 在 ICFG 上很低(7.92)→ 难样本挖掘类改进有空间。

## 4. 换 backbone / 改输入

- **换 ViT 型号**:`--pretrain_choice ViT-L/14`。IRRA 的新模块全部用 `embed_dim` 自适应构建(头数=embed_dim//64),ViT-L(768)结构上直接兼容;但 batch/lr/fp16 稳定性要重调,且 `solver/build.py` 的分组规则不用改。**ResNet 系列别用**(09 文档坑 #7)。
- **改分辨率/stride**:改 `utils/options.py:34-35` 的**默认值**(命令行传 `--img_size` 会因 `type=tuple` 出错);stride<patch 即重叠 patch。位置编码插值自动完成(05 文档 §6),但 `resize_pos_embed` 假设预训练网格是正方形——成立。
- **改文本长度**:必须先补 `resize_text_pos_embed`(09 文档坑 #4),否则 NameError;同时 `bases.py` 的 49405 等魔数与 tokenizer 词表结构要重新核对。

## 5. 加新数据集

1. 照抄 `datasets/cuhkpedes.py` → 改三处:目录名/标注文件名、图片路径字段名(`file_path` vs `img_path`)、pid 是否需要偏移(**训练 pid 必须从 0 连续,`_process_anno` 里的 assert 强制,不满足就建 pid 重映射**)。
2. `datasets/build.py:17` 注册 `__factory`。
3. `utils/options.py:63` help 加名字。
评测侧自动适配(query=每条描述,gallery=每张图)。

## 6. 训练管线的实用小改

- **周期性存 last.pth**:`processor.py` 的 epoch 循环里加 `if epoch % k == 0: checkpointer.save("last", **arguments)`(现在只有 best.pth,断了只能从 best 续)。
- **严格确定性**:`train.py:27` 删 `benchmark=True`。
- **监控**:TensorBoard 已记 loss/acc/lr/temperature;`mlm_acc` 是 IRR 是否在学的直接信号(应快速升到 60%+ 再缓慢爬);temperature 是恒线(0.02),动了说明有人把 logit_scale 改成了可学习。
- **调 SDM 温度**:改 `--temperature`;注意它同时作用于 itc。想让三个损失温度解耦,要在 `compute_sdm` 签名里单独传。
- **调损失权重**:只有 mlm/id 有现成参数;SDM 想加权重需在 `model/build.py:107` 手乘,或给 `compute_sdm` 加 weight 参数。

## 7. 迁移复用 IRRA 组件

- **只要双塔**(做纯检索/特征):`loss_names 'sdm'` + `--MLM` 关掉,IRR/MLM 头不会建;评测走 `encode_image/encode_text`。
- **只要 IRR 结构**(别的多模态任务):`model/build.py` 的 `cross_former` 是自包含模块(Q/K/V 三输入 + LN + MCA + 4 层 self-attn),可直接抄;初始化套路在 `__init__` 40-51 行。
- **复用 CLIP 改造**(非方形输入 + 全 token 输出):`clip_model.py` 单文件即插即用,关键是 `build_CLIP_from_openai_pretrained(name, img_size, stride)` 的形状推断,不依赖本仓库其他文件(除 LayerNorm 等同文件内类)。

## 8. 改动前必读的三条军规

1. **先读 09 文档 §2①(污染问题)再动 datasets/bases.py 或 model/build.py 的文本通路**——很多"改进"的实际效果会被这个别名行为放大或抵消。
2. **fp16 全模型**:任何新模块默认会被 `convert_weights` 转 fp16;自定义算子若对精度敏感,学 `LayerNorm` 的写法(转 fp32 计算再转回)。
3. **评测协议锁死**:`Evaluator` 用归一化余弦 + 同 pid 判正;自定义损失如果优化别的几何(如欧氏),推理口径要同步改,否则指标不动。
