"""
第一个可视化脚本：看看图片和文字是怎么变成数字的
运行方式：python 01_see_tokens.py
"""

import torch
from PIL import Image
import clip
from clip.model import LayerNorm

# ============================================================
# 第 0 步：加载 CLIP 预训练模型
# ============================================================
# ViT-B/16 是 IRRA 用的同一个 backbone
# "这行代码就是在加载一个已经训练好的模型，就像你打开一个现成的工具"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"用设备: {device}")

# 加载模型和预处理方式
model, preprocess = clip.load("ViT-B/16", device=device)
model.eval()  # 切换到"推理模式"（不训练，只看结果）

print("\n=== 模型加载完成 ===")
print(f"模型类型: {type(model)}")


# ============================================================
# 第 1 步：准备一张图片
# ============================================================
print("\n\n========== 第一步：看图片怎么变成数字 ==========")

# 找一张你自己的行人图片（改成你自己的图片路径）
# 如果没有，就随便找一张
image_path = "my_dataset_root/CUHK-PEDES/imgs/0001_c1s1_000151_00.jpg"  # ← 改成你自己的图

try:
    image = Image.open(image_path)
    print(f"原始图片大小: {image.size}  (宽 × 高)")
except:
    print("没找到示例图片，用随机生成的假图代替")
    image = Image.new("RGB", (128, 384), color=(128, 100, 80))

# 预处理：把图片变成模型能吃的格式
# 这一步就是：缩放到模型要求的尺寸 + 归一化像素值
image_input = preprocess(image).unsqueeze(0).to(device)  # unsqueeze(0) = 加一个"批次"维度

print(f"预处理后图片形状: {image_input.shape}")
print(f"  → 翻译成人话: 1 张图 × 3 个颜色通道(RGB) × 高 {image_input.shape[2]} × 宽 {image_input.shape[3]}")
print(f"  → 每个数字的范围: 大概 -2 ~ +2 之间（因为做了归一化）")
print(f"  → 看几个像素值: {image_input[0, 0, 0, :5].tolist()}")  # 看第一个像素的前 5 个值


# ============================================================
# 第 2 步：看图像编码器中间过程
# ============================================================
print("\n\n========== 第二步：图像编码器里发生了什么 ==========")

with torch.no_grad():  # no_grad = 不计算梯度（我们不训练，只看）
    # 2.1: 先过 patch embedding（把图片切成小块）
    x = image_input
    print(f"\n输入图像形状: {x.shape}  [批次, 通道, 高, 宽]")
    
    # 调用 CLIP 的视觉模型的前半部分
    # 我们手动拆开看每一步
    visual = model.visual
    
    # patch embedding：把图片切成 16×16 的小块，每块变成一个向量
    x = visual.conv1(x)  # 卷积层做 patch embedding
    print(f"经过卷积(patch切分)后: {x.shape}")
    print(f"  → 翻译: 1 张图 × 768 个数字/块 × 24 个块(高方向) × 8 个块(宽方向)")
    print(f"  → 意思: 原图被切成了 24×8 = 192 个小块，每块用 768 个数字表示")
    
    # 整理形状
    x = x.reshape(x.shape[0], x.shape[1], -1)  # [1, 768, 192]
    x = x.permute(0, 2, 1)  # [1, 192, 768]
    print(f"整理形状后: {x.shape}  [批次, patch数量, 每块的维度]")
    
    # 加 CLS token（一个特殊符号，代表"整张图"）
    cls_token = visual.class_embedding.unsqueeze(0).unsqueeze(0).expand(x.shape[0], -1, -1)
    x = torch.cat([cls_token, x], dim=1)
    print(f"加上CLS符号后: {x.shape}")
    print(f"  → 翻译: 现在有 193 个 token 了（1 个 CLS + 192 个图像块）")
    print(f"  → 这就是 IRRA 论文里说的 '193 个图像 token'")
    
    # 加位置编码（告诉模型每个小块在图片的哪个位置）
    x = x + visual.positional_embedding.unsqueeze(0)
    print(f"加上位置编码后: {x.shape}  (形状没变，只是每个 token 里的数字变了)")
    
    # 过 Transformer（12 层注意力）
    x = visual.ln_pre(x)
    x = x.permute(1, 0, 2)  # [193, 1, 768] Transformer 要求的形状
    x = visual.transformer(x)
    x = x.permute(1, 0, 2)  # [1, 193, 768]
    x = visual.ln_post(x)
    print(f"经过12层Transformer后: {x.shape}")
    print(f"  → 翻译: 还是 193 个 token，但每个 token 不再只是"这块的像素"了")
    print(f"  → 经过注意力机制后，每个 token 都"看过"了其他所有块")
    print(f"  → CLS token(第0个) 现在代表了"整张图的整体信息"")
    
    # 投影到 512 维（和文本对齐的空间）
    x = x @ visual.proj
    print(f"投影到联合空间后: {x.shape}")
    print(f"  → 翻译: 每个 token 从 768 维变成了 512 维")
    print(f"  → 这个 512 维就是和文本对齐的空间")
    
    # 拿出 CLS token（全局图像特征）
    image_global_feature = x[:, 0, :]  # 第 0 个 token 就是 CLS
    print(f"\n全局图像特征(CLS): {image_global_feature.shape}")
    print(f"  → 翻译: 1 张图的整体表征，用 512 个数字表示")
    print(f"  → 前 10 个数字: {image_global_feature[0, :10].tolist()}")


# ============================================================
# 第 3 步：看文字怎么变成数字
# ============================================================
print("\n\n========== 第三步：看一句话怎么变成数字 ==========")

# 一句话（随便写一句行人描述）
text = "a man in a blue jacket and black pants"
print(f"输入文字: {text}")

# 分词：把句子切成一个个词/字
# CLIP 的分词器会把句子变成一串 token ID
text_tokens = clip.tokenize([text]).to(device)
print(f"\n分词后形状: {text_tokens.shape}  [批次, 序列长度]")
print(f"  → 翻译: 1 句话，固定长度 77（不够补零，太长截断）")
print(f"  → 每个数字代表一个词，看前 20 个: {text_tokens[0, :20].tolist()}")
print(f"  → 第一个数字 49406 = <|startoftext|> 句子开头")
print(f"  → 最后一个非零 49407 = <|endoftext|> 句子结尾")

with torch.no_grad():
    # 过文本编码器
    text_features = model.encode_text(text_tokens)
    print(f"\n文本特征形状: {text_features.shape}")
    print(f"  → 翻译: 1 句话的表征，512 个数字")
    print(f"  → 前 10 个数字: {text_features[0, :10].tolist()}")


# ============================================================
# 第 4 步：看看图像和文本有多"像"
# ============================================================
print("\n\n========== 第四步：算相似度 ==========")

# L2 归一化（把向量长度变成 1，这样点积就是余弦相似度）
image_feature_norm = image_global_feature / image_global_feature.norm(dim=-1, keepdim=True)
text_feature_norm = text_features / text_features.norm(dim=-1, keepdim=True)

# 算余弦相似度
similarity = (image_feature_norm @ text_feature_norm.T).item()
print(f"这张图和这句话的相似度: {similarity:.4f}")
print(f"  → 范围是 -1 ~ +1，越接近 1 越像")
print(f"  → 如果换一张不相关的图，相似度会低很多")


# ============================================================
# 总结
# ============================================================
print("\n\n" + "="*60)
print("总结：数据从输入到输出的形状变化")
print("="*60)
print("""
图片: [1, 3, 384, 128]     ← 原始像素（RGB 三通道）
   ↓ 切成 patch
    [1, 192, 768]           ← 192 个小块，每块 768 维
   ↓ 加 CLS + 位置编码
    [1, 193, 768]           ← 193 个 token（1 CLS + 192 patch）
   ↓ 过 12 层 Transformer 注意力
    [1, 193, 768]           ← 形状不变，但每个 token 都看过了全图
   ↓ 投影到 512 维
    [1, 193, 512]           ← 这就是 IRRA 里说的"193 个图像 token"
   ↓ 取 CLS（第 0 个）
    [1, 512]                ← 全局图像特征


文字: [1, 77]               ← 77 个词 ID
   ↓ 查 embedding 表
    [1, 77, 512]            ← 每个词 512 维
   ↓ 过 12 层 Transformer 注意力（因果掩码）
    [1, 77, 512]            ← 每个词都融合了上下文
   ↓ 取 EOT（句子结尾位置）
    [1, 512]                ← 全局文本特征


最后: 图像特征 [1, 512] 和 文本特征 [1, 512] 点积
   → 一个分数（相似度）
""")
