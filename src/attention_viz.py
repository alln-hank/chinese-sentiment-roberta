import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ------------------------------------------------------------
# 1. 加载模型（必须开启 attention 输出）
# ------------------------------------------------------------
model_path = "../models/full_ft/best_model"   # 使用全参数微调的模型，你也可以换成 lora_ft

print("正在加载分词器...")
tokenizer = AutoTokenizer.from_pretrained(model_path)

print("正在加载模型（开启 attention 输出）...")
model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    output_attentions=True   # 关键参数：让模型在前向传播时返回每一层的注意力权重
)
model.eval()  # 切换到评估模式，关闭 Dropout 等训练专用层

# ------------------------------------------------------------
# 2. 准备输入文本（选一句包含转折的评论，便于观察注意力变化）
# ------------------------------------------------------------
text = "酒店环境很好，但是服务态度极差"
print(f"输入文本：{text}")

# 分词并转换为张量
inputs = tokenizer(text, return_tensors="pt")

# 获取 token 列表，用于后续热力图的坐标轴标签
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
print(f"分词结果：{tokens}")

# ------------------------------------------------------------
# 3. 前向传播，获取注意力权重
# ------------------------------------------------------------
with torch.no_grad():                # 禁用梯度计算，节省显存，加快推理速度
    outputs = model(**inputs)        # 将 inputs 字典解包后传入模型
    attentions = outputs.attentions  # attentions 是一个 tuple，长度 = 模型层数（BERT-base 为 12 层）

print(f"模型层数：{len(attentions)}")
print(f"第 0 层注意力张量形状：{attentions[0].shape}")
# 形状解释：(batch_size, num_heads, seq_len, seq_len) = (1, 12, 序列长度, 序列长度)

# ------------------------------------------------------------
# 4. 选择某一层、某一个注意力头，绘制热力图
# ------------------------------------------------------------
layer = 8      # 选择第 8 层（0~11 任选，中间层通常语义信息较丰富）
head = 0       # 选择第 0 个注意力头（共 12 个头，任选一个）

# 取出对应层和头的注意力矩阵，并转换为 NumPy 数组
attn_matrix = attentions[layer][0, head].numpy()   # 形状：(seq_len, seq_len)

# ============================================================================
# 5. 终极美化版热力图绘制
# ============================================================================
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体（防止乱码）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建画布（宽高比适当，标签不拥挤）
plt.figure(figsize=(14, 12))

# 核心绘图：绝对不显示数字（annot=False），线条分割清晰
sns.heatmap(
    attn_matrix,
    xticklabels=tokens,
    yticklabels=tokens,
    cmap='RdBu_r',               # 红蓝渐变，白色为零点，红色为高权重，蓝色为低权重
    center=0.05,                 # 将颜色中心设在略高于0的位置，突出有效注意力
    square=True,
    linewidths=0.5,              # 网格线宽度
    linecolor='gray',            # 网格线颜色
    cbar_kws={"shrink": 0.8, "label": "注意力权重", "pad": 0.02},
    annot=False,                 # 坚决不显示数字
    fmt='.2f'                    # 即使 annot=False 也保留此参数，无影响
)

# 优化坐标轴标签
plt.xticks(rotation=45, ha='right', fontsize=11)
plt.yticks(rotation=0, fontsize=11)

# 标题与轴标签
plt.title(f'Transformer 自注意力可视化 (第 {layer} 层, 第 {head} 头)\n'
          '颜色越红表示注意力权重越高，越蓝表示越低',
          fontsize=14, pad=20)
plt.xlabel('Key (被关注的词)', fontsize=12, labelpad=10)
plt.ylabel('Query (发起关注的词)', fontsize=12, labelpad=10)

# 自动调整布局，防止标签被切
# 替换原来的 plt.tight_layout()
plt.subplots_adjust(top=0.92, bottom=0.12, left=0.08, right=0.98)

# 保存高清图片
output_path = '../outputs/attention_heatmap_final.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

print(f"✅ 终极版热力图已保存至 {output_path}")
print("请去 outputs 文件夹双击打开查看效果。")