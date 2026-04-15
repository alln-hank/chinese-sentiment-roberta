# ============================================================================
# src/data_augmentation.py
# 目的：通过对训练数据进行简单增强，提升模型的泛化能力和鲁棒性
# ============================================================================

import pandas as pd
import random
from tqdm import tqdm

# ------------------------------ 1. 加载原始数据 ------------------------------
print("正在加载原始清洗数据...")
df = pd.read_csv("../data/processed/cleaned_reviews.csv")
print(f"原始数据形状：{df.shape}")


# ------------------------------ 2. 定义文本增强函数 ------------------------------
def simple_augment(text, strategy='swap'):
    """
    对中文文本进行简单的字符级扰动，模拟真实场景中的错别字或语序变化。
    这种轻量级增强能有效防止模型过拟合到特定字符组合。

    参数:
        text: 原始文本字符串
        strategy: 增强策略，可选 'swap'（交换相邻字符）或 'drop'（随机删除一个字符）
    返回:
        增强后的文本字符串
    """
    if not isinstance(text, str) or len(text) < 3:
        return text  # 太短的文本不处理，避免变成空串

    chars = list(text)  # 将字符串拆成字符列表，便于修改

    if strategy == 'swap':
        # 随机选择一个位置，交换它和下一个字符的位置
        idx = random.randint(0, len(chars) - 2)
        chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
    elif strategy == 'drop':
        # 随机删除一个字符
        idx = random.randint(0, len(chars) - 1)
        chars.pop(idx)
    else:
        return text

    return ''.join(chars)


# ------------------------------ 3. 只对正面样本做增强 ------------------------------
# 原因：你的数据显示正面样本（5322）远多于负面（2443），适度增加正面样本的多样性，
# 既能平衡样本，又能让模型学到正面表达的更多变体，减少过拟合。

positive_df = df[df['label'] == 1].copy()
print(f"正面样本数：{len(positive_df)}")

augmented_data = []
# 使用 tqdm 显示进度条
for _, row in tqdm(positive_df.iterrows(), total=len(positive_df), desc="正在增强正面样本"):
    text = row['clean_review']
    # 为每条正面评论生成 2 条增强样本（1条交换，1条删除）
    aug_text_1 = simple_augment(text, strategy='swap')
    aug_text_2 = simple_augment(text, strategy='drop')

    augmented_data.append({'clean_review': aug_text_1, 'label': 1})
    augmented_data.append({'clean_review': aug_text_2, 'label': 1})

# ------------------------------ 4. 合并原始数据与增强数据 ------------------------------
aug_df = pd.DataFrame(augmented_data)
print(f"生成的增强样本数：{len(aug_df)}")

final_df = pd.concat([df, aug_df], ignore_index=True)
# 打乱数据顺序，防止训练时连续出现大量增强样本
final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"增强后数据集总样本数：{len(final_df)}")
print(f"增强后标签分布：\n{final_df['label'].value_counts()}")

# ------------------------------ 5. 保存增强后的数据集 ------------------------------
output_path = "../data/processed/augmented_reviews.csv"
final_df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"\n增强数据集已保存至：{output_path}")