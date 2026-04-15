import pandas as pd

# 1. 读取清洗后的数据
df = pd.read_csv("../data/processed/cleaned_reviews.csv")

# 2. 统计标签分布
print("===== 标签分布 =====")
print(df['label'].value_counts())

# 3. 计算文本长度
df['text_len'] = df['clean_review'].apply(len)

print("\n===== 文本长度统计 =====")
print(f"平均长度：{df['text_len'].mean():.2f}")
print(f"最大长度：{df['text_len'].max()}")
print(f"90％ 分位数：{df['text_len'].quantile(0.9):.0f}")
print(f"95 ％分位数：{df['text_len'].quantile(0.95):.0f}")
