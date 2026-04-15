import pandas as pd
import os

# 1. 定义文件路径（相对路径）
raw_data_path = "../data/raw/ChnSentiCorp_htl_all.csv"
processed_data_path = "../data/processed/cleaned_reviews.csv"

# 2. 创建保存目录
os.makedirs("../data/processed", exist_ok=True)

# 3. 读取原始数据
print("正在读取原始数据...")
df = pd.read_csv(raw_data_path)
print(f"原始数据形状：{df.shape}")

# 4. 查看列名（确认是否为‘label’和‘review’）
print("列名", df.columns.tolist())

# 5. 简单清洗函数
def clean_text(text):
    if not isinstance(text,str):
        return ""
    #去掉换行、回车、多余空格
    text = text.replace('\n', '').replace('\r', '').strip()
    #去掉连续多个空格
    import re
    text = re.sub(r'\s+', '', text)
    return text

# 6. 应用清晰
print("正在清晰文本...")
df['clean_review'] = df['review'].apply(clean_text)

# 7. 删除清洗后为空的样本
df = df[df['clean_review'] != ""]

# 8. 保存清洗后的数据
df.to_csv(processed_data_path, index=False, encoding='utf-8-sig')
print(f"清洗完成！处理后数据形状为：{df.shape}")
print(f"文件已保存至：{processed_data_path}")
