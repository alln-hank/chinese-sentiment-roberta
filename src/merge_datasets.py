# src/merge_datasets.py
import pandas as pd
import os
import glob

def clean_text(text):
    """与 data_preprocess.py 中保持一致的清洗逻辑"""
    if not isinstance(text, str):
        return ""
    text = text.replace('\n', '').replace('\r', '').strip()
    import re
    text = re.sub(r'\s+', ' ', text)
    return text

def load_and_standardize(file_path, text_col, label_col, label_map):
    """加载单个CSV文件，并统一列名和标签格式"""
    try:
        df = pd.read_csv(file_path)
        # 1. 重命名列
        df = df.rename(columns={text_col: 'review', label_col: 'label'})
        # 2. 映射标签为 0 或 1
        df['label'] = df['label'].map(label_map)
        # 3. 剔除标签映射后为NaN的行（即非0或1的无效标签）
        df = df.dropna(subset=['label'])
        df['label'] = df['label'].astype(int)
        # 4. 清洗文本
        df['clean_review'] = df['review'].apply(clean_text)
        # 5. 删除清洗后为空的样本
        df = df[df['clean_review'] != ""]
        return df[['clean_review', 'label']]
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        return pd.DataFrame(columns=['clean_review', 'label'])

# 确保 processed 目录存在
os.makedirs("../data/processed", exist_ok=True)

# --- 配置各个数据集的信息 ---
# 每个元组：(文件路径, 文本列名, 标签列名, 标签映射字典)
datasets_config = [
    ("../data/raw/ChnSentiCorp_htl_all.csv", "review", "label", {1: 1, 0: 0}),
    ("../data/raw/online_shopping_10_cats.csv", "review", "label", {1: 1, 0: 0}),
    ("../data/raw/waimai_10k.csv", "review", "label", {1: 1, 0: 0}),
    ("../data/raw/weibo_senti_100k.csv", "review", "label", {1: 1, 0: 0})
]

# 对于 .zip 文件，需要先解压
zip_files = glob.glob("../data/raw/*.zip")
if zip_files:
    for zip_file in zip_files:
        try:
            print(f"正在解压 {zip_file}...")
            # 假设解压后的CSV文件名与zip文件同名
            csv_name = os.path.basename(zip_file).replace('.zip', '.csv')
            df_temp = pd.read_csv(zip_file, compression='zip')
            df_temp.to_csv(f"../data/raw/{csv_name}", index=False, encoding='utf-8-sig')
        except Exception as e:
            print(f"解压失败: {e}")

# 加载并合并所有数据
all_dfs = []
for file_path, text_col, label_col, label_map in datasets_config:
    if os.path.exists(file_path):
        print(f"正在处理: {os.path.basename(file_path)}")
        df_temp = load_and_standardize(file_path, text_col, label_col, label_map)
        if not df_temp.empty:
            all_dfs.append(df_temp)
            print(f"  -> 读取到 {len(df_temp)} 条数据")
    else:
        print(f"警告：文件不存在 - {file_path}")

if not all_dfs:
    print("没有找到任何数据集，请检查文件路径。")
    exit()

final_df = pd.concat(all_dfs, ignore_index=True)
# 打乱数据
final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

output_path = "../data/processed/large_reviews.csv"
final_df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"\n✅ 合并完成！总数据量: {len(final_df)} 条，已保存至 {output_path}")
print(f"标签分布：\n{final_df['label'].value_counts()}")