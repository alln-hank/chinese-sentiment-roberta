# ============================================================================
# src/generate_hard_samples_v2.py
# 目的：生成包含“弱正面/中性词汇 + 转折 + 强负面”的困难样本，
#      解决模型对“还行”、“还可以”等词的过度敏感问题。
# ============================================================================

import pandas as pd
import random

# ------------------------------ 正面/中性模板（扩充了弱正面表达）------------------------------
positive_templates = [
    # 原有强正面模板
    "环境很好", "房间很干净", "早餐很丰富", "位置很方便", "价格很实惠",
    "服务态度不错", "设施很新", "景色很美", "交通便利", "性价比高",
    "大堂很气派", "前台很热情", "床很舒服", "卫生做得不错", "周边配套齐全",
    # 新增弱正面/中性模板（关键补充）
    "服务还行", "环境还可以", "价格一般", "位置勉强", "设施尚可",
    "卫生说得过去", "早餐马马虎虎", "整体凑合", "性价比一般", "房间不大",
    "也就那样", "没啥特别的", "中规中矩", "不功不过", "勉勉强强"
]

# ------------------------------ 强负面模板（保持原样）------------------------------
negative_templates = [
    "服务态度极差", "隔音太差了", "卫生条件糟糕", "空调不制冷", "热水不热",
    "床品有异味", "前台态度恶劣", "周边环境嘈杂", "早餐很难吃", "设施老旧",
    "卫生间漏水", "电梯太慢", "有蟑螂", "网络连不上", "停车不方便",
    "房间有霉味", "被子不干净", "半夜吵得睡不着", "前台爱答不理"
]

# ------------------------------ 转折连接词 ------------------------------
conjunctions = ["，但是", "，然而", "，可是", "，不过", "，就是", "，偏偏"]

# ------------------------------ 生成样本（500 条）------------------------------
hard_samples = []
num_samples = 500  # 生成 500 条

for _ in range(num_samples):
    pos = random.choice(positive_templates)
    neg = random.choice(negative_templates)
    conj = random.choice(conjunctions)
    text = pos + conj + neg
    hard_samples.append({"clean_review": text, "label": 0})  # 整体情感为负面

# 保存
hard_df = pd.DataFrame(hard_samples)
output_path = "../data/processed/hard_negative_samples_v2.csv"
hard_df.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"✅ 已生成 {len(hard_df)} 条扩展版转折型负面样本（含弱正面/中性前缀）")
print(f"保存路径：{output_path}")
print("\n样本示例（含弱正面词）：")
# 筛选几条包含弱正面词的样本展示
weak_examples = [s for s in hard_samples if any(w in s['clean_review'] for w in ['还行', '还可以', '一般', '勉强', '尚可', '凑合'])]
for i in range(min(5, len(weak_examples))):
    print(f"  {weak_examples[i]['clean_review']}")