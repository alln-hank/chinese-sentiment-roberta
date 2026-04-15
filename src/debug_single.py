# ============================================================================
# src/debug_single.py
# 作用：加载指定模型，对单条评论文本进行情感预测，用于快速验证模型效果
# ============================================================================

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# ------------------------------ 1. 指定模型路径（根据你要测试的模型修改） ------------------------------
# 你可以在这里切换不同的模型路径，对比效果
MODEL_PATH = "../models/full_ft_hard_augmented/best_model"   # 方案一训练出的最新模型

# ------------------------------ 2. 加载分词器和模型 ------------------------------
print(f"正在加载模型：{MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()  # 切换到评估模式

# ------------------------------ 3. 定义要测试的文本 ------------------------------
# 你可以修改这里的文本，测试任意句子
test_texts = [
    "环境很好，但是服务态度极差，不会再住了。",
    "酒店房间很干净，早餐也很丰富，下次还会来。",
    "隔音太差了，隔壁说话听得一清二楚，一夜没睡好。",
]

# ------------------------------ 4. 批量预测并输出结果 ------------------------------
for text in test_texts:
    # 分词并转换为张量（参数必须与训练时保持一致）
    inputs = tokenizer(
        text,
        max_length=273,           # 与训练时一致
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    # 推理
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1).squeeze().tolist()

    # 输出结果
    print("\n" + "=" * 60)
    print(f"评论文本：{text}")
    print(f"正面概率：{probs[1]:.4f} ({probs[1]*100:.2f}%)")
    print(f"负面概率：{probs[0]:.4f} ({probs[0]*100:.2f}%)")
    print(f"预测结果：{'正面' if probs[1] > probs[0] else '负面'}")