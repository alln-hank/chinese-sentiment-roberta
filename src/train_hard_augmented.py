# ============================================================================
# src/train_hard_augmented.py
# 最终版训练脚本：
# - 加载增强数据 + 转折困难样本 v1 + 转折困难样本 v2
# - 使用 Focal Loss (gamma=1.0)
# - 输出模型到 models/full_ft_hard_augmented_v2
# ============================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from dataset import SentimentDataset

# ------------------------------ 1. 加载并合并所有数据 ------------------------------
print("正在加载增强数据...")
df_aug = pd.read_csv("../data/processed/augmented_reviews.csv")
print(f"增强数据样本数：{len(df_aug)}")

print("正在加载转折困难样本 v1（强正面前缀）...")
df_hard1 = pd.read_csv("../data/processed/hard_negative_samples.csv")
print(f"v1 样本数：{len(df_hard1)}")

print("正在加载转折困难样本 v2（含弱正面/中性前缀）...")
df_hard2 = pd.read_csv("../data/processed/hard_negative_samples_v2.csv")
print(f"v2 样本数：{len(df_hard2)}")

# 合并所有困难样本
df_hard = pd.concat([df_hard1, df_hard2], ignore_index=True)

# 合并增强数据与困难样本
df = pd.concat([df_aug, df_hard], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"合并后总样本数：{len(df)}")
print(f"标签分布：\n{df['label'].value_counts()}")

texts = df['clean_review'].tolist()
labels = df['label'].tolist()

# ------------------------------ 2. 划分训练/验证集 ------------------------------
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)
print(f"训练集：{len(train_texts)}，验证集：{len(val_texts)}")

# ------------------------------ 3. 加载分词器与模型 ------------------------------
MODEL_NAME = "bert-base-chinese"
print("正在加载分词器...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print("正在加载预训练模型...")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# ------------------------------ 4. 超参数 ------------------------------
MAX_LEN = 273
BATCH_SIZE = 16
EPOCHS = 3
OUTPUT_DIR = "../models/full_ft_hard_augmented_v2"  # 新输出目录

# ------------------------------ 5. 构建 Dataset ------------------------------
train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, MAX_LEN)
val_dataset = SentimentDataset(val_texts, val_labels, tokenizer, MAX_LEN)

# ------------------------------ 6. Focal Loss Trainer (gamma=1.0) ------------------------------
class FocalLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        gamma = 1.0  # 使用 1.0 降低对噪声的过度聚焦
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** gamma * ce_loss).mean()

        return (focal_loss, outputs) if return_outputs else focal_loss

# ------------------------------ 7. 评估指标 ------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='binary')
    return {'accuracy': acc, 'f1': f1}

# ------------------------------ 8. 训练参数 ------------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir="../logs",
    logging_steps=20,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=torch.cuda.is_available(),
    dataloader_pin_memory=False,
    report_to="tensorboard"
)

# ------------------------------ 9. 开始训练 ------------------------------
trainer = FocalLossTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

print("\n开始训练（增强数据 + 转折样本 v1+v2 + Focal Loss gamma=1.0）...")
trainer.train()

# ------------------------------ 10. 保存最佳模型 ------------------------------
trainer.save_model(f"{OUTPUT_DIR}/best_model")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/best_model")
print(f"\n 训练完成！最佳模型已保存至 {OUTPUT_DIR}/best_model")