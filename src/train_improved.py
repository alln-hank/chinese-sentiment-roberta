#   结合数据增强 + Focal Loss的改进训练脚本

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score
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

# 1. 加载增强后的数据
print("正在加载增强后的数据...")
df = pd.read_csv("../data/processed/augmented_reviews.csv")
texts = df['clean_review'].tolist()
labels = df['label'].tolist()

# 2. 划分训练/测试集
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)
print(f"训练集样本数：{len(train_texts)}")
print(f"验证集样本数：{len(val_texts)}")

# 3. 加载分词器与模型
MODEL_NAME = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# 4. 超参数设置
MAX_lEN = 273
BATCH_SIZE = 16
EPOCHS = 3
OUTPUT_DIR = "../models/full_ft_augmented_focal"    #新保存路径

# 5. 构建Dataset
train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, MAX_lEN)
val_dataset = SentimentDataset(val_texts, val_labels, tokenizer, MAX_lEN)

# 6. 自定义Trainer(继承Focal Loss)
class FocalLossTrainer(Trainer):
    """
    继承HuggingFace Trainer, 重写compute_loss方法以使用Focal Loss
    Focal Loss通过降低易分类样本的权重，时模型更聚焦于难分类样本。
    """
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")       # 取出标签，并从inputs中移除
        outputs = model(**inputs)           #向前传播
        logits = outputs.logits             #获取logits

        #计算标准交叉熵损失（不进行reduction，保留每个样本的损失值）
        ce_loss = F.cross_entropy(logits, labels, reduction='none')

        #计算Focal Loss
        gamma = 2.0                                         #聚焦参数，值越大对难样本的权重越高
        pt = torch.exp(-ce_loss)                            #预测正确的概率p_t
        focal_loss = ((1-pt) ** gamma * ce_loss).mean()     #对batch取平均

        return(focal_loss, outputs) if return_outputs else focal_loss

# ------------------------------ 7. 评估指标 ------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='binary')
    return {'accuracy': acc, 'f1': f1}

# ------------------------------ 8. 训练参数配置 ------------------------------
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

# ------------------------------ 9. 创建 Trainer 并开始训练 ------------------------------
trainer = FocalLossTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

print("\n开始训练（数据增强 + Focal Loss）...")
trainer.train()

# ------------------------------ 10. 保存最佳模型 ------------------------------
trainer.save_model(f"{OUTPUT_DIR}/best_model")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/best_model")
print(f"\n训练完成！最佳模型已保存至 {OUTPUT_DIR}/best_model")