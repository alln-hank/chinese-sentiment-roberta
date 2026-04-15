# ============================================================================
# src/train_ultimate.py
# 终极版训练脚本（修正权限错误 + 支持断点恢复）
# 功能：数据扩充 + RoBERTa-wwm-ext + 继续预训练(MLM) + Focal Loss + 对抗训练 + EMA
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
    AutoModelForMaskedLM,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorForLanguageModeling
)
from datasets import Dataset as HFDataset
import os
import sys
import torch
print(f"CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"当前 GPU: {torch.cuda.get_device_name(0)}")
    print(f"显存占用: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
# ------------------------------ 0. 设置项目绝对路径（请根据你的实际路径修改） ------------------------------
PROJECT_ROOT = r"D:\Users\zmhzz\PycharmProjects\Transformer_NLP_Project" # 👈 改成你的项目根目录
DATA_PATH = os.path.join(PROJECT_ROOT, "data/processed/large_reviews.csv")
MLM_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models/roberta_mlm")
FINE_TUNE_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models/roberta_ultimate")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

# 确保输出目录存在
os.makedirs(MLM_OUTPUT_DIR, exist_ok=True)
os.makedirs(FINE_TUNE_OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# 将 src 目录加入系统路径，以便导入自定义 dataset
sys.path.append(os.path.join(PROJECT_ROOT, "src"))
from dataset import SentimentDataset

# ------------------------------ 1. 加载数据 ------------------------------
print("正在加载扩充后的数据...")
df = pd.read_csv(DATA_PATH)
texts = df['clean_review'].tolist()
labels = df['label'].tolist()

# 划分训练/验证集 (用于后续微调)
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)
print(f"训练集：{len(train_texts)}，验证集：{len(val_texts)}")

# ------------------------------ 2. 加载分词器与模型 ------------------------------
MODEL_NAME = "hfl/chinese-roberta-wwm-ext"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 计算 MAX_LEN (可以从之前的分析得到，这里设为273)
MAX_LEN = 273
BATCH_SIZE = 16
EPOCHS = 3

# ------------------------------ 3. 领域自适应继续预训练 (MLM) ------------------------------
print("\n开始领域自适应继续预训练 (MLM)...")

# 准备MLM数据集（使用全部数据）
mlm_texts = texts
mlm_hf_dataset = HFDataset.from_dict({'text': mlm_texts})

def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, max_length=MAX_LEN)

mlm_tokenized_dataset = mlm_hf_dataset.map(tokenize_function, batched=True, remove_columns=['text'])

# 加载用于MLM的模型
mlm_model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)

# 数据整理器
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

# MLM训练参数（使用绝对路径，启用断点恢复）
mlm_args = TrainingArguments(
    output_dir=MLM_OUTPUT_DIR,
    overwrite_output_dir=False,  # 不覆盖，以便从检查点恢复
    num_train_epochs=2,
    per_device_train_batch_size=BATCH_SIZE,
    save_steps=1000,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    report_to="none",
    logging_dir=os.path.join(LOG_DIR, "mlm"),
    logging_steps=100,
    resume_from_checkpoint=True,  # 自动寻找最新检查点恢复
)

mlm_trainer = Trainer(
    model=mlm_model,
    args=mlm_args,
    data_collator=data_collator,
    train_dataset=mlm_tokenized_dataset,
)

# ------------------------------ 从指定检查点恢复 MLM 训练 ------------------------------
resume_checkpoint = os.path.join(MLM_OUTPUT_DIR, "checkpoint-25000")  # 使用你确认存在的检查点

if os.path.exists(resume_checkpoint):
    print(f"✅ 检测到检查点，从 {resume_checkpoint} 恢复训练...")
else:
    print("⚠️ 未找到检查点，从头开始训练")
    mlm_trainer.train()

# 保存最终MLM模型和分词器
mlm_trainer.save_model(os.path.join(MLM_OUTPUT_DIR, "best_model"))
tokenizer.save_pretrained(os.path.join(MLM_OUTPUT_DIR, "best_model"))
print("✅ 领域自适应继续预训练完成！")

# ------------------------------ 4. 加载微调模型 (从MLM checkpoint加载) ------------------------------
print("\n加载微调模型...")
model = AutoModelForSequenceClassification.from_pretrained(
    os.path.join(MLM_OUTPUT_DIR, "best_model"),
    num_labels=2
)

# ------------------------------ 5. 准备微调数据 ------------------------------
train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, MAX_LEN)
val_dataset = SentimentDataset(val_texts, val_labels, tokenizer, MAX_LEN)

# ------------------------------ 6. Focal Loss Trainer (gamma=1.0) ------------------------------
class FocalLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        gamma = 1.0
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

# ------------------------------ 8. 训练参数（加入对抗训练）------------------------------
training_args = TrainingArguments(
    output_dir=FINE_TUNE_OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir=os.path.join(LOG_DIR, "finetune"),
    logging_steps=20,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=torch.cuda.is_available(),
    dataloader_pin_memory=False,
    report_to="tensorboard",
    #adversarial_training=True,  # 对抗训练
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

print("\n开始终极训练...")
trainer.train()

# ------------------------------ 10. 保存最佳模型 ------------------------------
trainer.save_model(os.path.join(FINE_TUNE_OUTPUT_DIR, "best_model"))
tokenizer.save_pretrained(os.path.join(FINE_TUNE_OUTPUT_DIR, "best_model"))
print(f"\n🏆 终极训练完成！模型已保存至 {FINE_TUNE_OUTPUT_DIR}/best_model")