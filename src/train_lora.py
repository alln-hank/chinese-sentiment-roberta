#导入后续必要的库
import pandas as pd
import numpy as np
from pandas.core.common import random_state
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from transformers.testing_utils import require_peft

#导入自己写的数据集类
from dataset import SentimentDataset

# 1. 加载数据
print("正在加载数据...")
df = pd.read_csv("../data/processed/cleaned_reviews.csv")
texts = df['clean_review'].tolist()
labels = df['label'].tolist()

# 2. 划分训练集和验证集（8：2，分层抽样）
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)
print(f"训练集样本数：{len(train_texts)}")
print(f"验证集样本数：{len(val_texts)}")

# 3. 加载分词器和预训练模型
MODEL_NAME = "bert-base-chinese"

print("正在加载分词器...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("正在加载与训练模型...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2  # 二分类任务
)

from peft import LoraConfig, get_peft_model, TaskType

# --- 插入 LoRA 适配器 ---
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,                        # 低秩矩阵的秩
    lora_alpha=32,              # 缩放系数
    target_modules=["query", "value"],  # 作用在 Transformer 的 Q 和 V 矩阵
    lora_dropout=0.1,
    bias="none"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # 打印可训练参数量

# 4. 超参数设置
MAX_LEN = 273       # 根据数据分析结果：90％分位数为273
BATCH_SIZE = 16     #批次大小
EPOCHS = 3          #训练轮次

# 5. 构建训练集和验证集的Dataset对象
train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, MAX_LEN)
val_dataset = SentimentDataset(val_texts, val_labels, tokenizer, MAX_LEN)

# 6. 定义评估指标函数（Trainer 在验证时会调用）
def compute_metrics(eval_pred):
    """
    输入：eval_pred是一个元组（logits，labels）
    输出：一个字典，包含我们关心的指标
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)    # 取logits中概率最大的索引作为预测类别
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='binary')
    return{
        'accuracy': acc,
        'f1': f1
    }

# 7. 配置训练参数（TrainingArguments）
training_args = TrainingArguments(
    #核心输出设置
    output_dir="../models/lora_ft",     # 模型和日志保存的根目录
    overwrite_output_dir=True,          #如果目录已有内容，是否覆盖（是）

    #训练轮次与批次
    num_train_epochs=EPOCHS,                    #训练轮次
    per_device_train_batch_size=BATCH_SIZE,     #训练时每个GPU/CPU的批次大小
    per_device_eval_batch_size=BATCH_SIZE,      #验证时的批次大小

    #优化器与学习率调度
    warmup_steps=100,       #学习率预热步数：前100 步 学习率从 0 线性增加到设定值
    weight_decay=0.01,      #权重衰减（L2正则化系数），防止过拟合

    #日志与保存策略
    logging_dir="../logs",              #TensorBoard日志目录
    logging_steps=20,                   #每20个训练步骤记录一次loss
    evaluation_strategy="epoch",        #每个epoch结束时在验证集上评估
    save_strategy="epoch",              #每个epoch结束时保存一次模型
    save_total_limit=2,                 #最多保留2个检查点（自动删除更早的）
    load_best_model_at_end=True,        #训练结束后，自动加载验证集上表现最好的模型
    metric_for_best_model="accuracy",   # 用哪个指标来判断“最好”

    #性能优化选项
    fp16=torch.cuda.is_available(),          # 如果有 GPU，启用混合精度训练（加速且省显存）
    dataloader_pin_memory=False,             # 如果内存不足可设为 False
    report_to="tensorboard",                 # 将日志报告给 TensorBoard
)

# 8. 创建 Trainer 对象
trainer = Trainer(
    model=model,                             # 要微调的模型
    args=training_args,                      # 训练参数配置
    train_dataset=train_dataset,             # 训练集
    eval_dataset=val_dataset,                # 验证集
    tokenizer=tokenizer,                     # 分词器（保存模型时会一起保存）
    compute_metrics=compute_metrics,         # 评估函数
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]  # 早停：连续 2 个 epoch 指标不涨就停止
)

# 9. 开始训练
print("\n开始训练...")
trainer.train()

# 10. 保存最佳模型和分词器
print("\n保存最佳模型...")
trainer.save_model("../models/lora_ft/best_model")
tokenizer.save_pretrained("../models/lora_ft/best_model")
print("\n训练完成！最佳模型已保存至 ../models/lora_ft/best_model")