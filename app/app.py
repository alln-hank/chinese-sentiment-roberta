# 0. 强制离线模式（防止 Hugging Face 联网，避免网络超时卡住）
import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

# 1. 导入必要的库
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 2. 加载训练好的模型和分词器
# 这里使用全参数微调的模型，你也可以改成 "../models/lora_ft/best_model"
MODEL_PATH = "../models/roberta_ultimate/best_model"

print("正在加载分词器...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)

print("正在加载模型...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)
model.eval()  # 切换到评估模式（关闭 Dropout 等）
print("模型加载完成！")

# 3. 定义预测函数（Gradio 会调用它来处理用户输入）
def predict_sentiment(text):
    """
    输入：用户输入的评论文本（字符串）
    输出：一个字典，包含 "正面" 和 "负面" 的概率
    """
    # 3.1 对文本进行分词和编码
    inputs = tokenizer(
        text,
        max_length=273,           # 与训练时保持一致
        padding='max_length',     # 填充到最大长度
        truncation=True,          # 超长则截断
        return_tensors='pt'       # 返回 PyTorch 张量
    )

    # 3.2 模型推理（不计算梯度，节省显存）
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits   # 形状：(1, 2)，两个类别的原始得分

    # 3.3 将 logits 转换为概率
    probs = torch.nn.functional.softmax(logits, dim=-1)  # softmax 归一化
    probs = probs.squeeze().tolist()  # 去除 batch 维度，转为 Python 列表

    # 3.4 返回 Gradio Label 组件需要的格式
    return {
        "正面": probs[1],   # 索引 1 对应正面（训练时 label=1）
        "负面": probs[0]    # 索引 0 对应负面
    }

# 4. 创建 Gradio 界面
iface = gr.Interface(
    fn=predict_sentiment,                # 指定处理函数
    inputs=gr.Textbox(
        lines=3,                         # 文本框显示 3 行
        placeholder="请输入酒店评论文本...",  # 占位提示文字
        label="评论文本"                   # 输入框标签
    ),
    outputs=gr.Label(
        num_top_classes=2,               # 显示前 2 个类别
        label="情感预测概率"               # 输出标签
    ),
    title="基于 BERT 的中文情感分析系统",   # 页面标题
    description="""
    **作者**：华南师范大学 数据科学与大数据技术  
    **技术栈**：PyTorch + Transformers + LoRA  
    **说明**：输入一段中文酒店评论，模型将判断其为正面或负面的概率。
    """,
    examples=[                           # 预设示例，点击即可自动填充
        ["酒店房间很干净，早餐也很丰富，下次还会来。"],
        ["隔音太差了，隔壁说话听得一清二楚，一夜没睡好。"],
        ["环境很好，但是服务态度极差，不会再住了。"]
    ],
    theme="soft"                         # 界面主题（可选：default, soft, monochrome 等）
)

# 5. 启动服务

if __name__ == "__main__":
    # share=True 会生成一个公网链接（有效期 72 小时），可发给任何人测试
    iface.launch(share=True)