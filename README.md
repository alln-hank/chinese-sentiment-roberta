# 基于 RoBERTa 的中文情感分析系统

##  项目简介
本项目是华南师范大学数据科学与大数据技术专业复试作品。基于 `hfl/chinese-roberta-wwm-ext` 预训练模型，在 20 万+ 多领域中文评论数据上进行微调，实现了高精度的二分类情感分析（正面/负面）。

##  技术栈
- Python 3.9 + PyTorch 2.5.1
- HuggingFace Transformers
- LoRA（PEFT）、Focal Loss
- Gradio（演示界面）
- 数据增强 + 针对性困难样本构造

##  实验对比
| 模型版本 | 验证准确率 | 验证 F1 | 说明 |
|:---|:---|:---|:---|
| 基础 BERT 全参数微调 | 91.31% | 93.60% | 基线 |
| BERT + LoRA | 91.31% | 93.56% | 高效微调 |
| BERT + 数据增强 + Focal Loss | 96.85% | 98.20% | 数据增强 |
| BERT + 转折样本 v1 | 96.55% | 98.00% | 针对性困难样本 |
| **RoBERTa + 扩充数据 + MLM（最终版）** | **96.81%** | **96.76%** | **终极模型** |

##  快速开始
### 1. 安装依赖
```bash
pip install -r requirements.txt