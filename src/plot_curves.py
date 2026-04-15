import json
import glob
import matplotlib.pyplot as plt

def load_logs(model_dir):
    # 找到最新的 trainer_state.json
    files = glob.glob(f"{model_dir}/checkpoint-*/trainer_state.json")
    if not files:
        print(f"未在 {model_dir} 中找到 trainer_state.json")
        return None, None, None
    with open(files[-1], 'r') as f:
        state = json.load(f)

    epochs, train_loss, eval_loss, eval_acc = [], [], [], []
    for entry in state['log_history']:
        if 'loss' in entry:
            train_loss.append(entry['loss'])
            epochs.append(entry['epoch'])
        if 'eval_loss' in entry:
            eval_loss.append(entry['eval_loss'])
            eval_acc.append(entry['eval_accuracy'])
    return epochs, train_loss, eval_loss, eval_acc

# 加载两个实验的日志
epochs_full, train_loss_full, eval_loss_full, eval_acc_full = load_logs("../models/full_ft")
epochs_lora, train_loss_lora, eval_loss_lora, eval_acc_lora = load_logs("../models/lora_ft")

plt.figure(figsize=(14, 5))

# 子图1：训练损失对比
plt.subplot(1, 2, 1)
plt.plot(epochs_full[:len(train_loss_full)], train_loss_full, label='Full FT - Train Loss')
plt.plot(epochs_full[:len(eval_loss_full)], eval_loss_full, '--', label='Full FT - Eval Loss')
plt.plot(epochs_lora[:len(train_loss_lora)], train_loss_lora, label='LoRA - Train Loss')
plt.plot(epochs_lora[:len(eval_loss_lora)], eval_loss_lora, '--', label='LoRA - Eval Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training & Evaluation Loss')
plt.grid(True, linestyle='--', alpha=0.6)

# 子图2：验证准确率对比
plt.subplot(1, 2, 2)
plt.plot(epochs_full[:len(eval_acc_full)], eval_acc_full, 'o-', label='Full FT - Accuracy')
plt.plot(epochs_lora[:len(eval_acc_lora)], eval_acc_lora, 's-', label='LoRA - Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Validation Accuracy')
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('../outputs/training_curves_comparison.png', dpi=150)
plt.show()
print("对比曲线图已保存至 ../outputs/training_curves_comparison.png")