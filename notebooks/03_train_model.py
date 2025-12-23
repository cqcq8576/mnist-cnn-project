"""
notebooks.03_train_model 的 Docstring
阶段3：模型训练
"""
import sys
sys.path.append('../src')

import torch
from model import SimpleCNN,get_device
from data_loader import MNISTDataLoader
from train import Trainer
from visualize_training import plot_comparison,plot_training_history,plot_training_summary

print('='*70)
print('阶段3：MNIST手写数字识别模型训练')
print('='*70)

# 1.设置随机种子，保证可以复现
print('\n[步骤1]设置环境')
print('-'*70)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
print('√ 随机种子设置完成')

# 2.加载数据
print("\n【步骤2】加载数据集")
print("-" * 70)
mnist_loader=MNISTDataLoader(
    data_dir='../data',
    batch_size=64,
    val_split=0.2
)
mnist_loader.load_data()
train_loader,val_loader,test_loader=mnist_loader.get_data_loaders()

# 3. 创建模型
print("\n【步骤3】创建模型")
print("-" * 70)
device=get_device()
model=SimpleCNN(num_classes=10)
print(model.get_model_summary())

# 4. 创建训练器
print("\n【步骤4】配置训练参数")
print("-" * 70)

# 训练超参数
LEARNING_RATE = 0.001
EPOCHS = 15

print(f"学习率: {LEARNING_RATE}")
print(f"训练轮数: {EPOCHS}")
print(f"Batch大小: {train_loader.batch_size}")
print(f"优化器: Adam")
print(f"损失函数: CrossEntropyLoss")
print(f"学习率调度: ReduceLROnPlateau")

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    learning_rate=LEARNING_RATE,
    epochs=EPOCHS
)

# 5. 开始训练
print("\n【步骤5】开始训练")
print("-" * 70)
history=trainer.train(save_path='../models/simple_cnn_best.pth')

# 6. 可视化训练结果
print("\n【步骤6】生成训练可视化")
print("-" * 70)

plot_training_history(history, save_path='../results/training_history.png')
plot_training_summary(history, save_path='../results/training_summary.png')
plot_comparison(history, save_path='../results/train_val_comparison.png')

# 7. 保存训练历史
print("\n【步骤7】保存训练记录")
print("-" * 70)

import json
history_path = '../results/training_history. json'
with open(history_path, 'w') as f:
    json.dump(history, f, indent=4)
print(f"✓ 训练历史保存到: {history_path}")

# 8. 总结
print("\n" + "=" * 70)
print("训练完成总结")
print("=" * 70)

print(f"""
✓ 模型训练完成！

📊 训练结果: 
  • 训练轮数: {EPOCHS}
  • 最佳验证准确率: {trainer.best_val_acc:.2f}%
  • 最佳模型轮次:  Epoch {trainer.best_epoch}
  • 最终训练准确率: {history['train_acc'][-1]:.2f}%
  • 最终验证准确率: {history['val_acc'][-1]:.2f}%

💾 保存文件:
  • 最佳模型: models/simple_cnn_best. pth
  • 训练曲线: results/training_history.png
  • 训练总结: results/training_summary.png
  • 训练记录: results/training_history. json

🎯 下一步: 阶段4 - 模型评估与测试
""")

print("=" * 70)
print("阶段3完成！")
print("=" * 70)

