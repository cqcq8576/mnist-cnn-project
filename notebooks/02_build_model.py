"""
notebooks.02_build_model 的 Docstring
阶段2：构建基础的CNN模型
"""

import sys
sys.path.append('../src')

import torch
import torch.nn.functional as F
from model import SimpleCNN,get_device,count_parameters
from data_loader import MNISTDataLoader

print("=" * 60)
print("阶段2：构建基础CNN模型")
print("=" * 60)

# 1. 创建模型
print("\n【步骤1】创建SimpleCNN模型")
print("-" * 60)
model = SimpleCNN(num_classes=10)
print("✓ 模型创建成功\n")

# 2. 查看模型结构
print("【步骤2】模型结构详情")
print("-" * 60)
print(model)

# 3. 模型概要
print("\n" + model.get_model_summary())

# 4. 参数统计
print("\n【步骤3】参数统计")
print("-" * 60)
total_params, trainable_params = count_parameters(model)
print(f"总参数量: {total_params:,}")
print(f"可训练参数:  {trainable_params:,}")

# 逐层参数详情
print("\n逐层参数详情:")
for name, param in model.named_parameters():
    print(f"{name:20s}:{param.shape}→{param.numel():,} 参数")

# 5. 设备配置
print("\n【步骤4】设备配置")
print("-" * 60)
device = get_device()
model = model.to(device)

# 6. 测试模型前向传播
print("\n【步骤5】测试模型前向传播")
print("-" * 60)

# 创建测试数据
test_input = torch.randn(8, 1, 28, 28).to(device)
print(f"输入张量形状: {test_input.shape}")
print(f"  - batch_size: 8")
print(f"  - channels: 1 (灰度图)")
print(f"  - height: 28")
print(f"  - width: 28")

# 前向传播
with torch.no_grad():  # 测试时不计算梯度
    output = model(test_input)

print(f"\n输出张量形状:  {output.shape}")
print(f"  - batch_size:  8")
print(f"  - num_classes: 10")

print(f"\n第一个样本的输出（logits）:")
print(output[0])

# 转换为概率
probabilities = F.softmax(output, dim=1)
print(f"\n第一个样本的概率分布:")
for i, prob in enumerate(probabilities[0]):
    print(f"  数字 {i}: {prob. item():.4f}")

predicted_class = output.argmax(dim=1)
print(f"\n预测类别: {predicted_class. tolist()}")

# 7. 使用真实数据测试
print("\n【步骤6】使用MNIST真实数据测试")
print("-" * 60)

# 加载数据
mnist_loader = MNISTDataLoader(data_dir='../data', batch_size=16)
mnist_loader.load_data()
train_loader, val_loader, test_loader = mnist_loader.get_data_loaders()

# 获取一个batch
images, labels = next(iter(train_loader))
images = images.to(device)
labels = labels.to(device)

print(f"真实数据形状: {images.shape}")
print(f"真实标签:  {labels[: 10]. tolist()}")

# 模型预测
with torch.no_grad():
    outputs = model(images)
    predictions = outputs.argmax(dim=1)

print(f"模型预测:  {predictions[: 10].tolist()}")
print(f"\n注意:  模型尚未训练，预测结果是随机的")

# 8. 模型保存
print("\n【步骤7】保存模型结构")
print("-" * 60)

import os
os.makedirs('../models', exist_ok=True)

# 保存模型结构和初始参数
model_path = '../models/simple_cnn_initial.pth'
torch.save({
    'model_state_dict': model.state_dict(),
    'model_architecture': 'SimpleCNN',
    'num_classes': 10,
}, model_path)

print(f"✓ 模型已保存到: {model_path}")

# 测试加载
checkpoint = torch.load(model_path)
model_loaded = SimpleCNN(num_classes=checkpoint['num_classes'])
model_loaded.load_state_dict(checkpoint['model_state_dict'])
print(f"✓ 模型加载测试成功")

# 9. 总结
print("\n" + "=" * 60)
print("阶段2总结")
print("=" * 60)
print(f"""
✓ CNN模型设计完成
✓ 模型结构:  2层卷积 + 2层池化 + 2层全连接
✓ 总参数量: {total_params:,}
✓ 设备: {device}
✓ 输入:  (batch, 1, 28, 28)
✓ 输出:  (batch, 10)

📊 网络层次: 
  1. Conv2d(1→32, 3x3) + ReLU + MaxPool(2x2)
  2. Conv2d(32→64, 3x3) + ReLU + MaxPool(2x2)
  3. Linear(1600→128) + ReLU
  4. Linear(128→10)

🎯 设计思路:
  - 卷积层:  提取图像特征（从边缘到形状）
  - 池化层: 降维并增强特征不变性
  - 全连接层: 将特征映射到类别
  - ReLU激活: 引入非线性，增强表达能力

📝 模型已保存，可用于后续训练

🚀 下一步:  阶段3 - 模型训练与评估
""")

print("=" * 60)
print("阶段2完成！")
print("=" * 60)