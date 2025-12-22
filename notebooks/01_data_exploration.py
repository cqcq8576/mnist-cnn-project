"""
notebooks.01_data_exploration 的 Docstring
阶段1：MNIST数据集的探索与可视化
"""
import sys
sys.path.append('../src')

from data_loader import MNISTDataLoader,get_class_distribution
from visualize import (plot_sample_images,plot_class_distribution,plot_pixel_statistics,visualize_batch)

import os

os.makedirs('../results',exist_ok=True)
print('='*60)
print('MNIST手写数字数据集探索')
print('='*60)

# 1.加载数据
print('\n[步骤1]加载MNIST数据集')
print('-'*60)
mnist_loader=MNISTDataLoader(data_dir='../data',batch_size=64,val_split=0.2)
mnist_loader.load_data()

# 2.查看数据基本信息
print('\n[步骤2]数据集基本信息')
print('-'*60)
info=mnist_loader.get_data_info()
for key,value in info.items():
    print(f'{key:15s}:{value}')

# 3. 可视化样本图像
print("\n【步骤3】可视化样本图像")
print("-" * 60)
plot_sample_images(mnist_loader. train_dataset, num_samples=10, 
                  title="MNIST训练集样本展示")

# 4. 分析类别分布
print("\n【步骤4】分析类别分布")
print("-" * 60)

# 训练集分布
train_dist = get_class_distribution(mnist_loader.train_dataset)
print("训练集类别分布:")
for digit, count in train_dist.items():
    print(f"  数字 {digit}: {count:5d} 样本 ({count/len(mnist_loader.train_dataset)*100:.2f}%)")

plot_class_distribution(train_dist, title="训练集类别分布")

# 测试集分布
test_dist = get_class_distribution(mnist_loader. test_dataset)
print("\n测试集类别分布:")
for digit, count in test_dist.items():
    print(f"  数字 {digit}: {count:5d} 样本 ({count/len(mnist_loader.test_dataset)*100:.2f}%)")

plot_class_distribution(test_dist, title="测试集类别分布")

# 5. 像素值统计分析
print("\n【步骤5】像素值统计分析")
print("-" * 60)
plot_pixel_statistics(mnist_loader.train_dataset, num_samples=1000)

# 6. 可视化一个Batch
print("\n【步骤6】可视化数据批次")
print("-" * 60)
train_loader, val_loader, test_loader = mnist_loader.get_data_loaders()
visualize_batch(train_loader, title="训练集Batch示例 (Batch Size=64)")

# 7. 总结
print("\n" + "=" * 60)
print("数据探索总结")
print("=" * 60)
print(f"""
✓ 数据集已成功加载
✓ 训练集:  {len(mnist_loader.train_dataset)} 样本
✓ 验证集: {len(mnist_loader.val_dataset)} 样本  
✓ 测试集: {len(mnist_loader. test_dataset)} 样本
✓ 图像尺寸: 28×28 像素
✓ 类别数量: 10 (数字0-9)
✓ 数据分布:  基本均衡
✓ 像素范围:  [0, 1] (已归一化)

📊 所有可视化结果已保存到 results/ 目录

🎯 下一步:  开始阶段2 - 构建基础CNN模型
""")

print("=" * 60)
print("阶段1完成！")
print("=" * 60)


