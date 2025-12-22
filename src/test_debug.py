"""
调试脚本 - 逐步测试每个部分
"""
import torch
from torchvision import datasets, transforms
from torch.utils. data import DataLoader, random_split

print("=" * 50)
print("开始调试")
print("=" * 50)

# 步骤1: 测试基本的数据加载
print("\n【步骤1】加载原始MNIST数据集")
try:
    transform = transforms.Compose([transforms.ToTensor()])
    full_train = datasets.MNIST(root='../data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='../data', train=False, download=True, transform=transform)
    print(f"✓ 完整训练集大小: {len(full_train)}")
    print(f"✓ 测试集大小: {len(test_dataset)}")
except Exception as e: 
    print(f"✗ 错误: {e}")
    exit(1)

# 步骤2: 测试直接访问
print("\n【步骤2】测试直接访问数据")
try:
    img, label = full_train[0]
    print(f"✓ 图像形状: {img.shape}")
    print(f"✓ 标签:  {label}")
except Exception as e:
    print(f"✗ 错误: {e}")
    exit(1)

# 步骤3: 测试random_split
print("\n【步骤3】测试random_split")
try:
    train_size = 48000
    val_size = 12000
    train_dataset, val_dataset = random_split(
        full_train,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"✓ 训练集大小: {len(train_dataset)}")
    print(f"✓ 验证集大小: {len(val_dataset)}")
except Exception as e:
    print(f"✗ 错误: {e}")
    exit(1)

# 步骤4: 测试访问split后的数据
print("\n【步骤4】测试访问split后的数据")
try:
    img, label = train_dataset[0]
    print(f"✓ 训练集第一个样本 - 图像形状: {img.shape}, 标签: {label}")
    
    img, label = val_dataset[0]
    print(f"✓ 验证集第一个样本 - 图像形状: {img.shape}, 标签: {label}")
except Exception as e:
    print(f"✗ 错误: {e}")
    print(f"   这可能是索引问题")
    exit(1)

# 步骤5: 测试DataLoader
print("\n【步骤5】测试DataLoader")
try:
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    print(f"✓ DataLoader创建成功")
    print(f"✓ 总batch数: {len(train_loader)}")
except Exception as e:
    print(f"✗ 错误: {e}")
    exit(1)

# 步骤6: 测试获取一个batch
print("\n【步骤6】测试获取一个batch")
try:
    images, labels = next(iter(train_loader))
    print(f"✓ Batch图像形状: {images.shape}")
    print(f"✓ Batch标签形状: {labels.shape}")
except StopIteration:
    print(f"✗ StopIteration错误 - DataLoader为空!")
    print(f"   训练集大小: {len(train_dataset)}")
    print(f"   这很奇怪，数据集不为空但迭代器为空")
except Exception as e:
    print(f"✗ 其他错误: {e}")
    exit(1)

print("\n" + "=" * 50)
print("所有测试通过！")
print("=" * 50)