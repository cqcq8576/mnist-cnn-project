"""
数据可视化工具模块
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import os


# 设置中文字体（解决中文显示问题）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_sample_images(dataset, num_samples=10, title="MNIST样本展示"):
    """
    可视化数据集中的随机样本
    
    Args:
        dataset: PyTorch数据集
        num_samples: 要显示的样本数量
        title: 图表标题
    """
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    fig.suptitle(title, fontsize=16)
    
    # 随机选择样本
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        image, label = dataset[idx]
        
        # 如果是Tensor，转换为numpy并去掉通道维度
        if torch.is_tensor(image):
            image = image.squeeze().numpy()  # (1, 28, 28) -> (28, 28)
        
        row = i // 5
        col = i % 5
        
        axes[row, col].imshow(image, cmap='gray')
        axes[row, col].set_title(f'标签: {label}')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('../results/sample_images.png', dpi=150, bbox_inches='tight')
    print("样本图像已保存到:  results/sample_images.png")
    plt.show()


def plot_class_distribution(distribution, title="类别分布"):
    """
    绘制类别分布柱状图
    
    Args:
        distribution: 字典，键为类别，值为数量
        title: 图表标题
    """
    classes = list(distribution.keys())
    counts = list(distribution.values())
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(classes, counts, color='steelblue', alpha=0.8)
    
    # 在柱子上显示数值
    for bar in bars:
        height = bar. get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    plt.xlabel('数字类别', fontsize=12)
    plt.ylabel('样本数量', fontsize=12)
    plt.title(title, fontsize=14)
    plt.xticks(classes)
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../results/class_distribution.png', dpi=150, bbox_inches='tight')
    print("类别分布图已保存到: results/class_distribution.png")
    plt.show()


def plot_pixel_statistics(dataset, num_samples=1000):
    """
    分析像素值统计信息
    
    Args:  
        dataset: PyTorch数据集
        num_samples: 用于统计的样本数量
    """
    # 随机采样
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    pixels = []
    for idx in indices:
        image, _ = dataset[idx]
        if torch.is_tensor(image):
            image = image.numpy()
        pixels.append(image. flatten())
    
    pixels = np.concatenate(pixels)
    
    # 创建子图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 像素值分布直方图
    axes[0]. hist(pixels, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('像素值', fontsize=12)
    axes[0].set_ylabel('频数', fontsize=12)
    axes[0].set_title('像素值分布', fontsize=14)
    axes[0].grid(alpha=0.3)
    
    # 统计信息
    stats_text = f"""
    统计信息: 
    ━━━━━━━━━━━━━━━
    均值: {pixels.mean():.4f}
    标准差:  {pixels.std():.4f}
    最小值: {pixels.min():.4f}
    最大值: {pixels.max():.4f}
    中位数: {np.median(pixels):.4f}
    """
    
    axes[1].text(0.1, 0.5, stats_text, fontsize=12, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', 
                facecolor='wheat', alpha=0.5))
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('../results/pixel_statistics.png', dpi=150, bbox_inches='tight')
    print("像素统计图已保存到: results/pixel_statistics.png")
    plt.show()


def visualize_batch(data_loader, title="一个Batch的数据"):
    """
    可视化一个batch的数据
    
    Args:
        data_loader: PyTorch数据加载器
        title: 图表标题
    """
    # 获取一个batch
    images, labels = next(iter(data_loader))
    
    # 显示前16张图片
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    fig.suptitle(title, fontsize=16)
    
    for i in range(16):
        row = i // 4
        col = i % 4
        
        # 去掉通道维度
        image = images[i].squeeze().numpy()  # (1, 28, 28) -> (28, 28)
        label = labels[i].item()
        
        axes[row, col].imshow(image, cmap='gray')
        axes[row, col].set_title(f'标签: {label}')
        axes[row, col]. axis('off')
    
    plt.tight_layout()
    plt.savefig('../results/batch_visualization.png', dpi=150, bbox_inches='tight')
    print("Batch可视化已保存到: results/batch_visualization.png")
    plt.show()


if __name__ == "__main__":  
    # 测试可视化功能
    from data_loader import MNISTDataLoader
    
    print("加载数据...")
    mnist_loader = MNISTDataLoader(data_dir='../data')
    mnist_loader.load_data()
    
    print("\n生成可视化...")
    plot_sample_images(mnist_loader.train_dataset)