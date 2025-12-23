"""
训练过程可视化模块
"""
import matplotlib.pyplot as plt
import numpy as np


# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt. rcParams['axes.unicode_minus'] = False


def plot_training_history(history, save_path='../results/training_history.png'):
    """
    绘制训练历史曲线
    
    Args: 
        history: 训练历史字典，包含train_loss, train_acc, val_loss, val_acc
        save_path: 保存路径
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. 损失曲线
    axes[0].plot(epochs, history['train_loss'], 'b-o', label='训练损失', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-s', label='验证损失', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('训练与验证损失', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # 2. 准确率曲线
    axes[1].plot(epochs, history['train_acc'], 'b-o', label='训练准确率', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-s', label='验证准确率', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('训练与验证准确率', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    # 3. 学习率变化
    if 'learning_rates' in history and history['learning_rates']:
        axes[2].plot(epochs, history['learning_rates'], 'g-^', linewidth=2)
        axes[2].set_xlabel('Epoch', fontsize=12)
        axes[2].set_ylabel('Learning Rate', fontsize=12)
        axes[2].set_title('学习率变化', fontsize=14, fontweight='bold')
        axes[2].set_yscale('log')  # 对数坐标
        axes[2]. grid(True, alpha=0.3)
    else:
        axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ 训练历史图保存到: {save_path}")
    plt.show()


def plot_training_summary(history, save_path='../results/training_summary.png'):
    """
    绘制训练总结（包含最终指标）
    
    Args: 
        history: 训练历史
        save_path: 保存路径
    """
    fig = plt.figure(figsize=(14, 6))
    
    # 左侧：曲线图
    ax1 = plt.subplot(1, 2, 1)
    epochs = range(1, len(history['train_acc']) + 1)
    
    ax1.plot(epochs, history['train_acc'], 'b-o', label='训练准确率', linewidth=2, markersize=6)
    ax1.plot(epochs, history['val_acc'], 'r-s', label='验证准确率', linewidth=2, markersize=6)
    
    # 标记最佳点
    best_epoch = np.argmax(history['val_acc']) + 1
    best_acc = max(history['val_acc'])
    ax1.plot(best_epoch, best_acc, 'g*', markersize=20, label=f'最佳 (Epoch {best_epoch})')
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('训练过程：准确率变化', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 右侧：统计信息
    ax2 = plt.subplot(1, 2, 2)
    ax2.axis('off')
    
    # 计算统计信息
    final_train_acc = history['train_acc'][-1]
    final_val_acc = history['val_acc'][-1]
    final_train_loss = history['train_loss'][-1]
    final_val_loss = history['val_loss'][-1]
    
    acc_improvement = history['val_acc'][-1] - history['val_acc'][0]
    
    summary_text = f"""
    训练总结
    {'='*40}
    
    训练轮数: {len(history['train_acc'])} epochs
    
    最终结果: 
      • 训练准确率: {final_train_acc:.2f}%
      • 验证准确率: {final_val_acc:.2f}%
      • 训练损失: {final_train_loss:.4f}
      • 验证损失: {final_val_loss:.4f}
    
    最佳表现:
      • 最佳验证准确率:  {best_acc:.2f}%
      • 达到轮次:  Epoch {best_epoch}
    
    改进情况:
      • 准确率提升: {acc_improvement:.2f}%
      • 从 {history['val_acc'][0]:.2f}% → {final_val_acc:.2f}%
    
    {'='*40}
    """
    
    ax2.text(0.1, 0.5, summary_text, 
            fontsize=11,
            family='monospace',
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ 训练总结图保存到: {save_path}")
    plt.show()


def plot_comparison(history, save_path='../results/train_val_comparison.png'):
    """
    绘制训练集与验证集的对比
    
    Args: 
        history: 训练历史
        save_path: 保存路径
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 损失对比
    ax1.plot(epochs, history['train_loss'], 'b-', label='训练集', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r--', label='验证集', linewidth=2)
    ax1.fill_between(epochs, history['train_loss'], history['val_loss'], alpha=0.2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('损失对比', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 准确率对比
    ax2.plot(epochs, history['train_acc'], 'b-', label='训练集', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r--', label='验证集', linewidth=2)
    ax2.fill_between(epochs, history['train_acc'], history['val_acc'], alpha=0.2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('准确率对比', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ 对比图保存到: {save_path}")
    plt.show()


if __name__ == "__main__":
    # 测试可视化（使用模拟数据）
    print("测试训练可视化模块...")
    
    # 模拟训练历史
    mock_history = {
        'train_loss': [0.5, 0.3, 0.2, 0.15, 0.12, 0.10, 0.08, 0.07, 0.06, 0.05],
        'train_acc': [85, 90, 93, 95, 96, 97, 97.5, 98, 98.2, 98.5],
        'val_loss': [0.4, 0.25, 0.18, 0.15, 0.13, 0.12, 0.11, 0.11, 0.11, 0.12],
        'val_acc':  [87, 91, 94, 95, 96, 96.5, 97, 97.2, 97.3, 97.2],
        'learning_rates': [0.001] * 6 + [0.0005] * 4
    }

    


    plot_training_history(mock_history)
    plot_training_summary(mock_history)
    plot_comparison(mock_history)
    
    print("\n✓ 可视化测试完成！")