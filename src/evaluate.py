"""模型评估模块"""
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class Evaluator:
    """模型评估器"""
    def __init__(self,model,test_loader,device):
        """
        初始化评估器
        
        :param model: 训练好的模型
        :param test_loader: 测试数据加载器
        :param device: 使用的设备
        """
        self.model=model.to(device)
        self.test_loader=test_loader
        self.device=device
        self.model.eval()
    
    def evaluate(self):
        """
        在测试集上评估模型
        会返回一个包含预测结果，真实结果，准确率的字典
        """
        all_predictions=[]
        all_labels=[]
        all_probs=[]
        all_images = []  # ← 添加这一行，保存图像
        correct=0
        total=0

        print("=" * 70)
        print("开始评估模型")
        print("=" * 70)
        with torch.no_grad():
            # 进度条
            pbar=tqdm(self.test_loader, desc='评估中')
            for images,labels in pbar:
                # 保存原始图像（在移到device之前）
                all_images.extend(images.numpy())  # ✅ 确保有这一行
                images=images.to(self.device)
                labels=labels.to(self.device)
                # 前向传播
                outputs=self.model(images)
                # 获取概率分布
                probs=torch.softmax(outputs,dim=1)
                # 获取预测结果
                _,predicted=outputs.max(1)
                # 统计
                total+=labels.size(0)
                correct+=predicted.eq(labels).sum().item()
                # 保存结果
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                # 更新进度条
                pbar.set_postfix({'准确率':f'{100.*correct/total:.2f}%'})
        accuracy=100.*correct/total
        print(f"\n测试集准确率: {accuracy:.2f}%")
        print(f"正确预测:  {correct}/{total}")
        print("=" * 70)     
        return {
            'predictions': np.array(all_predictions),
            'labels': np.array(all_labels),
            'probabilities': np.array(all_probs),
            'images': np.array(all_images),  # ← 添加这一行
            'accuracy':  accuracy,
            'correct':  correct,
            'total': total
        }         

    def get_classification_report(self, results):
        """
        生成分类报告
        
        Args:
            results: evaluate()返回的结果字典
            
        Returns: 
            str: 分类报告文本
        """
        target_names = [f'数字 {i}' for i in range(10)]
        
        report = classification_report(
            results['labels'],
            results['predictions'],
            target_names=target_names,
            digits=4
        )
        
        return report        
    
    def get_confusion_matrix(self, results):
        """
        计算混淆矩阵
        
        Args:
            results: evaluate()返回的结果字典
            
        Returns:
            numpy.ndarray: 混淆矩阵
        """
        cm = confusion_matrix(results['labels'], results['predictions'])
        return cm
    
    def find_misclassified_samples(self, results, num_samples=20):
        """
        找出错误分类的样本
        
        Args: 
            results: evaluate()返回的结果字典
            num_samples: 返回的错误样本数量
            
        Returns:
            dict: 错误样本的信息
        """
        predictions = results['predictions']
        labels = results['labels']
        probs = results['probabilities']
        
        # 找出预测错误的索引
        wrong_indices = np.where(predictions != labels)[0]
        
        print(f"\n找到 {len(wrong_indices)} 个错误预测样本")
        
        # 随机选择若干个
        if len(wrong_indices) > num_samples:
            selected_indices = np.random.choice(wrong_indices, num_samples, replace=False)
        else:
            selected_indices = wrong_indices
        
        misclassified = {
            'indices': selected_indices,
            'true_labels': labels[selected_indices],
            'predictions': predictions[selected_indices],
            'probabilities': probs[selected_indices]
        }
        
        return misclassified
    
    def analyze_per_class_accuracy(self, results):
        """
        分析每个类别的准确率
        
        Args:
            results:  evaluate()返回的结果字典
            
        Returns: 
            dict: 每个类别的统计信息
        """
        predictions = results['predictions']
        labels = results['labels']
        
        per_class_stats = {}
        
        for digit in range(10):
            # 该数字的所有样本
            mask = labels == digit
            total_samples = mask.sum()
            
            # 正确预测的数量
            correct_predictions = (predictions[mask] == digit).sum()
            
            # 准确率
            accuracy = 100. * correct_predictions / total_samples if total_samples > 0 else 0
            
            per_class_stats[digit] = {
                'total':  int(total_samples),
                'correct': int(correct_predictions),
                'accuracy': float(accuracy)
            }
        
        return per_class_stats


def plot_confusion_matrix(cm, save_path='../results/confusion_matrix.png'):
    """
    绘制混淆矩阵
    
    Args:
        cm: 混淆矩阵
        save_path: 保存路径
    """
    plt. figure(figsize=(12, 10))
    
    # 使用seaborn绘制热力图
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10),
                cbar_kws={'label': '样本数量'})
    
    plt.xlabel('预测标签', fontsize=12)
    plt.ylabel('真实标签', fontsize=12)
    plt.title('混淆矩阵 - MNIST测试集', fontsize=14, fontweight='bold')
    
    # 添加说明文本
    plt.text(5, -1.5, '对角线：正确预测  |  非对角线：错误预测', 
             ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ 混淆矩阵保存到: {save_path}")
    plt.show()


def plot_per_class_accuracy(per_class_stats, save_path='../results/per_class_accuracy.png'):
    """
    绘制每个类别的准确率
    
    Args:
        per_class_stats: 每个类别的统计信息
        save_path: 保存路径
    """
    digits = list(per_class_stats.keys())
    accuracies = [per_class_stats[d]['accuracy'] for d in digits]
    totals = [per_class_stats[d]['total'] for d in digits]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 准确率柱状图
    bars = ax1.bar(digits, accuracies, color='steelblue', alpha=0.8)
    ax1.set_xlabel('数字类别', fontsize=12)
    ax1.set_ylabel('准确率 (%)', fontsize=12)
    ax1.set_title('各类别准确率', fontsize=14, fontweight='bold')
    ax1.set_ylim([95, 100])
    ax1.grid(axis='y', alpha=0.3)
    
    # 在柱子上显示数值
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2f}%',
                ha='center', va='bottom', fontsize=9)
    
    # 样本数量柱状图
    bars2 = ax2.bar(digits, totals, color='coral', alpha=0.8)
    ax2.set_xlabel('数字类别', fontsize=12)
    ax2.set_ylabel('样本数量', fontsize=12)
    ax2.set_title('各类别样本分布', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # 在柱子上显示数值
    for bar, total in zip(bars2, totals):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{total}',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ 类别准确率图保存到: {save_path}")
    plt.show()


def visualize_misclassified_samples(misclassified, results, 
                                     save_path='../results/misclassified_samples.png'):
    """
    可视化错误分类的样本
    
    Args:
        misclassified: 错误样本信息字典
        test_dataset: 测试数据集
        save_path:  保存路径
    """
    indices = misclassified['indices']
    true_labels = misclassified['true_labels']
    predictions = misclassified['predictions']
    probs = misclassified['probabilities']

    # 检查results中是否有images
    if 'images' not in results:
        raise ValueError("results字典中缺少'images'键，请确保evaluate()方法返回了图像数据")
    
    # 从results中获取图像
    all_images = results['images']

    # 最多显示20个
    num_samples = min(len(indices), 20)
    
    fig, axes = plt. subplots(4, 5, figsize=(15, 12))
    fig.suptitle('错误预测样本分析', fontsize=16, fontweight='bold')
    
    for i in range(num_samples):
        row = i // 5
        col = i % 5
        
        # 获取图像
        idx = indices[i]
        image = all_images[idx]. squeeze()  # (1, 28, 28) -> (28, 28)
        
        true_label = true_labels[i]
        pred_label = predictions[i]
        pred_prob = probs[i][pred_label]
        true_prob = probs[i][true_label]
        
        # 显示图像
        axes[row, col].imshow(image, cmap='gray')
        
        # 设置标题（红色表示错误）
        title = f'真实:  {true_label} ({true_prob*100:.1f}%)\n'
        title += f'预测: {pred_label} ({pred_prob*100:.1f}%)'
        axes[row, col].set_title(title, fontsize=9, color='red')
        axes[row, col].axis('off')
    
    # 隐藏多余的子图
    for i in range(num_samples, 20):
        row = i // 5
        col = i % 5
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ 错误样本可视化保存到: {save_path}")
    plt.show()


def visualize_predictions(model, test_loader, device, num_samples=16,
                         save_path='../results/prediction_examples.png'):
    """
    可视化模型预测结果（包括正确和错误的）
    
    Args:
        model: 模型
        test_loader: 测试数据加载器
        device: 设备
        num_samples: 显示样本数
        save_path: 保存路径
    """
    model.eval()
    
    # 获取一批数据
    images, labels = next(iter(test_loader))
    images = images.to(device)
    labels = labels.to(device)
    
    # 预测
    with torch.no_grad():
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        _, predictions = outputs.max(1)
    
    # 转回CPU
    images = images.cpu()
    labels = labels.cpu()
    predictions = predictions.cpu()
    probs = probs.cpu()
    
    # 可视化
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle('模型预测示例', fontsize=16, fontweight='bold')
    
    for i in range(num_samples):
        row = i // 4
        col = i % 4
        
        image = images[i].squeeze().numpy()
        true_label = labels[i].item()
        pred_label = predictions[i]. item()
        confidence = probs[i][pred_label].item()
        
        # 显示图像
        axes[row, col].imshow(image, cmap='gray')
        
        # 设置标题（正确为绿色，错误为红色）
        if true_label == pred_label: 
            color = 'green'
            title = f'✓ 预测:  {pred_label} ({confidence*100:.1f}%)'
        else:
            color = 'red'
            title = f'✗ 真实: {true_label}\n预测: {pred_label} ({confidence*100:.1f}%)'
        
        axes[row, col].set_title(title, fontsize=10, color=color, fontweight='bold')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ 预测示例保存到: {save_path}")
    plt.show()


if __name__ == "__main__": 
    # 测试评估模块
    print("=" * 50)
    print("测试评估模块")
    print("=" * 50)
    
    from model import SimpleCNN, get_device
    from data_loader import MNISTDataLoader
    from train import load_checkpoint
    
    # 加载数据
    print("\n加载数据...")
    mnist_loader = MNISTDataLoader(data_dir='../data', batch_size=64)
    mnist_loader.load_data()
    train_loader, val_loader, test_loader = mnist_loader. get_data_loaders()
    
    # 加载训练好的模型
    print("\n加载模型...")
    device = get_device()
    model = SimpleCNN(num_classes=10)
    
    try:
        model, history = load_checkpoint(model, '../models/simple_cnn_best.pth', device)
    except FileNotFoundError:
        print("⚠ 未找到训练好的模型，使用未训练的模型进行测试")
    
    # 创建评估器
    evaluator = Evaluator(model, test_loader, device)
    
    # 评估
    results = evaluator.evaluate()

    # 检查results中是否有images
    print(f"\nresults keys: {results.keys()}")
    print(f"images shape: {results['images'].shape}")
    
    # 分类报告
    print("\n分类报告:")
    print(evaluator.get_classification_report(results))
    
    # 混淆矩阵
    cm = evaluator.get_confusion_matrix(results)
    plot_confusion_matrix(cm)
    
    # 每个类别的准确率
    per_class_stats = evaluator.analyze_per_class_accuracy(results)
    plot_per_class_accuracy(per_class_stats)

    # 找出错误样本并可视化
    misclassified = evaluator.find_misclassified_samples(results, num_samples=20)
    visualize_misclassified_samples(misclassified, results)  # ✅ 传入results
    
    print("\n✓ 评估模块测试成功！")