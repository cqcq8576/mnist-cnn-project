"""
notebooks.04_evaluate_model 的 Docstring
阶段4：模型评估与性能分析
"""
import sys
sys.path.append('../src')

import torch
import json
from model import SimpleCNN,get_device
from data_loader import MNISTDataLoader
from train import load_checkpoint
from evaluate import (Evaluator,plot_confusion_matrix,plot_per_class_accuracy,
                      visualize_misclassified_samples,visualize_predictions)

print("=" * 70)
print("阶段4：模型评估与性能分析")
print("=" * 70)

# 1. 加载数据
print("\n【步骤1】加载测试数据")
print("-" * 70)
mnist_loader = MNISTDataLoader(data_dir='../data', batch_size=128)
mnist_loader.load_data()
train_loader, val_loader, test_loader = mnist_loader.get_data_loaders()

print(f"测试集样本数: {len(test_loader.dataset)}")

# 2. 加载训练好的模型
print("\n【步骤2】加载训练好的模型")
print("-" * 70)
device = get_device()
model = SimpleCNN(num_classes=10)

model_path = '../models/simple_cnn_best.pth'
model, history = load_checkpoint(model, model_path, device)

# 3. 创建评估器并评估
print("\n【步骤3】在测试集上评估模型")
print("-" * 70)
evaluator = Evaluator(model, test_loader, device)
results = evaluator.evaluate()

# 4. 生成分类报告
print("\n【步骤4】生成详细分类报告")
print("-" * 70)
report = evaluator.get_classification_report(results)
print(report)

# 保存报告
report_path = '../results/classification_report.txt'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("MNIST手写数字识别 - 分类报告\n")
    f.write("=" * 70 + "\n\n")
    f.write(report)
    f.write(f"\n\n总体准确率: {results['accuracy']:.4f}%\n")
    f.write(f"正确预测:  {results['correct']}/{results['total']}\n")

print(f"✓ 分类报告已保存到:  {report_path}")

# 5. 绘制混淆矩阵
print("\n【步骤5】绘制混淆矩阵")
print("-" * 70)
cm = evaluator.get_confusion_matrix(results)
plot_confusion_matrix(cm, save_path='../results/confusion_matrix.png')

# 6. 分析每个类别的准确率
print("\n【步骤6】分析各类别性能")
print("-" * 70)
per_class_stats = evaluator.analyze_per_class_accuracy(results)

print("\n各类别详细统计:")
print(f"{'数字':<8} {'总样本':<10} {'正确数':<10} {'准确率':<10}")
print("-" * 45)
for digit in range(10):
    stats = per_class_stats[digit]
    print(f"{digit:<8} {stats['total']:<10} {stats['correct']:<10} {stats['accuracy']:.2f}%")

# 找出表现最好和最差的类别
best_digit = max(per_class_stats.items(), key=lambda x: x[1]['accuracy'])
worst_digit = min(per_class_stats.items(), key=lambda x: x[1]['accuracy'])

print(f"\n表现最好:  数字 {best_digit[0]} (准确率: {best_digit[1]['accuracy']:.2f}%)")
print(f"表现最差: 数字 {worst_digit[0]} (准确率: {worst_digit[1]['accuracy']:.2f}%)")

plot_per_class_accuracy(per_class_stats, save_path='../results/per_class_accuracy.png')

# 7. 分析错误预测样本
print("\n【步骤7】分析错误预测样本")
print("-" * 70)
misclassified = evaluator.find_misclassified_samples(results, num_samples=20)

print("\n错误预测样本示例:")
for i in range(min(10, len(misclassified['indices']))):
    true_label = misclassified['true_labels'][i]
    pred_label = misclassified['predictions'][i]
    confidence = misclassified['probabilities'][i][pred_label]
    print(f"  样本 {i+1}:  真实={true_label}, 预测={pred_label}, 置信度={confidence*100:.2f}%")

# 可视化错误样本
visualize_misclassified_samples(misclassified, results,
                                save_path='../results/misclassified_samples.png')

# 8. 可视化预测示例
print("\n【步骤8】生成预测示例可视化")
print("-" * 70)
visualize_predictions(model, test_loader, device, num_samples=16,
                     save_path='../results/prediction_examples.png')

# 9. 保存评估结果
print("\n【步骤9】保存评估结果")
print("-" * 70)

evaluation_summary = {
    'test_accuracy': float(results['accuracy']),
    'total_samples': int(results['total']),
    'correct_predictions': int(results['correct']),
    'wrong_predictions': int(results['total'] - results['correct']),
    'per_class_accuracy': {int(k): v for k, v in per_class_stats.items()},
    'best_class': {
        'digit': int(best_digit[0]),
        'accuracy': float(best_digit[1]['accuracy'])
    },
    'worst_class': {
        'digit': int(worst_digit[0]),
        'accuracy': float(worst_digit[1]['accuracy'])
    }
}

summary_path = '../results/evaluation_summary.json'
with open(summary_path, 'w', encoding='utf-8') as f:
    json.dump(evaluation_summary, f, indent=4, ensure_ascii=False)

print(f"✓ 评估摘要已保存到: {summary_path}")

# 10. 总结
print("\n" + "=" * 70)
print("评估完成总结")
print("=" * 70)

print(f"""
✓ 模型评估完成！

📊 总体性能: 
  • 测试集准确率:  {results['accuracy']:.2f}%
  • 正确预测: {results['correct']}/{results['total']}
  • 错误预测:  {results['total'] - results['correct']}

🎯 类别性能:
  • 表现最好: 数字 {best_digit[0]} ({best_digit[1]['accuracy']:.2f}%)
  • 表现最差: 数字 {worst_digit[0]} ({worst_digit[1]['accuracy']:.2f}%)
  • 平均准确率: {sum(s['accuracy'] for s in per_class_stats.values()) / 10:.2f}%

💾 生成文件:
  • 分类报告: results/classification_report.txt
  • 混淆矩阵: results/confusion_matrix. png
  • 类别准确率: results/per_class_accuracy.png
  • 错误样本: results/misclassified_samples.png
  • 预测示例:  results/prediction_examples.png
  • 评估摘要: results/evaluation_summary.json

🎉 MNIST手写数字识别项目完成！
""")

print("=" * 70)
print("阶段4完成！")
print("=" * 70)