"""
完整部署：识别任意尺寸图片中的多个数字
"""
import sys
from pathlib import Path

# 添加src到路径
current_file = Path(__file__).resolve()
project_root = current_file.parent. parent
src_dir = project_root / 'src'
sys.path. insert(0, str(src_dir))

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from model import SimpleCNN, get_device
from complete_recognizer import CompleteDigitRecognizer

print("=" * 70)
print("完整部署：多数字识别系统")
print("=" * 70)

# 1. 加载模型
print("\n【步骤1】加载模型")
print("-" * 70)

device = get_device()
model = SimpleCNN(num_classes=10)

recognizer = CompleteDigitRecognizer(
    model=model,
    model_path='../models/simple_cnn_best.pth',
    device=device
)

# 2. 创建各种测试场景
print("\n【步骤2】创建测试场景")
print("-" * 70)

test_dir = project_root / 'test_images' / 'complete_test'
test_dir.mkdir(parents=True, exist_ok=True)


def create_test_image_safe(digits, base_size=35, spacing=25, padding=20):
    """
    创建测试图像（安全稳定版本）
    
    Args:
        digits: 数字列表 [1, 2, 3, ...]
        base_size: 基础数字大小（像素）
        spacing: 数字间距（像素）
        padding: 画布边距（像素）
    
    Returns:
        numpy.ndarray: 生成的图像
    """
    from data_loader import MNISTDataLoader
    
    # 加载MNIST数据
    mnist_loader = MNISTDataLoader(data_dir='../data')
    mnist_loader.load_data()
    test_dataset = mnist_loader.test_dataset
    
    # 计算画布尺寸（确保足够大）
    canvas_width = padding * 2 + len(digits) * base_size + (len(digits) - 1) * spacing + 50  # 额外留50像素
    canvas_height = base_size + padding * 2
    
    print(f"  创建画布: {canvas_width} × {canvas_height}")
    
    # 创建白色画布
    canvas = np.ones((canvas_height, canvas_width), dtype=np.uint8) * 255
    
    # 当前x坐标
    current_x = padding
    
    # 逐个放置数字
    for idx, digit in enumerate(digits):
        # 在测试集中查找该数字
        digit_img = None
        for i in range(len(test_dataset)):
            img, label = test_dataset[i]
            if label == digit:
                digit_img = img. squeeze().numpy()
                break
        
        if digit_img is None:
            print(f"  警告: 未找到数字 {digit}")
            continue
        
        # 随机调整大小（±3像素）
        size_var = np.random.randint(-3, 4)
        digit_size = max(20, min(base_size + size_var, base_size + 5))
        
        # 转换为0-255并调整大小
        digit_img = (digit_img * 255).astype(np.uint8)
        digit_resized = cv2.resize(digit_img, (digit_size, digit_size))
        
        # 计算垂直位置（居中）
        y_pos = (canvas_height - digit_size) // 2
        
        # 检查边界
        if current_x + digit_size > canvas_width: 
            print(f"  警告: 数字 {idx+1} 超出边界，停止放置")
            break
        
        # 放置数字（反转颜色：MNIST是白底黑字，我们要黑底白字）
        canvas[y_pos:y_pos+digit_size, current_x:current_x+digit_size] = 255 - digit_resized
        
        # 更新x坐标
        current_x += digit_size + spacing
        
        if idx % 3 == 0:  # 每3个数字打印一次进度
            print(f"  已放置 {idx+1}/{len(digits)} 个数字")
    
    print(f"  ✓ 成功创建，最终尺寸: {canvas.shape}")
    
    return canvas


# 创建多种测试场景
test_scenarios = [
    {
        'name': '简单序列',
        'digits': [1, 2, 3, 4, 5],
        'base_size': 35,
        'spacing': 30
    },
    {
        'name': '长序列',
        'digits':  [9, 8, 7, 6, 5, 4, 3, 2, 1],
        'base_size': 32,
        'spacing': 25
    },
    {
        'name': '电话号码',
        'digits': [1, 3, 8, 0, 0, 1, 3, 8, 0, 0, 0],
        'base_size': 30,
        'spacing': 20
    },
    {
        'name': '门牌号',
        'digits': [2, 0, 2, 5],
        'base_size': 40,
        'spacing': 35
    },
    {
        'name': '密集排列',
        'digits': [5, 5, 5, 5, 5, 5],
        'base_size': 35,
        'spacing': 15
    }
]

test_image_paths = []

for scenario in test_scenarios: 
    img = create_test_image_safe(
        scenario['digits'],
        scenario['base_size'],
        scenario['spacing']
    )
    
    # 添加一些噪声（模拟真实环境）
    noise = np.random.randint(-15, 15, img.shape).astype(np.int16)
    img_noisy = np.clip(img. astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # 保存
    filename = f"test_{scenario['name']. replace(' ', '_')}_{''.join(map(str, scenario['digits']))}.png"
    save_path = test_dir / filename
    cv2.imwrite(str(save_path), img_noisy)
    
    test_image_paths.append({
        'path': save_path,
        'true_digits': ''.join(map(str, scenario['digits'])),
        'name': scenario['name']
    })
    
    print(f"✓ 创建测试图像: {scenario['name']}")

# 3. 逐个识别
print("\n【步骤3】开始识别")
print("-" * 70)

results_dir = project_root / 'results' / 'complete_test'
results_dir.mkdir(parents=True, exist_ok=True)

summary = []

for i, test_case in enumerate(test_image_paths):
    print(f"\n{'='*70}")
    print(f"测试 {i+1}/{len(test_image_paths)}: {test_case['name']}")
    print(f"{'='*70}")
    print(f"真实数字: {test_case['true_digits']}")
    
    # 识别
    result = recognizer.recognize(
        test_case['path'],
        confidence_threshold=0.3,  # 降低阈值以处理困难样本
        debug=False
    )
    
    recognized = result['sequence']
    print(f"识别结果:  {recognized}")
    print(f"检测区域:  {result['num_detected']}")
    print(f"识别数字: {result['num_recognized']}")
    
    # 计算准确率
    if recognized == test_case['true_digits']:
        status = "✓ 完全正确"
        accuracy = 100.0
    else:
        # 计算字符级准确率
        correct = sum(1 for a, b in zip(recognized, test_case['true_digits']) if a == b)
        accuracy = correct / len(test_case['true_digits']) * 100
        status = f"✗ 部分正确 ({accuracy:.1f}%)"
    
    print(f"状态: {status}")
    
    # 可视化
    save_path = results_dir / f"result_{i+1}_{test_case['name']. replace(' ', '_')}.png"
    recognizer.visualize_result(result, save_path=save_path, show=False)
    
    # 记录摘要
    summary.append({
        'name': test_case['name'],
        'true':  test_case['true_digits'],
        'predicted': recognized,
        'accuracy':  accuracy,
        'detected': result['num_detected'],
        'recognized': result['num_recognized']
    })

# 4. 总结报告
print("\n" + "=" * 70)
print("识别总结")
print("=" * 70)

total_accuracy = np.mean([s['accuracy'] for s in summary])

print(f"\n总体准确率: {total_accuracy:.2f}%\n")
print(f"{'场景':<15} {'真实':<15} {'识别':<15} {'准确率':<10}")
print("-" * 60)

for s in summary:
    print(f"{s['name']:<15} {s['true']:<15} {s['predicted']:<15} {s['accuracy']:.1f}%")

# 5. 实际应用示例
print("\n" + "=" * 70)
print("实际应用示例")
print("=" * 70)

print(f"""
✓ 系统已就绪，可以处理：

📱 真实场景: 
  - 手机拍摄的门牌号
  - 扫描的电话号码
  - 截图的数字序列
  - 低质量图片

🔧 使用方法: 

1. Python脚本: 
   
   from src.model import SimpleCNN, get_device
   from src. complete_recognizer import CompleteDigitRecognizer
   
   device = get_device()
   model = SimpleCNN()
   recognizer = CompleteDigitRecognizer(
       model, 
       'models/simple_cnn_best.pth', 
       device
   )
   
   result = recognizer.recognize('your_image.jpg')
   print(f"识别的数字: {{result['sequence']}}")
   recognizer.visualize_result(result)

2. 命令行:
   
   python recognize.py --image your_image.jpg --output result.png

3. 批量处理:
   
   for img in glob.glob('images/*.jpg'):
       result = recognizer.recognize(img)
       print(f"{{img}}: {{result['sequence']}}")

📊 系统特点:
  ✓ 支持任意尺寸图像
  ✓ 自动检测和分割数字
  ✓ 处理粘连、倾斜、噪声
  ✓ 置信度评估
  ✓ 可视化结果

⚙️ 参数调优:
  - confidence_threshold: 调整置信度阈值 (0.3-0.9)
  - min_area:  最小数字区域面积 (50-200)
  - max_aspect:  最大宽高比 (2. 0-4.0)

测试图像位置:  {test_dir}
结果保存位置: {results_dir}
""")

print("=" * 70)
print("完整部署演示完成！")
print("=" * 70)