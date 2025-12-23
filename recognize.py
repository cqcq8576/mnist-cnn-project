#!/usr/bin/env python3
"""
命令行识别工具
使用:  python recognize.py --image your_image.jpg
"""
import sys
sys.path.append('src')

import argparse
from pathlib import Path
from model import SimpleCNN, get_device
from complete_recognizer import CompleteDigitRecognizer


def main():
    parser = argparse.ArgumentParser(description='手写数字识别')
    parser.add_argument('--image', '-i', required=True, help='输入图像路径')
    parser.add_argument('--output', '-o', help='输出图像路径')
    parser.add_argument('--threshold', '-t', type=float, default=0.5, 
                       help='置信度阈值 (default: 0.5)')
    parser.add_argument('--debug', '-d', action='store_true', 
                       help='显示调试信息')
    parser.add_argument('--no-show', action='store_true', 
                       help='不显示结果图像')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not Path(args.image).exists():
        print(f"错误: 文件不存在 - {args.image}")
        sys.exit(1)
    
    # 加载模型
    print("加载模型...")
    device = get_device()
    model = SimpleCNN()
    recognizer = CompleteDigitRecognizer(
        model,
        'models/simple_cnn_best.pth',
        device
    )
    
    # 识别
    print(f"识别图像:  {args.image}")
    result = recognizer.recognize(
        args.image,
        confidence_threshold=args.threshold,
        debug=args.debug
    )
    
    # 输出结果
    print("\n" + "=" * 50)
    print(f"识别的数字序列: {result['sequence']}")
    print(f"检测到的区域: {result['num_detected']}")
    print(f"成功识别: {result['num_recognized']}")
    print("=" * 50)
    
    # 详细信息
    print("\n详细信息:")
    for i, digit_info in enumerate(result['digits']):
        print(f"  位置 {i+1}: 数字 {digit_info['digit']} "
              f"(置信度: {digit_info['confidence']*100:.1f}%)")
    
    # 可视化
    output_path = args.output or f"result_{Path(args.image).stem}.png"
    recognizer.visualize_result(
        result,
        save_path=output_path,
        show=not args.no_show
    )
    
    print(f"\n✓ 结果已保存: {output_path}")


if __name__ == '__main__':
    main()