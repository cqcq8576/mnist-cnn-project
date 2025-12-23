"""
完整的多数字识别系统
整合预处理、检测、识别
"""
import torch
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt


class CompleteDigitRecognizer:
    """完整的数字识别系统"""
    
    def __init__(self, model, model_path, device):
        """
        初始化识别系统
        
        Args: 
            model: CNN模型实例
            model_path: 模型权重路径
            device: 运行设备
        """
        from advanced_preprocessor import AdvancedPreprocessor
        from digit_detector import DigitDetector
        
        self.device = device
        self.model = model. to(device)
        
        # 加载模型
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # 初始化模块
        self.preprocessor = AdvancedPreprocessor()
        self.detector = DigitDetector(
            min_area=100,
            max_area=50000,
            min_aspect=0.2,
            max_aspect=2.5
        )
        
        print(f"✓ 完整识别系统已加载")
        print(f"  设备: {device}")
        print(f"  模型准确率: {checkpoint. get('val_acc', 'N/A')}")
    
    def recognize(self, image_path, confidence_threshold=0.5, debug=False):
        """
        识别图像中的所有数字
        
        Args:
            image_path: 图像路径
            confidence_threshold: 置信度阈值
            debug: 是否显示调试信息
            
        Returns:
            dict: 识别结果
        """
        # 1. 读取图像
        original = cv2.imread(str(image_path))
        if original is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        if debug:
            print(f"图像尺寸: {original.shape}")
        
        # 2. 预处理
        binary, gray = self.preprocessor. preprocess(original, debug=debug)
        
        # 3. 检测数字区域
        regions = self.detector.detect(binary, original, debug=debug)
        
        if debug:
            print(f"检测到 {len(regions)} 个候选区域")
        
        # 4. 逐个识别
        results = []
        
        for i, region in enumerate(regions):
            # 提取数字图像
            digit_img = self.detector.extract_digit_image(binary, region)
            
            if debug and i < 5:  # 只显示前5个
                plt.figure(figsize=(3, 3))
                plt. imshow(digit_img, cmap='gray')
                plt. title(f'Region {i}')
                plt.axis('off')
                plt.show()
            
            # 转为tensor
            tensor = torch.from_numpy(digit_img).unsqueeze(0).unsqueeze(0)
            tensor = tensor.to(self.device)
            
            # 推理
            with torch.no_grad():
                output = self.model(tensor)
                probs = torch.softmax(output, dim=1)
                confidence, predicted = probs.max(1)
            
            conf_value = confidence. item()
            pred_digit = predicted.item()
            
            # 置信度过滤
            if conf_value < confidence_threshold:
                if debug:
                    print(f"  区域 {i}:  置信度过低 ({conf_value:.2f}) - 跳过")
                continue
            
            results.append({
                'digit': pred_digit,
                'confidence': conf_value,
                'bbox': region.bbox,
                'position': i
            })
            
            if debug:
                print(f"  区域 {i}: 数字={pred_digit}, 置信度={conf_value:.3f}")
        
        # 5. 组合结果
        digit_sequence = ''.join(str(r['digit']) for r in results)
        
        return {
            'sequence': digit_sequence,
            'digits': results,
            'num_detected': len(regions),
            'num_recognized': len(results),
            'original_image': original,
            'binary_image': binary
        }
    
    def visualize_result(self, result, save_path=None, show=True):
        """
        可视化识别结果
        
        Args:
            result:  recognize()返回的结果
            save_path: 保存路径
            show: 是否显示
        """
        img = result['original_image']. copy()
        
        # 绘制检测框和标签
        for item in result['digits']:
            x, y, w, h = item['bbox']
            digit = item['digit']
            conf = item['confidence']
            
            # 颜色根据置信度
            if conf > 0.95:
                color = (0, 255, 0)  # 绿色
            elif conf > 0.8:
                color = (0, 255, 255)  # 黄色
            else:
                color = (0, 165, 255)  # 橙色
            
            # 绘制矩形
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            
            # 绘制标签
            label = f"{digit}"
            conf_label = f"{conf*100:.0f}%"
            
            # 数字标签（较大）
            cv2.putText(img, label, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            
            # 置信度标签（较小）
            cv2.putText(img, conf_label, (x, y+h+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # 创建显示图像
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 原图+标注
        axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0].set_title('识别结果', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # 二值图
        axes[1].imshow(result['binary_image'], cmap='gray')
        axes[1].set_title('预处理后的二值图', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        # 添加识别序列
        sequence = result['sequence']
        fig.text(0.5, 0.02, f'识别的数字序列: {sequence}',
                ha='center', fontsize=18, fontweight='bold', color='blue')
        
        # 添加统计信息
        stats = f"检测:  {result['num_detected']} | 识别: {result['num_recognized']}"
        fig.text(0.5, 0.95, stats,
                ha='center', fontsize=12, color='gray')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ 结果保存到: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return img


if __name__ == "__main__": 
    print("完整识别系统模块已加载")