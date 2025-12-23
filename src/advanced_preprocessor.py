"""
高级图像预处理模块
处理任意尺寸、任意质量的输入图像
"""
import cv2
import numpy as np
from PIL import Image


class AdvancedPreprocessor:
    """高级图像预处理器"""
    
    def __init__(self):
        pass
    
    def preprocess(self, image_path, debug=False):
        """
        预处理图像
        
        Args:
            image_path: 图像路径或numpy数组
            debug: 是否显示中间步骤
            
        Returns: 
            numpy. ndarray: 预处理后的二值图像
        """
        # 读取图像
        if isinstance(image_path, str):
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"无法读取图像: {image_path}")
        else:
            img = image_path
        
        # 转为灰度图
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img. copy()
        
        if debug:
            self._show_debug("1. 原始灰度图", gray)
        
        # 去噪
        denoised = cv2.fastNlMeansDenoising(gray, None, h=10)
        
        if debug:
            self._show_debug("2. 去噪后", denoised)
        
        # 自适应二值化
        binary = cv2.adaptiveThreshold(
            denoised, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=11, 
            C=2
        )
        
        if debug: 
            self._show_debug("3. 自适应二值化", binary)
        
        # 形态学操作 - 去除小噪点
        kernel_small = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small)
        
        if debug:
            self._show_debug("4. 去除噪点", binary)
        
        # 膨胀操作 - 连接断裂的笔画
        kernel_dilate = np.ones((2, 2), np.uint8)
        binary = cv2.dilate(binary, kernel_dilate, iterations=1)
        
        if debug: 
            self._show_debug("5. 连接笔画", binary)
        
        return binary, gray
    
    def deskew(self, image):
        """
        倾斜校正
        
        Args: 
            image: 输入图像
            
        Returns:
            numpy.ndarray: 校正后的图像
        """
        coords = np.column_stack(np.where(image > 0))
        if len(coords) == 0:
            return image
        
        angle = cv2.minAreaRect(coords)[-1]
        
        # 调整角度
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        
        # 只校正明显倾斜的情况
        if abs(angle) < 0.5:
            return image
        
        # 旋转图像
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        return rotated
    
    def _show_debug(self, title, img):
        """显示调试图像"""
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')
        plt.show()


if __name__ == "__main__": 
    preprocessor = AdvancedPreprocessor()
    print("高级预处理模块已加载")