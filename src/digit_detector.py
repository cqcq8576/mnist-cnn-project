"""
智能数字检测模块
处理粘连、重叠、各种尺寸的数字
"""
import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class DigitRegion:
    """数字区域"""
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    contour: np.ndarray
    area: int
    aspect_ratio: float
    center_x: int


class DigitDetector: 
    """智能数字检测器"""
    
    def __init__(self, min_area=50, max_area=50000, 
                 min_aspect=0.2, max_aspect=3.0):
        """
        初始化检测器
        
        Args: 
            min_area: 最小区域面积
            max_area: 最大区域面积
            min_aspect: 最小宽高比
            max_aspect: 最大宽高比
        """
        self.min_area = min_area
        self.max_area = max_area
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect
    
    def detect(self, binary_image, original_image=None, debug=False):
        """
        检测图像中的所有数字区域
        
        Args: 
            binary_image: 二值图像
            original_image:  原始图像（用于可视化）
            debug: 是否显示调试信息
            
        Returns:
            List[DigitRegion]: 检测到的数字区域列表
        """
        # 查找轮廓
        contours, hierarchy = cv2.findContours(
            binary_image, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        digit_regions = []
        
        for contour in contours: 
            # 获取边界框
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            
            # 基本过滤
            if area < self.min_area or area > self.max_area:
                continue
            
            # 宽高比过滤
            aspect_ratio = w / h if h > 0 else 0
            if aspect_ratio < self.min_aspect or aspect_ratio > self.max_aspect:
                # 可能是粘连数字，尝试分割
                if aspect_ratio > self.max_aspect:
                    # 宽度过大，可能是多个数字粘连
                    sub_regions = self._split_wide_region(
                        binary_image, x, y, w, h
                    )
                    digit_regions.extend(sub_regions)
                continue
            
            # 创建数字区域对象
            region = DigitRegion(
                bbox=(x, y, w, h),
                contour=contour,
                area=area,
                aspect_ratio=aspect_ratio,
                center_x=x + w // 2
            )
            
            digit_regions.append(region)
        
        # 按从左到右排序
        digit_regions.sort(key=lambda r: r. center_x)
        
        if debug:
            self._visualize_detection(original_image or binary_image, digit_regions)
        
        return digit_regions
    
    def _split_wide_region(self, binary, x, y, w, h):
        """
        分割宽度过大的区域（粘连数字）
        
        Args: 
            binary: 二值图像
            x, y, w, h: 区域坐标
            
        Returns: 
            List[DigitRegion]: 分割后的区域列表
        """
        roi = binary[y:y+h, x:x+w]
        
        # 计算垂直投影
        vertical_projection = np.sum(roi, axis=0)
        
        # 查找分割点（投影值最小的位置）
        avg_projection = np.mean(vertical_projection)
        threshold = avg_projection * 0.3
        
        # 找到所有低于阈值的位置
        split_points = []
        in_gap = False
        gap_start = 0
        
        for i, val in enumerate(vertical_projection):
            if val < threshold:
                if not in_gap:
                    gap_start = i
                    in_gap = True
            else:
                if in_gap:
                    gap_center = (gap_start + i) // 2
                    split_points.append(gap_center)
                    in_gap = False
        
        # 如果没有找到分割点，估算分割位置
        if len(split_points) == 0:
            estimated_digit_width = h * 0.8  # 假设数字宽高比约为0.8
            num_digits = int(w / estimated_digit_width + 0.5)
            if num_digits > 1:
                for i in range(1, num_digits):
                    split_points.append(int(w * i / num_digits))
        
        # 创建分割后的区域
        regions = []
        prev_x = 0
        
        for split_x in split_points + [w]:
            if split_x - prev_x < self.min_area ** 0.5:
                prev_x = split_x
                continue
            
            sub_x = x + prev_x
            sub_w = split_x - prev_x
            
            region = DigitRegion(
                bbox=(sub_x, y, sub_w, h),
                contour=None,
                area=sub_w * h,
                aspect_ratio=sub_w / h,
                center_x=sub_x + sub_w // 2
            )
            regions.append(region)
            
            prev_x = split_x
        
        return regions
    
    def extract_digit_image(self, image, region, target_size=28, padding=4):
        """
        从图像中提取单个数字，并标准化为28×28
        
        Args: 
            image: 原始图像（灰度或二值）
            region: DigitRegion对象
            target_size: 目标尺寸
            padding: 填充大小
            
        Returns: 
            numpy.ndarray: 标准化后的数字图像
        """
        x, y, w, h = region.bbox
        
        # 提取ROI
        roi = image[y:y+h, x:x+w]
        
        # 确保是二值图像
        if roi. max() > 1:
            _, roi = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY)
        
        # 去除边缘空白
        coords = cv2.findNonZero(roi)
        if coords is not None:
            x2, y2, w2, h2 = cv2.boundingRect(coords)
            roi = roi[y2:y2+h2, x2:x2+w2]
        
        # 计算缩放比例，保持宽高比
        h_roi, w_roi = roi.shape
        scale = (target_size - 2 * padding) / max(h_roi, w_roi)
        
        new_w = int(w_roi * scale)
        new_h = int(h_roi * scale)
        
        # 缩放
        resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # 创建正方形画布
        canvas = np.zeros((target_size, target_size), dtype=np.uint8)
        
        # 居中放置
        y_offset = (target_size - new_h) // 2
        x_offset = (target_size - new_w) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        # 归一化
        canvas = canvas.astype(np.float32) / 255.0
        
        return canvas
    
    def _visualize_detection(self, image, regions):
        """可视化检测结果"""
        import matplotlib.pyplot as plt
        
        if len(image.shape) == 2:
            vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis = image.copy()
        
        for i, region in enumerate(regions):
            x, y, w, h = region.bbox
            cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(vis, str(i), (x, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        plt.figure(figsize=(12, 6))
        plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        plt.title(f'检测到 {len(regions)} 个数字区域')
        plt.axis('off')
        plt.show()


if __name__ == "__main__":
    detector = DigitDetector()
    print("智能数字检测模块已加载")