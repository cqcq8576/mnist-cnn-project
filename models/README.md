# 模型文件目录

此目录用于存储训练好的模型权重。

## 文件说明

- `simple_cnn_initial.pth` - 初始化的模型（未训练）
- `simple_cnn_best.pth` - 训练过程中验证集上表现最好的模型
- `simple_cnn_final.pth` - 训练结束时的最终模型

## 加载模型示例

```python
import torch
from src.model import SimpleCNN

# 加载模型
checkpoint = torch.load('models/simple_cnn_best.pth')
model = SimpleCNN(num_classes=10)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()