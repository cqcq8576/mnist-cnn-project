"""
model 的 Docstring
CNN模型定义模块
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """
    SimpleCNN 的 Docstring
    简单的卷积神经网络模型
    Conv1->ReLU->MaxPool->Conv2->MaxPool->Flatten->FC1->ReLU->FC2->Softmax
    """

    def __init__(self,num_classes=10):
        """
        __init__ 的 Docstring
        
        :param num_classes: 分类类别数，MNIST是10
        """
        super(SimpleCNN,self).__init__()

        # 第一层卷积层
        # 输入：(batch_size,1,28,28)，我们的MNIST数据集图片是一些灰度图，in_channel=1
        # 输出：(batch_size,32,26,26)，先把out_channel设为32，卷积核大小设为3*3，于是会有26*26个卷积核
        self.conv1=nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=0 # 这里先不填充，会让输出尺寸变小为26*26
        )

        # 第一层池化层
        # 输入：(batch_size,32,26,26)
        # 输出：(batch_size,32,13,13)
        self.pool1=nn.MaxPool2d(
            kernel_size=2,
            stride=2  # 每个2*2的窗口只采一个样，26*26变成了13*13
        )

        # 第二层卷积层
        # 输入：(batch_size,32,13,13)
        # 输出：(batch_size,64,11,11)
        self.conv2=nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=0
        )

        # 第二层池化层
        # 输入：(batch_size,64,11,11)
        # 输出：(batch_size,64,5,5)
        self.pool2=nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # 全连接层
        # 输入：64*5*5=1600，把张量拉平成向量，1600维
        # 输出：128
        self.fc1=nn.Linear(
            in_features=64*5*5,
            out_features=128
        )

        # 输出层，还是个全连接层
        # 输入：128
        # 输出：10，对应10个数字类别
        self.fc2=nn.Linear(
            in_features=128,
            out_features=num_classes
        )

    def forward(self,x):
        """
        前向传播
        :param x: 输入的图像(batch_size,1,28,28)
        输出是(batch_size,10)
        """
        # 第一层卷积+relu激活+池化
        x=self.conv1(x)
        x=F.relu(x)
        x=self.pool1(x)
        # 第二层卷积+relu激活+池化
        x=self.conv2(x)
        x=F.relu(x)
        x=self.pool2(x)
        # 展平为向量
        x=x.view(x.size(0),-1)
        # 全连接层+relu激活
        x=self.fc1(x)
        x=F.relu(x)
        # 输出层
        x=self.fc2(x)
        # 这里不用softmax因为pytorch的cross entropy loss已经包含
        return x
    
    def get_model_summary(self):
        """获取模型概要信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p. numel() for p in self.parameters() if p.requires_grad)
        
        summary = f"""
        ============================================
        模型结构概要
        ============================================
        第1层卷积:  Conv2d(1, 32, kernel=3)
            - 参数量: {self.conv1.weight.numel() + self.conv1.bias.numel()}
            - 作用: 提取低级特征（边缘、纹理）
        
        第1层池化: MaxPool2d(kernel=2, stride=2)
            - 参数量: 0
            - 作用: 降维，增强特征不变性
        
        第2层卷积: Conv2d(32, 64, kernel=3)
            - 参数量: {self.conv2.weight.numel() + self.conv2.bias.numel()}
            - 作用: 提取高级特征（形状、结构）
        
        第2层池化: MaxPool2d(kernel=2, stride=2)
            - 参数量: 0
            - 作用: 进一步降维
        
        全连接层1: Linear(1600, 128)
            - 参数量: {self. fc1.weight.numel() + self.fc1.bias. numel()}
            - 作用: 特征组合与抽象
        
        输出层: Linear(128, 10)
            - 参数量: {self.fc2.weight.numel() + self.fc2.bias.numel()}
            - 作用: 分类决策
        
        ============================================
        总参数量: {total_params:,}
        可训练参数:  {trainable_params:,}
        ============================================
        """
        return summary
    
def get_device():
    """
    获取可用的设备（GPU或CPU）
    
    Returns:
        torch.device: 可用的设备
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✓ 使用GPU:  {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("✓ 使用CPU")
    return device


def count_parameters(model):
    """
    统计模型参数量
    
    Args:
        model: PyTorch模型
        
    Returns:
        tuple: (总参数量, 可训练参数量)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__": 
    # 测试模型
    print("=" * 50)
    print("测试SimpleCNN模型")
    print("=" * 50)
    
    # 创建模型
    model = SimpleCNN(num_classes=10)
    print("\n✓ 模型创建成功")
    
    # 打印模型结构
    print("\n模型结构:")
    print(model)
    
    # 打印模型概要
    print(model.get_model_summary())
    
    # 获取设备
    device = get_device()
    model = model.to(device)
    
    # 测试前向传播
    print("\n测试前向传播:")
    batch_size = 4
    test_input = torch.randn(batch_size, 1, 28, 28).to(device)
    print(f"输入形状:  {test_input.shape}")
    
    output = model(test_input)
    print(f"输出形状: {output.shape}")
    print(f"输出示例:\n{output[0]}")
    
    # 应用Softmax查看概率分布
    probabilities = F.softmax(output, dim=1)
    print(f"\n概率分布示例:\n{probabilities[0]}")
    print(f"概率和: {probabilities[0].sum():.4f}")
    
    # 预测类别
    predicted = output.argmax(dim=1)
    print(f"\n预测类别: {predicted}")
    
    print("\n" + "=" * 50)
    print("模型测试成功！")
    print("=" * 50)    


       

