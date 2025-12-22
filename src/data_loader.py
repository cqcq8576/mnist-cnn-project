import torch
from torchvision import datasets,transforms
from torch.utils.data import DataLoader,random_split
import numpy as np

class MNISTDataLoader:

    def __init__(self,data_dir='./data',batch_size=64,val_split=0.2):
        """
        初始化数据加载器类
        
        :param data_dir: 数据存储目录
        :param batch_size: 批次大小
        :param val_split: 验证集比例
        """
        self.data_dir=data_dir
        self.batch_size=batch_size
        self.val_split=val_split

        # 数据转换：转为tensor并归一化到[0,1]
        self.transform=transforms.Compose([transforms.ToTensor()])
        
        self.train_dataset=None
        self.val_dataset=None
        self.test_dataset=None
        self.train_loader=None
        self.val_loader=None
        self.test_loader=None

    def load_data(self):
        print('正在下载MNIST数据集……')

        # 加载完整训练集
        full_train_dataset=datasets.MNIST(
            root=self.data_dir,
            train=True,
            download=True,
            transform=self.transform
        )

        # 加载测试集
        self.test_dataset=datasets.MNIST(
            root=self.data_dir,
            train=False,
            download=True,
            transform=self.transform
        )

        # 划分训练集和验证集(validate)
        train_size=int((1-self.val_split)*len(full_train_dataset))
        val_size=len(full_train_dataset)-train_size

        self.train_dataset,self.val_dataset=random_split(
            full_train_dataset,
            [train_size,val_size],
            generator=torch.Generator().manual_seed(42)
        )

        print(f'数据加载完成！')
        print(f'训练集大小：{len(self.train_dataset)}')
        print(f'验证集大小：{len(self.val_dataset)}')
        print(f'测试集大小：{len(self.test_dataset)}')

        return self
    
    def get_data_loaders(self):
        '''创建数据加载方法'''
        if self.train_dataset is None:
            raise ValueError('请先调用load_data()加载器')
        
        self.train_loader=DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,  #shuffle,打乱的意思
            num_workers=0
        )

        self.val_loader=DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )

        self.test_loader=DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )

        return self.train_loader,self.val_loader,self.test_loader
    
    def get_data_info(self):
        '''获取数据集的基本信息'''
        if self.train_dataset is None:
            raise ValueError('请先调用load_data()加载数据')
        
        # 获取一个样本查看形状
        sample_data, sample_label = self.train_dataset[0]

        info={
            'train_size':len(self.train_dataset),
            'val_size':len(self.val_dataset),
            'test_size':len(self.test_dataset),
            'image_shape':sample_data.shape,
            'num_classes':10, # 因为阿拉伯数字就是0~9十个
            'data_type':sample_data.dtype
        }

        return info
    
def get_class_distribution(dataset):
    """
    统计数据集中各类别的分布
    输入：dataset，是数据集
    输出：dict，是各类别的样本数量
    """
    labels=[]
    # 遍历数据集收集所有标签
    for _,label in dataset:
        labels.append(label)
    labels=np.array(labels)

    # 统计每个类别的数量
    distribution={}
    for i in range(10):
        distribution[i]=np.sum(labels==i)
    
    return distribution
    
if __name__ =='__main__':
    print('='*50)
    print('测试MNIST数据加载器')
    print('='*50)

    mnist_loader=MNISTDataLoader(data_dir='../data',batch_size=64)
    mnist_loader.load_data()

    info=mnist_loader.get_data_info()
    print('\n数据集信息：')
    for key,value in info.items():
        print(f'{key}:{value}')
    
    train_loader,val_loader,test_loader=mnist_loader.get_data_loaders()

    images,labels=next(iter(train_loader))
    print(f'\n一个batch的形状')
    print(f'图像：{images.shape}')
    print(f'标签：{labels.shape}')

    print('\n数据加载成功')

