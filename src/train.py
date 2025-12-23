"""
train 的 Docstring
模型训练模块
"""
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time

class Trainer:
    """模型训练器"""

    def __init__(self,model,train_loader,val_loader,device,learning_rate=0.001,epochs=10):
        """
        __init__ 的 Docstring
        初始化训练器
        :param model: 使用的神经网络模型
        :param train_loader: 训练数据加载器
        :param val_loader: 验证数据加载器
        :param device: 运行设备，cuda还是cpu
        :param learning_rate: 学习率
        :param epochs: 训练轮数
        """
        self.model=model.to(device)
        self.train_loader=train_loader
        self.val_loader=val_loader
        self.device=device
        self.epochs=epochs
        # 损失函数，这里使用分类问题常用的交叉熵
        self.criterion=nn.CrossEntropyLoss()
        # 优化器，这里选择Adam优化，自适应学习率
        self.optimizer=optim.Adam(model.parameters(),lr=learning_rate)
        # 学习率调度器，当损失不再下降时降低学习率
        self.scheduler=optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min', # 监控指标越小越好
            factor=0.5, # 学习率衰减因子
            patience=3, # 容忍三个epoch没有改善
        )
        # 记录训练历史
        self.history={
            'train_loss':[],
            'train_acc':[],
            'val_loss':[],
            'val_acc':[],
            'learning_rates':[]
        }
        # 最佳模型记录
        self.best_val_acc=0.0
        self.best_epoch=0

    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        running_loss=0.0
        correct=0
        total=0

        # 使用tqdm显示进度条
        pbar=tqdm(self.train_loader,desc='Training',leave=False)

        for images,labels in pbar:
            # 数据移到设备上
            images=images.to(self.device)
            labels=labels.to(self.device)
            # 前向传播
            outputs=self.model(images)
            loss=self.criterion(outputs,labels)
            # 反向传播和优化
            self.optimizer.zero_grad() # 清空梯度
            loss.backward() # 反向传播计算梯度
            self.optimizer.step() # 更新参数
            # 统计
            running_loss+=loss.item()*images.size(0)
            _,predicted=outputs.max(1)
            total+=labels.size(0)
            correct+=predicted.eq(labels).sum().item()
            # 更新进度条
            pbar.set_postfix({
                'loss':f'{loss.item():.4f}',
                'acc':f'{100.*correct/total:.2f}%'
            })
        # 计算平均损失和准确率
        epoch_loss=running_loss/total
        epoch_acc=100.*correct/total
        return epoch_loss,epoch_acc
    
    def validate(self):
        """在验证集上评估"""
        self.model.eval()
        running_loss=0.0
        correct=0
        total=0
        with torch.no_grad():
            pbar=tqdm(self.val_loader,desc='Validation',leave=False)
            for images,labels in pbar:
                images=images.to(self.device)
                labels=labels.to(self.device)
                # 前向传播
                outputs=self.model(images)
                loss=self.criterion(outputs,labels)
                # 统计
                running_loss+=loss.item()*images.size(0)
                _,predicted=outputs.max(1)
                total+=labels.size(0)
                correct+=predicted.eq(labels).sum().item()
                # 更新进度条
                pbar.set_postfix({
                    'loss':f'{loss.item():.4f}',
                    'acc':f'{100.*correct/total:.2f}%'
                })
            # 计算平均损失和准确率
            epoch_loss=running_loss/total
            epoch_acc=100.*correct/total
            return epoch_loss,epoch_acc
        
    def train(self,save_path='../models/simple_cnn_best.pth'):
        """
        完整的训练流程
        
        :param save_path: 最佳模型保存路径
        """
        print("=" * 70)
        print("开始训练")
        print("=" * 70)
        print(f"设备: {self.device}")
        print(f"训练样本数: {len(self. train_loader.dataset)}")
        print(f"验证样本数: {len(self. val_loader.dataset)}")
        print(f"Batch大小: {self.train_loader.batch_size}")
        print(f"训练轮数: {self. epochs}")
        print(f"学习率: {self.optimizer.param_groups[0]['lr']}")
        print("=" * 70)
        
        start_time = time.time()
        
        for epoch in range(self. epochs):
            epoch_start = time.time()
            
            print(f"\nEpoch {epoch + 1}/{self.epochs}")
            print("-" * 70)
            
            # 训练
            train_loss, train_acc = self.train_epoch()
            
            # 验证
            val_loss, val_acc = self.validate()
            
            # 学习率调度
            self. scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(current_lr)
            
            # 计算epoch用时
            epoch_time = time.time() - epoch_start
            
            # 打印结果
            print(f"训练集 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            print(f"验证集 - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
            print(f"学习率: {current_lr:.6f}")
            print(f"用时: {epoch_time:.2f}秒")
            
            # 保存最佳模型
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch + 1
                
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                    'history': self.history
                }, save_path)
                
                print(f"✓ 保存最佳模型 (验证准确率: {val_acc:.2f}%)")
        
        # 训练完成
        total_time = time.time() - start_time
        
        print("\n" + "=" * 70)
        print("训练完成！")
        print("=" * 70)
        print(f"总用时: {total_time / 60:.2f}分钟")
        print(f"最佳验证准确率: {self.best_val_acc:.2f}% (Epoch {self.best_epoch})")
        print(f"最佳模型保存在: {save_path}")
        print("=" * 70)
        
        return self.history       

def load_checkpoint(model, checkpoint_path, device):
    """
    加载训练好的模型
    
    Args:
        model: 模型实例
        checkpoint_path: checkpoint文件路径
        device:  设备
        
    Returns:
        model: 加载了权重的模型
        history: 训练历史
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    history = checkpoint. get('history', None)
    
    print(f"✓ 模型加载成功")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  验证准确率: {checkpoint['val_acc']:.2f}%")
    print(f"  验证损失: {checkpoint['val_loss']:.4f}")
    
    return model, history         

if __name__ == "__main__": 
    # 测试训练模块
    print("=" * 50)
    print("测试训练模块")
    print("=" * 50)
    
    from model import SimpleCNN, get_device
    from data_loader import MNISTDataLoader
    
    # 加载数据
    print("\n加载数据...")
    mnist_loader = MNISTDataLoader(data_dir='../data', batch_size=64)
    mnist_loader.load_data()
    train_loader, val_loader, test_loader = mnist_loader. get_data_loaders()
    
    # 创建模型
    print("创建模型...")
    device = get_device()
    model = SimpleCNN(num_classes=10)
    
    # 创建训练器（只训练1个epoch测试）
    print("创建训练器...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=0.001,
        epochs=1
    )
    
    # 训练
    print("\n开始测试训练...")
    history = trainer.train(save_path='../models/test_model.pth')
    
    print("\n✓ 训练模块测试成功！")

        