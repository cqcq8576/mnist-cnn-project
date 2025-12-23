from model import SimpleCNN

model = SimpleCNN(num_classes=10)

total_params = 0
print("逐层参数统计:")
print("=" * 60)
for name, param in model.named_parameters():
    params = param.numel()
    total_params += params
    print(f"{name:20s}: {str(param.shape):20s} → {params:,} 参数")

print("=" * 60)
print(f"总参数量: {total_params:,}")