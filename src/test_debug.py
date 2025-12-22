import torch

print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA是否可用: {torch.cuda.is_available()}")
print(f"CUDA版本:  {torch.version.cuda}")
print(f"PyTorch是否支持CUDA: {torch.backends.cudnn.enabled if torch.cuda.is_available() else 'N/A'}")