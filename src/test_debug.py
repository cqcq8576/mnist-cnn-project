import torch
print(f"PyTorch version: {torch.__version__}")
print(f"ReduceLROnPlateau signature: {torch.optim.lr_scheduler.ReduceLROnPlateau.__init__.__doc__}")
import inspect
sig = inspect.signature(torch.optim.lr_scheduler.ReduceLROnPlateau.__init__)
print("ReduceLROnPlateau.__init__ 参数:", list(sig.parameters.keys()))