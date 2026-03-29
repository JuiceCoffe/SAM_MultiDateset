import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class FeatureAdder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None, use_checkpoint: bool = True):
        """
        带零初始化和梯度检查点 (Checkpoint) 的特征转化模块。
        
        Args:
            in_channels: 输入通道数 D1
            out_channels: 输出通道数 D2
            mid_channels: 中间层通道数
            use_checkpoint: 是否启用显存节省技术
        """
        super().__init__()
        self.use_checkpoint = use_checkpoint
        
        if mid_channels is None:
            mid_channels = max(in_channels // 2, out_channels)
            
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            LayerNorm2d(mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )
        
        self._zero_initialize_last_layer()

    def _zero_initialize_last_layer(self):
        """仅将最后一层的权重和偏置初始化为 0"""
        last_conv = self.block[-1]
        nn.init.zeros_(last_conv.weight)
        if last_conv.bias is not None:
            nn.init.zeros_(last_conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 仅在训练模式且启用了 use_checkpoint 时生效
        # 推理阶段 (model.eval()) 不需要节省显存，直接跑正常前向传播即可
        if self.use_checkpoint and self.training:
            
            # 【核心技巧】：解决冻结主干网络导致的梯度断流问题
            # 如果输入 x 是从冻结的 SAM ViT 传过来的，x.requires_grad 会是 False。
            # 这会导致 checkpoint 认为整个过程不需要求导。我们需要强制将其标记为需要求导。
            if not x.requires_grad:
                x = x.detach().requires_grad_(True)
            
            # 使用 PyTorch 提供的 checkpoint 函数
            # use_reentrant=False 是 PyTorch 1.11+ 推荐的新版安全机制
            return checkpoint(self._forward_impl, x, use_reentrant=False)
        else:
            return self.block(x)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        # checkpoint 需要调用一个具体的函数，将 block 的执行封装进这里
        return self.block(x)