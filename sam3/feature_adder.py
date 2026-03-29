import torch
import torch.nn as nn
import torch.nn.functional as F
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

class LayerNorm2d_ParamFree(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        # 移除了 self.weight 和 self.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        # 仅仅做纯粹的白化操作，不带任何可学习的仿射变换
        x = (x - u) / torch.sqrt(s + self.eps)
        return x

class FeatureAdder(nn.Module):
    def __init__(self, 
                 in_channels_x: int, 
                 in_channels_cond: int, 
                 out_channels: int = None, 
                 mid_channels: int = None, 
                 use_checkpoint: bool = True):
        """
        基于 SPADE 哲学的预激活残差注入模块：先调制，再非线性映射，最后零卷积输出。
        """
        super().__init__()
        self.use_checkpoint = use_checkpoint
        
        if out_channels is None:
            out_channels = in_channels_x
        if mid_channels is None:
            mid_channels = out_channels
            
        # 1. Param-free normalization
        self.param_free_norm = LayerNorm2d_ParamFree()
        
        # 2. 联合生成 actv 的投影层
        self.proj_x = nn.Conv2d(in_channels_x, mid_channels, kernel_size=3, stride=1, padding=1)
        self.proj_cond = nn.Conv2d(in_channels_cond, mid_channels, kernel_size=3, stride=1, padding=1)
        self.act = nn.GELU()
        
        # 3. 生成 gamma 和 beta 的网络 (注意：这里的输出维度变成了 in_channels_x ！)
        self.mlp_gamma = nn.Conv2d(mid_channels, in_channels_x, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(mid_channels, in_channels_x, kernel_size=3, padding=1)
        
        # 4. 最后的特征投影与重组层 (作用于调制后的特征)
        self.x_out_proj = nn.Sequential(
            nn.Conv2d(in_channels_x, mid_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=1),
            nn.GELU(),
            # 最后一层必须是纯线性卷积，不带激活
            nn.Conv2d(mid_channels, out_channels, kernel_size=1) 
        )
        
        # 5. 执行初始化策略
        self._initialize_weights()

    def _initialize_weights(self):
        """
        新的初始化策略：恢复 SPADE 恒等映射，由最后的 Zero-Conv 兜底保证 0 输出。
        """
        # Gamma 和 Beta 初始化为 0。
        # 配合公式 (1 + gamma)，初始状态下调制输出为 norm(x) * 1 + 0 = norm(x)
        nn.init.zeros_(self.mlp_gamma.weight)
        if self.mlp_gamma.bias is not None:
            nn.init.zeros_(self.mlp_gamma.bias)
            
        nn.init.zeros_(self.mlp_beta.weight)
        if self.mlp_beta.bias is not None:
            nn.init.zeros_(self.mlp_beta.bias)

        # 核心：将最后的投影层权重置 0 (Zero-Convolution)。
        # 这确保了无论前面怎么映射，整个模块的初始输出严格为 0，对主干零干扰。
        last_conv = self.x_out_proj[-1]
        nn.init.zeros_(last_conv.weight)
        if last_conv.bias is not None:
            nn.init.zeros_(last_conv.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            if not x.requires_grad:
                x = x.detach().requires_grad_(True)
            if not cond.requires_grad:
                cond = cond.detach().requires_grad_(True)
                
            return checkpoint(self._forward_impl, x, cond, use_reentrant=False)
        else:
            return self._forward_impl(x, cond)

    def _forward_impl(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # Part 1: 归一化被调制的特征 x
        normalized = self.param_free_norm(x)
        
        # Part 2: 空间分辨率对齐
        assert cond.shape[2:] == x.shape[2:]
            
        # Part 3: 投影并融合特征生成 actv
        actv = self.act(self.proj_x(x) + self.proj_cond(cond))
        
        # Part 4: 预测空间自适应的缩放和平移系数 (在原始通道维度上)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        
        # Part 5: 执行特征调制 (恢复了 1 + gamma，保留 x 的完整特征分布)
        modulated = normalized * (1 + gamma) + beta

        # Part 6: SPADE 哲学 -> 先激活，再做多层映射输出到目标维度
        out = self.act(modulated)
        out = self.x_out_proj(out)
        
        return out