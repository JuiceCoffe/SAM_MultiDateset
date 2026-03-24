import torch
import torch.nn as nn

# ==========================================
# 【重构】：专为 Deep VPT 设计的参数管理器
# ==========================================
class DeepVPTTuner(nn.Module):
    """
    管理所有 Deep VPT 相关的可训练参数：
    1. 为每一层 Transformer Block 初始化独立的 Prompts
    """
    def __init__(self, num_layers: int, num_prompts: int, embed_dim: int):
        super().__init__()
        # 统一管理所有层的 Prompts
        # 形状: [层数, Prompt数量, 特征维度]
        self.prompts = nn.Parameter(
            torch.randn(num_layers, num_prompts, embed_dim) * 0.02
        )

# ==========================================
# 主模块 (保留接口，接入参数管理器)
# ==========================================
class PromptInjector(nn.Module):
    def __init__(self, original_blocks: nn.Module, num_prompts: int = 10, embed_dim: int = 1280, num_skip: int = 1):
        super().__init__()
        self.original_blocks = original_blocks
        self.num_prompts = num_prompts
        self.num_layers = len(original_blocks)
        self.num_skip = num_skip  
        self.embed_dim = embed_dim
        
        self.is_active = True 
        
        # 🌟 实例化你的参数管理器
        self.tuner = DeepVPTTuner(self.num_layers, num_prompts, embed_dim)
        
        # 🌟 移除冻结代码：Backbone 原有参数默认 requires_grad=True，全程参与梯度更新

    def forward(self, x: torch.Tensor):
        if not self.is_active:
            for block in self.original_blocks:
                x = block(x)
            return x

        B = x.shape[0]

        for i, block in enumerate(self.original_blocks):
            layer_prompts = self.tuner.prompts[i].unsqueeze(0).expand(B, -1, -1)
            
            skip_tokens = x[:, :self.num_skip, :]
            patch_tokens = x[:, self.num_skip:, :]
            x_in = torch.cat([skip_tokens, layer_prompts, patch_tokens], dim=1)
            
            x_out = block(x_in)
            
            skip_tokens_out = x_out[:, :self.num_skip, :]
            patch_tokens_out = x_out[:, self.num_skip + self.num_prompts:, :]
            x = torch.cat([skip_tokens_out, patch_tokens_out], dim=1)

        return x