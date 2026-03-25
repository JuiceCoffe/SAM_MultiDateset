import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class DeepVPTTuner(nn.Module):
    """
    ???? Deep VPT ?????????
    1. ???? Transformer Block ?????? Prompts
    """
    def __init__(self, num_layers: int, num_prompts: int, embed_dim: int):
        super().__init__()
        # ???????? Prompts
        # ??: [??, Prompt??, ????]
        self.prompts = nn.Parameter(
            torch.randn(num_layers, num_prompts, embed_dim) * 0.02
        )


class PromptInjector(nn.Module):
    def __init__(self, original_blocks: nn.Module, num_prompts: int = 10, embed_dim: int = 1280, num_skip: int = 1):
        super().__init__()
        self.original_blocks = original_blocks
        self.num_prompts = num_prompts
        self.num_layers = len(original_blocks)
        self.num_skip = num_skip
        self.embed_dim = embed_dim

        self.is_active = True
        self.grad_checkpointing = False
        self.supports_gradient_checkpointing = True

        self.tuner = DeepVPTTuner(self.num_layers, num_prompts, embed_dim)


    def _forward_block(self, x: torch.Tensor, block_idx: int):
        B = x.shape[0]
        block = self.original_blocks[block_idx]
        layer_prompts = self.tuner.prompts[block_idx].unsqueeze(0).expand(B, -1, -1)

        skip_tokens = x[:, :self.num_skip, :]
        patch_tokens = x[:, self.num_skip:, :]
        x_in = torch.cat([skip_tokens, layer_prompts, patch_tokens], dim=1)

        x_out = block(x_in)

        skip_tokens_out = x_out[:, :self.num_skip, :]
        patch_tokens_out = x_out[:, self.num_skip + self.num_prompts:, :]
        return torch.cat([skip_tokens_out, patch_tokens_out], dim=1)

    def forward(self, x: torch.Tensor):
        if not self.is_active:
            for block in self.original_blocks:
                x = block(x)
            return x

        use_grad_ckpt = self.training and self.grad_checkpointing and x.requires_grad

        for i in range(self.num_layers):
            if use_grad_ckpt:
                x = checkpoint(
                    lambda hidden_states, idx=i: self._forward_block(hidden_states, idx),
                    x,
                    use_reentrant=False,
                )
            else:
                x = self._forward_block(x, i)

        return x
