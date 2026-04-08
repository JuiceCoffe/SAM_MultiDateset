import math
from typing import List, Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class DeepVPTTuner(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_prompts: int,
        embed_dim: int,
        patch_size: int = 16,
        project_dim: int = -1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_prompts = num_prompts
        self.embed_dim = embed_dim

        self.prompt_dropout = nn.Dropout(dropout)
        if project_dim > -1:
            prompt_dim = project_dim
            self.prompt_proj = nn.Linear(prompt_dim, embed_dim)
            nn.init.kaiming_normal_(self.prompt_proj.weight, a=0, mode="fan_out")
        else:
            prompt_dim = embed_dim
            self.prompt_proj = nn.Identity()

        val = math.sqrt(6.0 / float(3 * patch_size * patch_size + prompt_dim))
        self.prompts = nn.Parameter(torch.zeros(num_layers, num_prompts, prompt_dim))
        nn.init.uniform_(self.prompts.data, -val, val)


class DeepAttnQueryTuner(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_queries: int,
        embed_dim: int,
        text_feature_dim: int,
        query_dim: int = -1,
        num_heads: int = 8,
        patch_size: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_queries = num_queries
        self.embed_dim = embed_dim
        self.query_dim = embed_dim if query_dim is None or int(query_dim) <= 0 else int(query_dim)

        if self.query_dim % num_heads != 0:
            raise ValueError(
                f"Attn query dim must be divisible by the number of heads, got query_dim={self.query_dim}, num_heads={num_heads}"
            )

        self.query_dropout = nn.Dropout(dropout)
        self.image_proj = (
            nn.Identity() if self.query_dim == embed_dim else nn.Linear(embed_dim, self.query_dim)
        )
        self.text_proj = nn.Linear(text_feature_dim, self.query_dim)
        self.output_proj = (
            nn.Identity() if self.query_dim == embed_dim else nn.Linear(self.query_dim, embed_dim)
        )

        self.image_kv_norm = nn.LayerNorm(self.query_dim)
        self.text_query_norm = nn.LayerNorm(self.query_dim)
        self.text_kv_norm = nn.LayerNorm(self.query_dim)
        self.output_norm = nn.LayerNorm(self.query_dim)

        self.image_cross_attn = nn.MultiheadAttention(
            embed_dim=self.query_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.text_cross_attn = nn.MultiheadAttention(
            embed_dim=self.query_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        val = math.sqrt(6.0 / float(3 * patch_size * patch_size + self.query_dim))
        self.queries = nn.Parameter(torch.zeros(num_layers, num_queries, self.query_dim))
        nn.init.uniform_(self.queries.data, -val, val)


class PromptInjector(nn.Module):
    def __init__(
        self,
        original_blocks: nn.Module,
        num_prompts: int = 10,
        embed_dim: int = 1280,
        num_skip: int = 1,
        insert_start_layer: int = 1,
        insert_end_layer: int = -1,
        patch_size: int = 16,
        project_dim: int = -1,
        dropout: float = 0.1,
        num_attn_queries: int = 0,
        attn_query_dim: int = -1,
        text_feature_dim: int = 1536,
        attn_num_heads: int = 8,
        attn_dropout: Optional[float] = None,
    ):
        super().__init__()
        self.original_blocks = original_blocks
        self.num_prompts = num_prompts
        self.num_layers = len(original_blocks)
        self.num_skip = num_skip
        self.embed_dim = embed_dim
        self.num_attn_queries = max(0, int(num_attn_queries))
        self.active_layers = self._build_active_layers(insert_start_layer, insert_end_layer)
        self.layer_to_prompt_idx = {
            layer_idx: prompt_idx for prompt_idx, layer_idx in enumerate(self.active_layers)
        }

        self.is_active = True
        self.grad_checkpointing = False
        self.supports_gradient_checkpointing = True
        self.runtime_text_classifier: Optional[torch.Tensor] = None

        self.tuner = DeepVPTTuner(
            len(self.active_layers),
            num_prompts,
            embed_dim,
            patch_size=patch_size,
            project_dim=project_dim,
            dropout=dropout,
        )
        self.attn_query_tuner = None
        if self.num_attn_queries > 0:
            self.attn_query_tuner = DeepAttnQueryTuner(
                len(self.active_layers),
                self.num_attn_queries,
                embed_dim,
                text_feature_dim=text_feature_dim,
                query_dim=attn_query_dim,
                num_heads=attn_num_heads,
                patch_size=patch_size,
                dropout=dropout if attn_dropout is None else float(attn_dropout),
            )

    def _build_active_layers(self, insert_start_layer: int, insert_end_layer: int) -> List[int]:
        # Layer indices are 1-based from the input side:
        # layer 1 is the first ViT block that sees image tokens.
        start_idx = 0 if insert_start_layer is None else int(insert_start_layer) - 1
        end_idx = (
            self.num_layers - 1
            if insert_end_layer is None or int(insert_end_layer) < 0
            else int(insert_end_layer) - 1
        )

        start_idx = max(0, start_idx)
        end_idx = min(self.num_layers - 1, end_idx)

        if start_idx > end_idx:
            raise ValueError(
                f"Invalid VPT layer range: start={insert_start_layer}, end={insert_end_layer}, num_layers={self.num_layers}"
            )

        return list(range(start_idx, end_idx + 1))

    def set_text_classifier(self, text_classifier: Optional[torch.Tensor]):
        self.runtime_text_classifier = text_classifier

    def clear_text_classifier(self):
        self.runtime_text_classifier = None

    def _expand_text_classifier(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if self.runtime_text_classifier is None:
            return None

        text_classifier = self.runtime_text_classifier
        if not torch.is_tensor(text_classifier) or text_classifier.numel() == 0:
            return None

        if text_classifier.ndim == 2:
            text_classifier = text_classifier.unsqueeze(0).expand(batch_size, -1, -1)
        elif text_classifier.ndim == 3:
            if text_classifier.shape[0] == 1 and batch_size > 1:
                text_classifier = text_classifier.expand(batch_size, -1, -1)
            elif text_classifier.shape[0] != batch_size:
                raise ValueError(
                    f"Text classifier batch mismatch: expected batch {batch_size}, got {text_classifier.shape[0]}"
                )
        else:
            raise ValueError(
                f"Unsupported text classifier shape {tuple(text_classifier.shape)}. Expected [C, D] or [B, C, D]."
            )

        return text_classifier.to(device=device, dtype=dtype)

    def _build_attn_queries(
        self,
        prompt_idx: int,
        patch_tokens: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if self.attn_query_tuner is None:
            return None

        batch_size = patch_tokens.shape[0]
        queries = self.attn_query_tuner.queries[prompt_idx]
        queries = queries.unsqueeze(0).expand(batch_size, -1, -1)
        queries = queries.to(device=patch_tokens.device, dtype=patch_tokens.dtype)
        queries = self.attn_query_tuner.query_dropout(queries)

        image_tokens = self.attn_query_tuner.image_proj(patch_tokens)
        image_tokens = self.attn_query_tuner.image_kv_norm(image_tokens)
        image_updates, _ = self.attn_query_tuner.image_cross_attn(
            query=queries,
            key=image_tokens,
            value=image_tokens,
            need_weights=False,
        )
        queries = queries + self.attn_query_tuner.query_dropout(image_updates)

        text_tokens = self._expand_text_classifier(
            batch_size=batch_size,
            device=patch_tokens.device,
            dtype=patch_tokens.dtype,
        )
        if text_tokens is not None:
            text_tokens = self.attn_query_tuner.text_proj(text_tokens)
            text_tokens = self.attn_query_tuner.text_kv_norm(text_tokens)
            text_updates, _ = self.attn_query_tuner.text_cross_attn(
                query=self.attn_query_tuner.text_query_norm(queries),
                key=text_tokens,
                value=text_tokens,
                need_weights=False,
            )
            queries = queries + self.attn_query_tuner.query_dropout(text_updates)

        queries = self.attn_query_tuner.output_proj(self.attn_query_tuner.output_norm(queries))
        return self.attn_query_tuner.query_dropout(queries)

    def _forward_block(self, x: torch.Tensor, block_idx: int):
        block = self.original_blocks[block_idx]
        prompt_idx = self.layer_to_prompt_idx.get(block_idx)

        if prompt_idx is None:
            return block(x)

        batch_size = x.shape[0]
        skip_tokens = x[:, :self.num_skip, :]
        patch_tokens = x[:, self.num_skip :, :]
        inserted_tokens = []

        if self.num_prompts > 0:
            raw_prompts = self.tuner.prompts[prompt_idx]
            projected_prompts = self.tuner.prompt_proj(raw_prompts)
            layer_prompts = projected_prompts.unsqueeze(0).expand(batch_size, -1, -1)
            layer_prompts = layer_prompts.to(device=x.device, dtype=x.dtype)
            layer_prompts = self.tuner.prompt_dropout(layer_prompts)
            inserted_tokens.append(layer_prompts)

        attn_queries = self._build_attn_queries(prompt_idx, patch_tokens)
        if attn_queries is not None:
            inserted_tokens.append(attn_queries)

        num_inserted_tokens = sum(token.shape[1] for token in inserted_tokens)
        if num_inserted_tokens == 0:
            return block(x)

        x_in = torch.cat([skip_tokens, *inserted_tokens, patch_tokens], dim=1)

        x_out = block(x_in)

        skip_tokens_out = x_out[:, :self.num_skip, :]
        patch_tokens_out = x_out[:, self.num_skip + num_inserted_tokens :, :]
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
