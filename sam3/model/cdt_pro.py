import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Optional

from maft.modeling.transformer_decoder.position_encoding import PositionEmbeddingSine


class ShortCut_CrossAttention(nn.Module):
    def __init__(self, d_model, nhead, panoptic_on = False):
        super().__init__()
        self.norm = nn.LayerNorm(d_model) # Pre-Norm
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=0.0)
        
        self.null_key = nn.Parameter(torch.zeros(1, 1, d_model))
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        self.panoptic_on = panoptic_on
        
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        # Null key 初始化为一个极小的高斯分布，帮助网络起步寻找
        nn.init.normal_(self.null_key, std=0.02)
        
        # FFN 最后一层严格 0 初始化
        nn.init.zeros_(self.ffn[-1].weight)
        nn.init.zeros_(self.ffn[-1].bias)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        
        tgt_normed = self.norm(tgt)
        
        # memory 的默认形状是 [Seq_Len, Batch, Dim]
        L, B, D = memory.shape
        
        # ========================================================
        # 核心逻辑：构造 Null Token
        # ========================================================
        # 1. 构造全零的 Null Value (形状: [1, Batch, Dim])
        null_value = torch.zeros(1, B, D, device=memory.device, dtype=memory.dtype)
        # 将其拼接到原有的 memory 上 -> [L+1, Batch, Dim]
        aug_memory = torch.cat([memory, null_value], dim=0)
        
        # 2. 构造对应的 Null Key
        orig_key = self.with_pos_embed(memory, pos)
        # 将可学习的 null_key 扩展到 Batch 大小，并拼接到 key 上 -> [L+1, Batch, Dim]
        # 注意：Null Token 没有空间位置，所以不需要加 pos
        aug_key = torch.cat([orig_key, self.null_key.expand(1, B, D)], dim=0)
        
        # 3. 处理 key_padding_mask (通常在你的代码里是 None，但为了严谨处理一下)
        aug_memory_key_padding_mask = memory_key_padding_mask
        if memory_key_padding_mask is not None:
            # 原 mask 形状: [Batch, Seq_Len]
            # 我们需要增加一列 False (0)，表示 Null Token 永远对网络可见，不被 Mask
            null_mask = torch.zeros(B, 1, device=memory.device, dtype=torch.bool)
            aug_memory_key_padding_mask = torch.cat([memory_key_padding_mask, null_mask], dim=1)
        # ========================================================

        # 使用拼装好的 augmented 张量去做 Cross Attention
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt_normed, query_pos),
                                   key=aug_key,
                                   value=aug_memory,
                                   attn_mask=memory_mask,
                                   key_padding_mask=aug_memory_key_padding_mask)[0]

        # 残差相加
        tgt = tgt + self.ffn(tgt2)

        return tgt

class ContentDependentTransferPRO(nn.Module):

    def __init__(self, d_model, nhead, panoptic_on):
        super().__init__()
        self.pe_layer = PositionEmbeddingSine(d_model//2, normalize=True)
        self.cross_atten = ShortCut_CrossAttention(d_model = d_model, nhead = nhead, panoptic_on = panoptic_on)
    

    def forward(self, img_feat, text_classifier, ):
        text_classifier = text_classifier.unsqueeze(0).repeat(img_feat.shape[0],1,1)

        pos = self.pe_layer(img_feat, None).flatten(2).permute(2, 0, 1)  # hw * b * c
        img_feat = img_feat.flatten(2).permute(2, 0, 1)  # hw * b * c

        bias = self.cross_atten(text_classifier.permute(1, 0, 2), img_feat, memory_mask=None, memory_key_padding_mask=None, pos=pos, query_pos=None)

        return bias.permute(1, 0, 2) 
