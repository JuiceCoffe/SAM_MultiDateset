from torch import nn
from torch.nn import functional as F
from .convnext import ConvNextBlock
from sam3.model.model_misc import MLP

class AttnAdapter(nn.Module):

    def __init__(
        self,
        feat_dim: int,
        radio_dim: int,
        sam_dim: int,
    ):
        super().__init__()
        self.radio_dim = radio_dim
        self.sam_dim = sam_dim

        self.fusion_feat_proj = MLP(sam_dim, 1024, feat_dim, 3)
        self.radio_feat_proj =  MLP(radio_dim, 1024, feat_dim, 3)

        self.cnext1 = ConvNextBlock(feat_dim)
        self.cnext2 = ConvNextBlock(feat_dim)
        self.cnext3 = ConvNextBlock(feat_dim)
        self.norm = nn.LayerNorm(feat_dim)
        self.final = nn.Conv2d(feat_dim, 1, 1)

    def forward(self, radio_feature, sam_feature):
        
        bs, radio_feat_D, radio_feat_H, radio_feat_W = radio_feature.shape
        radio_feature = radio_feature.view(bs, self.radio_dim, -1).permute(0,2,1) # bs, l, d
        
        outputs = self.radio_feat_proj(radio_feature) + self.fusion_feat_proj(sam_feature.permute(1,0,2))
        
        outputs = outputs.view(bs, radio_feat_H, radio_feat_W, -1).permute(0, 3, 1, 2)
        outputs = self.cnext1(outputs)
        outputs = self.cnext2(outputs)
        outputs = self.cnext3(outputs)

        outputs = outputs.permute(0, 2, 3, 1)
        outputs = self.norm(outputs.contiguous())
        outputs = outputs.permute(0, 3, 1, 2) 
        
        outputs = self.final(outputs.contiguous()) 

        return outputs