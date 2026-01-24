import torch
import torch.nn as nn
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec
from detectron2.utils import comm


from core.vision_encoder import pe
from core.vision_encoder import transforms

@BACKBONE_REGISTRY.register()
class PEEncoder(Backbone):
    """
    A Detectron2-style Backbone that wraps the Perception Encoder (PE) model.
    It is designed to be a drop-in replacement for the previous CLIP backbone
    for the MAFT_Plus architecture.
    """
    def __init__(self, cfg):
        super().__init__()

        model_name = "PE-Core-L14-336"
        # model_name = "PE-Core-G14-448"

        if comm.get_local_rank() == 0:
            pe.CLIP.from_config(model_name, pretrained=True)
        comm.synchronize()

        self.pe_model = pe.CLIP.from_config(model_name, pretrained=True)

        self.tokenizer = transforms.get_text_tokenizer(self.pe_model.context_length)
        self.image_preprocess = transforms.get_image_transform(self.pe_model.image_size)
        
        # --- 2. 定义输出规格 (满足 MAFT_Plus 的需求) ---
        visual_backbone = self.pe_model.visual
        patch_size = visual_backbone.patch_size
        output_dim = visual_backbone.proj_dim

        # 只输出最终的密集特征图 'clip_vis_dense'
        self._out_features = ["clip_vis_dense"]
        self._out_feature_strides = {"clip_vis_dense": patch_size}
        self._out_feature_channels = {"clip_vis_dense": output_dim}
        
        self.eval()
        for param in self.pe_model.parameters():
            param.requires_grad = False

    # --- 3. 实现 MAFT_Plus 所需的属性和方法 ---

    @property
    def dim_latent(self):
        """ Requirement 1: 返回图文对齐空间的特征维度 """
        return self.pe_model.text_projection.shape[-1]

    @property
    def clip_model(self):
        return self.pe_model

    def tokenize_text(self, text_list):
        return self.tokenizer(text_list)


    def get_text_classifier(self, text_list, device):
        """ Requirement 2: 封装分词和编码的高级接口 """
        tokens = self.tokenize_text(text_list).to(device)
        text_features = self.pe_model.encode_text(tokens, normalize=True)
        return text_features

    def forward(self, x):
        """ 
        重写从密集特征图到真正用于匹配的语义特征图的过程，放在 core/vision_encoder/pe.py 的 class AttentionPooling
        def dense_forward(self, x: torch.Tensor):
            batch, num, _ = x.shape
            q = self.probe.repeat((batch * num, 1, 1)).to(x.dtype)
            x = x.view(batch * num, 1, -1)
            x = self.attn(q, x, x, need_weights=False)[0]
            x = x.view(batch, num, -1)
            # x = self.attn(x, x, x, need_weights=False)[0] # 这样修改输出整张图都是同一类别
            x = x + self.mlp(self.layernorm(x))

            return x
        """
        with torch.no_grad():
            visual_backbone = self.pe_model.visual
            
            dense_tokens = visual_backbone.forward_features(
                x,
                norm=True,            # LayerNorm
                strip_cls_token=True  # 只是在结尾去掉 CLS token，不影响中间计算（仍可有cls token）
            )
            
            attn_pool = self.pe_model.visual.attn_pool
            dense_tokens = attn_pool.dense_forward(dense_tokens)
            dense_tokens = dense_tokens @ visual_backbone.proj

            # 将 token 序列 (B, N, C) 重塑为特征图 (B, C, H_grid, W_grid)
            B, N, C = dense_tokens.shape
            patch_size = visual_backbone.patch_size
            grid_h = x.shape[2] // patch_size
            grid_w = x.shape[3] // patch_size
            
            # 确保维度匹配
            assert N == grid_h * grid_w, f"Token 数量({N})与图像/patch尺寸({grid_h}*{grid_w})不匹配"
            
            feature_map = dense_tokens.permute(0, 2, 1).reshape(B, C, grid_h, grid_w)

            return {"clip_vis_dense": feature_map}

    def extract_features(self, x):
        return self.forward(x)

    def output_shape(self):
        """ Detectron2 需要的辅助函数 """
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }