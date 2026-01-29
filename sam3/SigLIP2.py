import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List

from detectron2.modeling import BACKBONE_REGISTRY, Backbone
from transformers import AutoModel, AutoProcessor

@BACKBONE_REGISTRY.register()
class SigLIP2Backbone(Backbone):
    def __init__(self):
        super().__init__()
 
        # 1. 加载模型
        model_name = "google/siglip2-so400m-patch16-naflex" 
        # model_name = "/path/to/your/local/weights" # 如果本地下载了权重
        print(f"Loading SigLIP 2 model from {model_name} ...")
        
        self.hf_model = AutoModel.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hf_model.to(self.device)
        self.hf_model.eval()
        
        # 冻结参数
        for param in self.hf_model.parameters():
            param.requires_grad = False

        self.patch_size = 16
        # 获取 Text Config 中的 hidden_size 作为输出维度
        # 在 SigLIP2 中，Vision backbone 的输出维度通常与 Text 维度对齐（或通过 head 对齐）
        # 对于 so400m，两者都是 1152
        self.embed_dim = self.hf_model.config.text_config.hidden_size
        
        # Detectron2 接口定义
        self._out_features = ["clip_vis_dense"]
        self._out_feature_strides = {"clip_vis_dense": self.patch_size}
        self._out_feature_channels = {"clip_vis_dense": self.embed_dim}
        
        # 预处理常数 (SigLIP2 使用 0.5 mean/std)
        self.register_buffer("pixel_mean", torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(self.device))
        self.register_buffer("pixel_std", torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(self.device))

    @property
    def dim_latent(self):
        return self.embed_dim

    def get_text_classifier(self, text_list: List[str], device=None):
        """
        获取文本特征。
        根据源码：text_model 输出的 pooler_output 已经经过了内部的 self.head 投影。
        """
        if device is None:
            device = self.device

        text_list = [t.lower() for t in text_list] # SigLIP2 偏好小写
        
        inputs = self.processor(text=text_list, padding="max_length", return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device)
        # attention_mask = inputs.attention_mask.to(self.device)

        with torch.no_grad():
            # 源码分析：Siglip2TextModel 的 forward 会调用 TextTransformer
            # TextTransformer 内部：pooled_output = self.head(pooled_output)
            # 所以这里的 pooler_output 已经是投影后的特征了。
            text_outputs = self.hf_model.text_model(
                input_ids=input_ids, 
                attention_mask=None
            )
            text_embeds = text_outputs.pooler_output
            print("Text Embeds Shape:", text_embeds.shape)  # [N, D]
            
            # 源码 Siglip2Model.forward 显示：
            # text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
            # text_embeds = F.normalize(text_embeds, p=2, dim=-1)

        return text_embeds.to(device)

    def forward(self, x):
        return self.extract_features(x)

    def extract_features(self, img: torch.Tensor):
        """
        Args:
            img: Tensor [B, 3, H, W], 数值 0-255 (Detectron2 默认)
                 例如 H=W=1008
        """

        print("Input Image Shape:", img.shape)  # [B, 3, H, W]
        B, C, H, W = img.shape
        SIG_P = 16 
        MAIN_P = 28

        new_H = int((H / MAIN_P) * SIG_P)
        new_W = int((W / MAIN_P) * SIG_P)

        if (H, W) != (new_H, new_W):
            img = F.interpolate(img, size=(new_H, new_W), mode="bicubic", align_corners=False)
        
        # 2. 图像预处理 (0-255 -> 0-1 -> Normalize)
        x = img / 255.0
        x = (x - self.pixel_mean) / self.pixel_std

        patches = x.unfold(2, SIG_P, SIG_P).unfold(3, SIG_P, SIG_P)
        # 调整轴顺序 -> [B, H//16, W//16, 3, 16, 16]
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        # 展平 -> [B, Num_Patches, 768]
        pixel_values = patches.view(B, -1, C * SIG_P * SIG_P)

        # 4. 构造 spatial_shapes (这里代表 H 和 W 方向上的 Patch 数量)
        num_patches_h = new_H // SIG_P
        num_patches_w = new_W // SIG_P
        spatial_shapes = torch.tensor([[num_patches_h, num_patches_w]], 
                                     device=self.device, dtype=torch.long).repeat(B, 1)

        # 5. 构造 pixel_attention_mask
        # 对应 SeqLen 维度的 mask，形状为 [B, Num_Patches]
        num_patches = num_patches_h * num_patches_w

        with torch.no_grad():
            vision_outputs = self.hf_model.vision_model(
                pixel_values=pixel_values,
                attention_mask=None,
                spatial_shapes=spatial_shapes 
            )
            
            # 这里的 last_hidden_state 是投影并归一化前的特征
            features_before_pool = vision_outputs.last_hidden_state

            head = self.hf_model.vision_model.head
            D = features_before_pool.shape[-1]  # 原始维度 768
            features = head(
                features_before_pool.view(-1, 1, D),  
                None,  
            ).view(B, -1, D)
            
            features = F.normalize(features, p=2, dim=-1)
            # features = vision_outputs.pooler_output.repeat(1, num_patches_h * num_patches_w, 1)

        features = features.view(B, num_patches_h, num_patches_w, self.embed_dim)

        features = features.permute(0, 3, 1, 2).contiguous()
        
        # 返回结果 {"clip_vis_dense": [B, 1152, 72, 72]}
        return {"clip_vis_dense": features}