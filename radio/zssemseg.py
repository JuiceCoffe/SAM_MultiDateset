from PIL import Image

import torch
from torch.nn import functional as F
from torchvision.transforms.functional import pil_to_tensor
model_version="c-radio_v4-h" # for C-RADIOv3-H model (ViT-H/16)

# # 相对路径指向你的源码文件夹
# local_path = './NVlabs_RADIO_main'

# # 使用 source='local'
# model = torch.hub.load(
#     local_path,          # 本地文件夹路径
#     'radio_model',       # hubconf.py 里的函数名
#     version="c-radio_v4-h", 
#     adaptor_names=['siglip2-g'],
#     source='local'       # 关键：告诉 torch 在本地找
# )

model = torch.hub.load('NVlabs/RADIO', 'radio_model', version=model_version, progress=True, skip_validation=True, adaptor_names=['siglip2-g'])
model.cuda().eval()

x = Image.open('/workspace/hmp/datasets/ade/ADEChallengeData2016/images/validation/ADE_val_00000001.jpg').convert('RGB')
x = pil_to_tensor(x).to(dtype=torch.float32, device='cuda')
x.div_(255.0)  # RADIO expects the input values to be between 0 and 1
x = x.unsqueeze(0) # Add a batch dimension

nearest_res = model.get_nearest_supported_resolution(*x.shape[-2:])
x = F.interpolate(x, nearest_res, mode='bilinear', align_corners=False)


#### Example 5 ####
# Teacher adaptors, e.g. for text alignment
###################

vis_output = model(x)
# These are the usual RADIO features
backbone_summary, backbone_features = vis_output['backbone']
# There will also be summary and feature pairs for each of the loaded adaptors
sig2_vis_summary, sig2_vis_features = vis_output['siglip2-g']

# The 'siglip2-g' and 'clip' adaptors (when available) are special because they also support text tokenization and encoding
sig2_adaptor = model.adaptors['siglip2-g']
text_input = sig2_adaptor.tokenizer(['Windows']).to('cuda')
text_tokens = sig2_adaptor.encode_text(text_input, normalize=True)

sig2_vis_features =  sig2_adaptor.head_mlp(backbone_features.to(next(sig2_adaptor.parameters()).dtype)).to(dtype=backbone_features.dtype)

sim = F.cosine_similarity(sig2_vis_summary, text_tokens)
print(sim)


import torch
import torch.nn.functional as F
import matplotlib
# 必须在导入 pyplot 之前设置 backend，防止在无 GUI 环境报错
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np
import os

# --- 接续你的代码环境 ---
# 假设变量 x, sig2_vis_features, text_tokens, model 均已存在
# x: [1, 3, H, W]
# sig2_vis_features: [1, num_tokens, 1536]
# text_tokens: [1, 1536] (已经 normalized)

def save_segmentation_vis(image_tensor, features, text_emb, save_path="/root/hmp/mysamMultiDateset/radio/radio_segmentation_result.png"):
    print("正在处理可视化...")
    
    # 1. 归一化特征
    # [1, N, 1536]
    feats_norm = F.normalize(features, dim=-1)
    # [1, 1536] (确保文本也归一化)
    text_norm = F.normalize(text_emb, dim=-1)
    
    # 2. 计算余弦相似度 (Dot Product)
    # [1, N, 1536] @ [1, 1536, 1] -> [1, N, 1] -> [1, N]
    similarity = torch.matmul(feats_norm, text_norm.unsqueeze(-1)).squeeze(-1)
    
    # 3. 推导 Grid 尺寸 (Reshape)
    # RADIO 的 patch size 通常是 14 或 16，根据输入分辨率反推
    n_tokens = features.shape[1]
    H_img, W_img = image_tensor.shape[-2:]
    
    # 计算宽高比
    aspect_ratio = W_img / H_img
    # W_grid * H_grid = n_tokens
    # W_grid / H_grid = aspect_ratio  => W_grid^2 = n_tokens * aspect_ratio
    w_grid = int(np.round(np.sqrt(n_tokens * aspect_ratio)))
    h_grid = int(np.round(n_tokens / w_grid))
    
    print(f"输入分辨率: {H_img}x{W_img}, 特征点数: {n_tokens}")
    print(f"推导出的特征网格: {h_grid}x{w_grid}")
    
    if h_grid * w_grid != n_tokens:
        # 如果简单的宽高比推导对不上，通常是因为 padding 或者 patch size 对齐问题
        # 这里做一个简单的容错，或者你需要手动指定 model.patch_size
        print("警告: 自动推导的网格尺寸与Token数量不完全匹配，尝试直接按 patch_size=16 或 14 强行计算...")
        # 尝试 patch 16
        if (H_img // 16) * (W_img // 16) == n_tokens:
            h_grid, w_grid = H_img // 16, W_img // 16
        # 尝试 patch 14
        elif (H_img // 14) * (W_img // 14) == n_tokens:
            h_grid, w_grid = H_img // 14, W_img // 14
            
    # 重塑为 [1, 1, H_grid, W_grid]
    sim_map = similarity.view(1, 1, h_grid, w_grid)
    
    # 4. 上采样到原图尺寸
    sim_map_up = F.interpolate(sim_map, size=(H_img, W_img), mode='bilinear', align_corners=False)
    
    # 5. 绘图并保存
    # 转 CPU numpy
    img_np = image_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    # 简单的反归一化用于显示 (0-1之间)
    img_np = np.clip(img_np, 0, 1)
    
    seg_np = sim_map_up.squeeze().detach().cpu().numpy()
    
    # 创建画布
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # 子图1: 原图
    axs[0].imshow(img_np)
    axs[0].set_title("Input Image")
    axs[0].axis('off')
    
    # 子图2: 相似度热力图
    im = axs[1].imshow(seg_np, cmap='jet')
    axs[1].set_title("Text Similarity Map")
    axs[1].axis('off')
    plt.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)
    
    # 子图3: 叠加图
    axs[2].imshow(img_np)
    # 设定一个透明度阈值，只高亮显示相似度较高的区域，或者直接半透明叠加
    axs[2].imshow(seg_np, cmap='jet', alpha=0.5) 
    axs[2].set_title("Overlay")
    axs[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"结果已保存至: {os.path.abspath(save_path)}")

# --- 执行函数 ---
save_segmentation_vis(x, sig2_vis_features, text_tokens)