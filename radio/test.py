import os
import torch
# torch hub 默认下载路径
hub_dir = torch.hub.get_dir()
print(f"Torch Hub Cache Dir: {hub_dir}")
# RADIO 代码通常在 ~/.cache/torch/hub/NVlabs_RADIO_master 目录下