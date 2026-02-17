import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.transforms.functional import pil_to_tensor


class RADIOwrapper(nn.Module):
    def __init__(self): 
        super().__init__()

        self.model = torch.hub.load('NVlabs/RADIO', 
            'radio_model', 
            version="c-radio_v4-h", 
            progress=True, 
            skip_validation=True, 
            adaptor_names=['siglip2-g']
        )

        self.sig2_adaptor = self.model.adaptors['siglip2-g']


    def forward(self, x):
        return x

    def get_text_classifier(self, text_list, device):
        text_input = self.sig2_adaptor.tokenizer(text_list).to(device)
        text_tokens = self.sig2_adaptor.encode_text(text_input, normalize=True)
        return text_tokens