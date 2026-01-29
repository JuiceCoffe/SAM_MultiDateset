import torch
import requests
from PIL import Image
from transformers import AutoProcessor, AutoModel

model = AutoModel.from_pretrained("google/siglip2-so400m-patch16-naflex", dtype=torch.float16, device_map="auto", attn_implementation="sdpa")
processor = AutoProcessor.from_pretrained("google/siglip2-so400m-patch16-naflex")
print(processor.image_processor)

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)
candidate_labels = ["a Pallas cat", "a lion", "a Siberian tiger"]
texts = [f'This is a photo of {label}.' for label in candidate_labels]

# default value for `max_num_patches` is 256, but you can increase resulted image resolution providing higher values e.g. `max_num_patches=512`
inputs = processor(text=texts, images=image, padding="max_length", max_num_patches=256, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model(**inputs)

logits_per_image = outputs.logits_per_image
probs = torch.sigmoid(logits_per_image)
print(f"{probs[0][0]:.1%} that image 0 is '{candidate_labels[0]}'")