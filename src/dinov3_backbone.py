# dinov3_backbone.py
import torch
from transformers import AutoImageProcessor, AutoModel

def load_dinov3(model_id="facebook/dinov3-vitb16-pretrain-lvd1689m", dtype=torch.bfloat16, device="cuda"):
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()
    return processor, model
