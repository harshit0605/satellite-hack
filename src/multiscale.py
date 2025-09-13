import torch
import torch.nn.functional as F


def resize_chw(img_chw, scale):
    C, H, W = img_chw.shape
    newH, newW = int(H * scale), int(W * scale)
    img = F.interpolate(
        img_chw.unsqueeze(0), size=(newH, newW), mode="bilinear", align_corners=False
    )
    return img.squeeze(0), (H, W), (newH, newW)
