import rasterio
import numpy as np
import torch


def read_tiff_4ch(path, to_dtype=torch.float32):
    with rasterio.open(path) as src:
        arr = src.read()  # shape: (bands, H, W)
    # Expect 4 bands: [B,G,R,NIR] or similar; reorder to [R,G,B,NIR] if needed
    # Here we assume (B,G,R,NIR) -> (R,G,B,NIR)
    if arr.shape == 4:
        b, g, r, nir = arr
        arr = np.stack([r, g, b, nir], axis=0)
    else:
        raise ValueError("Expected 4-band imagery")
    arr = arr.astype(np.float32)
    # simple normalization to [0,1] if dynamic range known; advanced normalization can be added
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
    tensor = torch.from_numpy(arr)  # CHW
    return tensor
