import torch
from retrieval import vit_tokens_to_grid


@torch.inference_mode()
def extract_dense_grid(model, pixel_inputs, H, W, dtype=torch.bfloat16):
    with torch.autocast("cuda", dtype=dtype):
        outputs = model(**pixel_inputs)
    grid = vit_tokens_to_grid(outputs, H, W, patch=16, num_registers=4)
    return grid
