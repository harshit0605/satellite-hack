import torch
import torch.nn as nn


def adapt_vit_patch_embed_to_4ch(model):
    # Works with HF DINOv3 ViT: find the patch projection conv
    # typical path: model.vit.embeddings.patch_embeddings.projection
    pe = model.vit.embeddings.patch_embeddings
    old = (
        pe.projection
    )  # Conv2d(in_channels=3, out_channels=embed_dim, kernel_size=16, stride=16)

    new_conv = nn.Conv2d(
        in_channels=4,
        out_channels=old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        bias=(old.bias is not None),
    )

    with torch.no_grad():
        w = old.weight  # [out_c, 3, k, k]
        w_mean = w.mean(dim=1, keepdim=True)  # [out_c, 1, k, k]
        w4 = w_mean.repeat(1, 4, 1, 1)  # [out_c, 4, k, k]
        new_conv.weight.copy_(w4)
        if old.bias is not None:
            new_conv.bias.copy_(old.bias)

    pe.projection = new_conv
    return model
