import torch
import torch.nn.functional as F


def vit_tokens_to_grid(outputs, H, W, patch=16, num_registers=4):
    """
    DINOv3 ViT returns last_hidden_state with [CLS] + 4 registers + N patches.
    """
    tok = outputs.last_hidden_state  # [B, 1+4+N, D]
    B, T, D = tok.shape
    nH, nW = H // patch, W // patch
    patch_tokens = tok[:, 1 + num_registers :, :]  # [B, N, D]
    grid = patch_tokens.view(B, nH, nW, D).contiguous()  # [B, nH, nW, D]
    return grid


def compute_query_prototype(chips_emb_list):
    # normalize and average for cosine-stable prototype
    embs = [F.normalize(e, dim=-1) for e in chips_emb_list]
    proto = torch.stack(embs, dim=0).mean(dim=0)
    proto = F.normalize(proto, dim=-1)
    return proto  # [D]


def cosine_heatmap(grid, proto):
    # grid: [1, nH, nW, D], proto: [D]
    g = F.normalize(grid, dim=-1)
    p = F.normalize(proto, dim=-1)
    heat = torch.einsum("bhwd,d->bhw", g, p)  # inner product == cosine
    return heat  # [1, nH, nW]
