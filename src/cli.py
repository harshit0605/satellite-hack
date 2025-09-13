import argparse, os, torch, numpy as np
from io_raster import read_tiff_4ch
from dinov3_backbone import load_dinov3
from four_channel_patch_embed import adapt_vit_patch_embed_to_4ch
from retrieval import compute_query_prototype, vit_tokens_to_grid, cosine_heatmap
from postprocess import heatmap_to_boxes, nms_boxes
from multiscale import resize_chw
from transformers import AutoImageProcessor


def to_model_inputs(processor, img_3ch_chw_or_4ch, device):
    # processor expects PIL or numpy HWC 3-channel; for 4-channel we bypass image processor normalization,
    # use manual normalization and feed as pixel_values directly
    if img_3ch_chw_or_4ch.shape == 3:
        hwc = img_3ch_chw_or_4ch.permute(1, 2, 0).cpu().numpy()
        inputs = processor(images=hwc, return_tensors="pt")
        return {k: v.to(device) for k, v in inputs.items()}
    else:
        # 4-channel path: assume model patch_embed was adapted; just pass pixel_values
        x = img_3ch_chw_or_4ch.unsqueeze(0).to(device)
        return {"pixel_values": x}


def embed_chip(model, processor, chip_chw, device):
    H, W = chip_chw.shape[1:]
    inputs = to_model_inputs(processor, chip_chw, device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(**inputs).last_hidden_state
    # use CLS token as global or mean of patches; here mean of patch tokens
    num_registers = 4
    patches = out[:, 1 + num_registers :, :]
    emb = patches.mean(dim=1).squeeze(0)  # [D]
    return emb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chips", nargs="+", required=True)
    ap.add_argument("--targets_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--object_name", required=True)
    ap.add_argument("--model_id", default="facebook/dinov3-vitb16-pretrain-lvd1689m")
    ap.add_argument("--satellite_mode", action="store_true")
    ap.add_argument("--thr", type=float, default=0.85)
    ap.add_argument("--nms", type=float, default=0.5)
    ap.add_argument("--scales", nargs="+", type=float, default=[0.75, 1.0, 1.25])
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor, model = load_dinov3(args.model_id, dtype=torch.bfloat16, device=device)

    if (
        args.satellite_mode
        and model.vit.embeddings.patch_embeddings.projection.in_channels == 3
    ):
        model = adapt_vit_patch_embed_to_4ch(model)

    # Build prototype
    chip_embs = []
    for cp in args.chips:
        chip = read_tiff_4ch(cp)  # CHW normalized
        if (
            chip.shape == 4
            and model.vit.embeddings.patch_embeddings.projection.in_channels == 3
        ):
            # fallback: drop NIR quickly
            chip = chip[:3, ...]
        emb = embed_chip(model, processor, chip.to(device), device)
        chip_embs.append(emb)
    proto = compute_query_prototype(chip_embs)  # [D]

    results = []
    for fn in sorted(os.listdir(args.targets_dir)):
        if not fn.lower().endswith((".tif", ".tiff", ".jp2")):
            continue
        path = os.path.join(args.targets_dir, fn)
        img = read_tiff_4ch(path)
        if (
            img.shape == 4
            and model.vit.embeddings.patch_embeddings.projection.in_channels == 3
        ):
            img = img[:3, ...]  # fallback
        all_boxes, all_scores = [], []
        for s in args.scales:
            im_s, orig_hw, s_hw = resize_chw(img, s)
            Hs, Ws = s_hw
            inputs = to_model_inputs(processor, im_s.to(device), device)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(**inputs)
            grid = vit_tokens_to_grid(
                outputs, Hs, Ws, patch=16, num_registers=4
            )  # [1,h,w,D]
            heat = cosine_heatmap(grid, proto).squeeze(0).detach().cpu().numpy()
            boxes, scores = heatmap_to_boxes(heat, thr=args.thr, scale=16)
            # rescale boxes back to original pixels
            if len(boxes):
                scale_back_x = orig_hw[1] / s_hw[1]
                scale_back_y = orig_hw / s_hw
                boxes[:, [0, 2]] *= scale_back_x
                boxes[:, [1, 3]] *= scale_back_y
                all_boxes.append(boxes)
                all_scores.append(scores)
        if len(all_boxes):
            boxes_np = np.concatenate(all_boxes, axis=0)
            scores_np = np.concatenate(all_scores, axis=0)
            boxes_np, scores_np = nms_boxes(boxes_np, scores_np, iou_thr=args.nms)
            for b, sc in zip(boxes_np, scores_np):
                x1, y1, x2, y2 = map(int, b.tolist())
                results.append([args.object_name, x1, y1, x2, y2, fn, float(sc)])

    with open(args.out, "w") as f:
        for rec in results:
            f.write("{} {} {} {} {} {} {:.6f}\n".format(*rec))
    print(f"Wrote {len(results)} detections to {args.out}")


if __name__ == "__main__":
    main()
