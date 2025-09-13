import numpy as np
from skimage.measure import label, regionprops
import torch
from torchvision.ops import nms


def heatmap_to_boxes(heat, thr=0.85, scale=16):
    """
    heat: [nH, nW] similarity map
    """
    mask = (heat >= thr).astype(np.uint8)
    lab = label(mask, connectivity=1)
    props = regionprops(lab)
    boxes = []
    scores = []
    for p in props:
        minr, minc, maxr, maxc = p.bbox
        # convert patch coords to pixel coords with patch size (=scale)
        x1, y1 = minc * scale, minr * scale
        x2, y2 = maxc * scale, maxr * scale
        boxes.append([x1, y1, x2, y2])
        # peak score within region
        region_scores = heat[minr:maxr, minc:maxc]
        scores.append(float(region_scores.max()))
    return np.array(boxes, dtype=np.float32), np.array(scores, dtype=np.float32)


def nms_boxes(boxes_np, scores_np, iou_thr=0.5):
    if len(boxes_np) == 0:
        return boxes_np, scores_np
    boxes = torch.from_numpy(boxes_np)
    scores = torch.from_numpy(scores_np)
    keep = nms(boxes, scores, iou_thr)
    return boxes[keep].numpy(), scores[keep].numpy()
