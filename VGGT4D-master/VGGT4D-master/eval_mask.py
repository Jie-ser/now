
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

from vggt4d.utils.eval_mask_utils import (eval_boundary, eval_iou,
                                          eval_statistics)

seq_list_2016 = [
    'blackswan',
    'bmx-trees',
    'breakdance',
    'camel',
    'car-roundabout',
    'car-shadow',
    'cows',
    'dance-twirl',
    'dog',
    'drift-chicane',
    'drift-straight',
    'goat',
    'horsejump-high',
    'kite-surf',
    'libby',
    'motocross-jump',
    'paragliding-launch',
    'parkour',
    'scooter-black',
    'soapbox'
]

seq_list_2017 = [
    'bike-packing',
    'blackswan',
    'bmx-trees',
    'breakdance',
    'camel',
    'car-roundabout',
    'car-shadow',
    'cows',
    'dance-twirl',
    'dog',
    'dogs-jump',
    'drift-chicane',
    'drift-straight',
    'goat',
    'gold-fish',
    'horsejump-high',
    'india',
    'judo',
    'kite-surf',
    'lab-coat',
    'libby',
    'loading',
    'mbike-trick',
    'motocross-jump',
    'paragliding-launch',
    'parkour',
    'pigs',
    'scooter-black',
    'shooting',
    'soapbox'
]

seq_list = seq_list_2016


def load_result_dyn_mask(res_dyn_mask_paths: List[Path]) -> np.ndarray:
    dyn_masks = []
    for res_dyn_mask_path in res_dyn_mask_paths:
        dyn_mask = np.array(Image.open(res_dyn_mask_path))
        dyn_masks.append(dyn_mask)
    return np.array(dyn_masks) > 0


def vggt_crop_img(img):
    target_size = 518
    width, height = img.size

    new_width = target_size
    # Calculate height maintaining aspect ratio, divisible by 14
    new_height = round(height * (new_width / width) / 14) * 14
    img = img.resize((new_width, new_height), Image.NEAREST)

    if new_height > target_size:
        start_y = (new_height - target_size) // 2
        img = img.crop((0, start_y, target_size, start_y + target_size))

    return img


def _resize_pil_image(img, long_edge_size, nearest=False):
    S = max(img.size)
    if S > long_edge_size:
        interp = Image.LANCZOS if not nearest else Image.NEAREST
    elif S <= long_edge_size:
        interp = Image.BICUBIC
    new_size = tuple(int(round(x*long_edge_size/S)) for x in img.size)
    return img.resize(new_size, interp)


def crop_img(img, size, square_ok=False, nearest=True, crop=True):
    W1, H1 = img.size
    if size == 224:
        # resize short side to 224 (then crop)
        img = _resize_pil_image(img, round(
            size * max(W1/H1, H1/W1)), nearest=nearest)
    else:
        # resize long side to 512
        img = _resize_pil_image(img, size, nearest=nearest)
    W, H = img.size
    cx, cy = W//2, H//2
    if size == 224:
        half = min(cx, cy)
        img = img.crop((cx-half, cy-half, cx+half, cy+half))
    else:
        halfw, halfh = ((2*cx)//16)*8, ((2*cy)//16)*8
        if not (square_ok) and W == H:
            halfh = 3*halfw/4
        if crop:
            img = img.crop((cx-halfw, cy-halfh, cx+halfw, cy+halfh))
        else:  # resize
            img = img.resize((2*halfw, 2*halfh), Image.NEAREST)
    return img


def load_gt_dyn_mask(gt_dyn_mask_paths: List[Path]) -> np.ndarray:
    dyn_masks = []
    for gt_dyn_mask_path in gt_dyn_mask_paths:
        dyn_mask = np.array(vggt_crop_img(Image.open(gt_dyn_mask_path)))
        dyn_masks.append(dyn_mask)
    return np.array(dyn_masks) > 0


def _extract_frame_idx(path: Path) -> int:
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    return int(digits) if digits else 0


def _collect_mask_paths(base_dir: Path, seq_name: str, pattern: str) -> List[Path]:
    seq_dir = base_dir / seq_name
    if not seq_dir.exists():
        return []
    return sorted(seq_dir.glob(pattern), key=_extract_frame_idx)


def _resolve_sequences(pred_root: Path, gt_root: Path, seq_candidates: Optional[List[str]]) -> List[str]:
    if seq_candidates is None:
        pred_seqs = {p.name for p in pred_root.iterdir() if p.is_dir()}
        gt_seqs = {p.name for p in gt_root.iterdir() if p.is_dir()}
        return sorted(pred_seqs & gt_seqs)

    return [
        name for name in seq_candidates
        if (pred_root / name).exists() and (gt_root / name).exists()
    ]


if __name__ == "__main__":
    pred_root = Path("outputs/ours")
    gt_root = Path("datasets/DAVIS/Annotations_unsupervised/480p")

    if not pred_root.exists():
        raise FileNotFoundError(f"Prediction directory does not exist: {pred_root}")
    if not gt_root.exists():
        raise FileNotFoundError(f"GT directory does not exist: {gt_root}")

    seq_names = _resolve_sequences(pred_root, gt_root, seq_list)
    if not seq_names:
        raise RuntimeError("No sequences found for evaluation. Please check if the prediction and GT paths are correct.")

    j_means = []
    j_recalls = []
    j_decays = []
    f_means = []
    f_recalls = []
    f_decays = []
    t_means = []
    t_recalls = []
    t_decays = []
    for seq_name in seq_names:
        res_dyn_mask_paths = _collect_mask_paths(
            pred_root, seq_name, "dynamic_mask_*.png")
        gt_dyn_mask_paths = _collect_mask_paths(gt_root, seq_name, "*.png")

        if not res_dyn_mask_paths:
            print(f"Skipping {seq_name}: prediction mask not found.")
            continue
        if not gt_dyn_mask_paths:
            print(f"Skipping {seq_name}: GT mask not found.")
            continue

        res_dyn_masks = load_result_dyn_mask(res_dyn_mask_paths)
        gt_dyn_masks = load_gt_dyn_mask(gt_dyn_mask_paths)

        n_res = res_dyn_masks.shape[0]
        n_gt = gt_dyn_masks.shape[0]
        if n_res < n_gt:
            res_dyn_masks = res_dyn_masks[:n_gt]
        elif n_res > n_gt:
            gt_dyn_masks = gt_dyn_masks[:n_res]

        iou = eval_iou(gt_dyn_masks, res_dyn_masks)
        boundary = eval_boundary(gt_dyn_masks, res_dyn_masks)
        M_iou, R_iou, D_iou = eval_statistics(iou)
        M_boundary, R_boundary, D_boundary = eval_statistics(boundary)

        j_means.append(M_iou)
        j_recalls.append(R_iou)
        j_decays.append(D_iou)
        f_means.append(M_boundary)
        f_recalls.append(R_boundary)
        f_decays.append(D_boundary)

        print(f"Sequence {seq_name}")
        print(
            f"MAX JM: {iou.max()} {iou.argmax()}, MAX FM: {boundary.max()} {boundary.argmax()}")
        print(f"JM: {M_iou}, JR: {R_iou}, JD: {D_iou}")
        print(f"FM: {M_boundary}, FR: {R_boundary}, FD: {D_boundary}\n")
    print(
        f"Average JM: {np.mean(j_means)}, Average JR: {np.mean(j_recalls)}, Average JD: {np.mean(j_decays)}")
    print(
        f"Average FM: {np.mean(f_means)}, Average FR: {np.mean(f_recalls)}, Average FD: {np.mean(f_decays)}")
