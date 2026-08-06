import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from einops import rearrange

from vggt4d.masks.refine_dyn_mask import RefineDynMask
from vggt4d.models.vggt4d import VGGTFor4D
from vggt4d.masks.dynamic_mask import (adaptive_multiotsu_variance,
                                             cluster_attention_maps,
                                             extract_dyn_map)
from vggt4d.utils.model_utils import inference, organize_qk_dict
from vggt4d.utils.store import (save_depth, save_depth_conf,
                                save_dynamic_masks, save_intrinsic_txt,
                                save_rgb, save_tum_poses)
from vggt.utils.load_fn import load_and_preprocess_images

device = torch.device("cuda") \
    if torch.cuda.is_available() \
    else torch.device("cpu")

model = VGGTFor4D()
model.load_state_dict(torch.load(
    "./ckpts/model_tracker_fixed_e20.pt", weights_only=True))
model.eval()
model = model.to(device)


def process_scene(scene_dir: Path, output_dir: Path):
    """
    Process a single scene

    Args:
        scene_dir: Scene input directory path
        output_dir: Scene output directory path
    """
    image_paths = list(scene_dir.glob("*.jpg")) + list(scene_dir.glob("*.png"))
    image_paths = sorted(image_paths)

    if len(image_paths) == 0:
        print(f"Warning: No images found in {scene_dir}, skipping this scene")
        return

    print(f"Processing scene: {scene_dir.name} ({len(image_paths)} images)")

    images = load_and_preprocess_images(
        [str(image_path) for image_path in image_paths]).to(device)
    n_img, _, h_img, w_img = images.shape

    output_dir.mkdir(parents=True, exist_ok=True)

    # stage 1 predict depth map and dynamic map
    print("  Stage 1: predict depth map and dynamic map")
    predictions1, qk_dict, enc_feat, agg_tokens_list = inference(
        model, images)
    del agg_tokens_list
    qk_dict = organize_qk_dict(qk_dict, images.shape[0])

    dyn_maps = extract_dyn_map(qk_dict, images)
    # save memory usage
    # dyn_maps = batch_extract_dyn_map(qk_dict, images)

    n_img, _, h_img, w_img = images.shape

    h_tok, w_tok = h_img // 14, w_img // 14

    feat_map = rearrange(
        enc_feat, "n_img (h w) c -> n_img h w c", h=h_tok, w=w_tok)

    norm_dyn_map, _ = cluster_attention_maps(
        feat_map, dyn_maps)

    upsampled_map = F.interpolate(rearrange(
        norm_dyn_map, "n_img h w -> n_img 1 h w"), size=(h_img, w_img), mode='bilinear', align_corners=False)
    upsampled_map = rearrange(
        upsampled_map, "n_img 1 h w -> n_img h w")

    thres = adaptive_multiotsu_variance(upsampled_map.cpu().numpy())
    dyn_masks = upsampled_map > thres

    # stage 2 refine extrinsics by dynamic map
    print("  Stage 2: refine extrinsics by dynamic map")
    if "enc_feat" in locals():
        del enc_feat
    if "feat_map" in locals():
        del feat_map

    torch.cuda.empty_cache()
    predictions2, _, _, _ = inference(model, images, dyn_masks.to(device))

    pred_intrinsic = predictions1["intrinsic"]
    pred_cam2world2 = predictions2["cam2world"]

    pred_depths = predictions1["depth"]
    pred_conf = predictions1["depth_conf"]

    # save predictions
    final_prediction = {**predictions1}
    final_prediction["extrinsic"] = predictions2["extrinsic"]
    final_prediction["cam2world"] = pred_cam2world2

    # stage 3 refine dynamic map
    print("  Stage 3: refine dynamic map")
    if "feat_map" in locals():
        del feat_map
    torch.cuda.empty_cache()

    pred_intrinsic = final_prediction["intrinsic"]
    pred_cam2world = final_prediction["cam2world"]

    pred_depths = final_prediction["depth"]
    pred_conf = final_prediction["depth_conf"]

    refiner = RefineDynMask(images, torch.tensor(pred_depths).to(device),
                            dyn_masks.to(device),
                            torch.tensor(
                                pred_cam2world).float().to(device),
                            torch.tensor(pred_intrinsic).to(device),
                            device)

    refined_mask = refiner.refine_masks()
    del refiner

    print(f"  Saving predictions to {output_dir}\n")
    save_intrinsic_txt(output_dir, pred_intrinsic)
    save_rgb(output_dir, images)
    save_depth(output_dir, pred_depths)
    save_depth_conf(output_dir, pred_conf)
    save_tum_poses(output_dir, pred_cam2world2)
    save_dynamic_masks(output_dir, refined_mask)


def main(input_dir: str, output_dir: str):
    """
    Main function

    Args:
        input_dir: Input data directory path
        output_dir: Output result directory path
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    scene_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
    scene_dirs = sorted(scene_dirs)

    if len(scene_dirs) == 0:
        raise ValueError(f"No scene directories found in {input_dir}")

    print(f"Found {len(scene_dirs)} scenes, starting processing...\n")

    for scene_dir in scene_dirs:
        scene_name = scene_dir.name
        scene_output_dir = output_dir / scene_name
        process_scene(scene_dir, scene_output_dir)

    print(f"All scenes processed! Results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VGGT4D demo script")
    parser.add_argument("--input_dir", type=str, default=None, help="Input data directory path")
    parser.add_argument("--output_dir", type=str,
                        default=None, help="Output result directory path")
    args = parser.parse_args()
    main(input_dir=args.input_dir, output_dir=args.output_dir)
