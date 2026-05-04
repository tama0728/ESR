"""Generate ensemble super-resolution images using GFPGAN-based adaptive facial restoration.

This script combines structure-oriented (MSE) and perception-oriented (GAN) SR outputs
using edge-guided blending and GFPGAN face restoration with histogram matching.

Dependencies:
    pip install gfpgan mediapipe opencv-python scikit-image

Usage:
    # Generate 'final' version (edge blending + GFPGAN face restoration + histogram matching)
    python generate_ensemble_gfpgan.py \
        --version final \
        --gt_dir data/SR/Set5/HR/ \
        --struct_dir data/swinir_classical_sr_x4_Set5/ \
        --struct_suffix _SwinIR \
        --prc_dir data/swinir_real_sr_x4_Set5/ \
        --prc_suffix x4_SwinIR \
        --output_dir results/GFPGAN/SwinIR/Set5/ \
        --gfpgan_model pretrained_models/GFPGANv1.4.pth

    # Generate 'edge' version (edge blending only, no face restoration)
    python generate_ensemble_gfpgan.py \
        --version edge \
        --gt_dir data/SR/Set5/HR/ \
        --struct_dir data/swinir_classical_sr_x4_Set5/ \
        --struct_suffix _SwinIR \
        --prc_dir data/swinir_real_sr_x4_Set5/ \
        --prc_suffix x4_SwinIR \
        --output_dir results/GFPGAN/SwinIR/Set5/
"""

import argparse
import os
import glob

import cv2
import numpy as np
from gfpgan import GFPGANer





# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def create_folder(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def calculate_sobel(gray: np.ndarray, resize: bool = False) -> np.ndarray:
    """Compute normalised Sobel gradient magnitude."""
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobelx ** 2 + sobely ** 2)
    sobel = sobel / np.max(sobel)
    if resize:
        sobel = cv2.resize(sobel, (0, 0), fx=4, fy=4, interpolation=cv2.INTER_LINEAR)
    return sobel


def merge_images(
    source: np.ndarray,
    ref: np.ndarray,
    mask: np.ndarray,
    reverse: bool = False,
) -> np.ndarray:
    """Alpha-blend *source* and *ref* using *mask*."""
    source = source.astype(np.float32)
    ref = ref.astype(np.float32)
    mask = mask.astype(np.float32)
    max_val = np.max(mask)
    if max_val > 0:
        mask /= max_val
    mask = np.expand_dims(mask, axis=2)
    if reverse:
        img = source * mask + ref * (1 - mask)
    else:
        img = source * (1 - mask) + ref * mask
    return img.astype(np.uint8)


def histogram_matching(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Reinhard colour transfer (L channel only)."""
    src_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float64)
    ref_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float64)

    src_l = src_lab[:, :, 0]
    ref_l = ref_lab[:, :, 0]

    src_mean, src_std = src_l.mean(), src_l.std()
    ref_mean, ref_std = ref_l.mean(), ref_l.std()

    if src_std > 0:
        src_lab[:, :, 0] = (src_l - src_mean) * (ref_std / src_std) + ref_mean

    src_lab[:, :, 0] = np.clip(src_lab[:, :, 0], 0, 255)
    return cv2.cvtColor(src_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------------------
# Main generation pipeline
# ---------------------------------------------------------------------------
def generate(args) -> None:
    """Run the GFPGAN ensemble image generation pipeline."""

    struct_suffix = args.struct_suffix  # e.g. "_SwinIR"
    prc_suffix = args.prc_suffix        # e.g. "x4_SwinIR"
    version = args.version              # "edge" or "final"

    # Initialise GFPGAN restorer (only needed for 'final' version)
    restorer = None
    if version == "final":
        restorer = GFPGANer(
            model_path=args.gfpgan_model,
            upscale=1,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,
        )

    create_folder(args.output_dir)
    face_cmp_dir = os.path.join(args.output_dir, "face_compare")
    create_folder(face_cmp_dir)

    gt_paths = sorted(glob.glob(os.path.join(args.gt_dir, "*")))
    print(f"Found {len(gt_paths)} GT images in {args.gt_dir}")

    for i, path in enumerate(gt_paths):
        imgname, imgext = os.path.splitext(os.path.basename(path))
        print(f"[{i + 1}/{len(gt_paths)}] {imgname}{imgext}")

        img_mse = cv2.imread(
            os.path.join(args.struct_dir, imgname + struct_suffix + imgext),
            cv2.IMREAD_COLOR_BGR,
        )
        img_gan = cv2.imread(
            os.path.join(args.prc_dir, imgname + prc_suffix + imgext),
            cv2.IMREAD_COLOR_BGR,
        )
        img_gt = cv2.imread(
            os.path.join(args.gt_dir, imgname + imgext),
            cv2.IMREAD_COLOR_BGR,
        )

        if img_mse is None or img_gan is None:
            print(f"  WARNING: could not read struct/prc image, skipping.")
            continue

        # --- GFPGAN face restoration + histogram matching (final only) ---
        if version == "final":
            # Restore faces on the perception-oriented output
            cropped_faces, restored_faces, img_face = restorer.enhance(
                img=img_gan,
                has_aligned=False,
                only_center_face=False,
                paste_back=True,
                weight=0.5,
            )

            # Save face comparison visualisations
            for idx, (cropped_face, restored_face) in enumerate(
                zip(cropped_faces, restored_faces)
            ):
                cmp_img = np.concatenate((cropped_face, restored_face), axis=1)
                cv2.imwrite(
                    os.path.join(face_cmp_dir, f"{imgname}_{idx:02d}.png"), cmp_img
                )

            # Histogram matching: align face-restored output to structural output
            img_face_histo = histogram_matching(img_face, img_mse)

        # --- Edge-guided blending ---
        gray = cv2.cvtColor(img_mse, cv2.COLOR_BGR2GRAY)
        sharpness = calculate_sobel(gray, resize=False)
        kernel = np.ones((7, 7), np.uint8)
        sharpness = cv2.dilate(sharpness, kernel, iterations=1)

        if version == "edge":
            img_out = merge_images(img_mse, img_gan, sharpness, reverse=True)
        else:
            # 'final': blend structural with face-restored + histogram-matched output
            img_out = merge_images(img_mse, img_face_histo, sharpness, reverse=True)

        # --- Save result ---
        cv2.imwrite(os.path.join(args.output_dir, f"{imgname}{imgext}"), img_out)

    print("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ensemble SR images using GFPGAN-based adaptive facial restoration.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--struct_suffix", type=str, default="", help="Filename suffix for structure-oriented SR images (e.g. '_SwinIR').")
    parser.add_argument("--prc_suffix", type=str, default="", help="Filename suffix for perception-oriented SR images (e.g. '_PFT').")

    parser.add_argument(
        "--version",
        type=str,
        required=True,
        choices=["edge", "final"],
        help="'edge': edge blending only. 'final': edge blending + GFPGAN face restoration + histogram matching.",
    )
    parser.add_argument("--gt_dir", type=str, required=True, help="Directory containing HR ground-truth images.")
    parser.add_argument("--struct_dir", type=str, required=True, help="Directory containing structure-oriented (MSE) SR images.")
    parser.add_argument("--prc_dir", type=str, required=True, help="Directory containing perception-oriented (GAN) SR images.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output images.")
    parser.add_argument(
        "--gfpgan_model",
        type=str,
        default="pretrained_models/GFPGANv1.4.pth",
        help="Path to GFPGAN model weights (default: pretrained_models/GFPGANv1.4.pth). Only used with --version final.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    args = parse_args()
    generate(args)
