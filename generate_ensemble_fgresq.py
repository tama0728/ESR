"""Generate ensemble super-resolution images using FGResQ-based adaptive facial restoration.

This script combines structure-oriented (MSE) and perception-oriented (GAN) SR outputs
using edge-guided blending and optional face quality-aware restoration via FGResQ.

Usage:
    # Generate 'final' version (edge blending + histogram matching + face restoration)
    python generate_ensemble_fgresq.py \
        --version final \
        --gt_dir data/SR/Set5/HR/ \
        --struct_dir data/swinir_classical_sr_x4_Set5/ \
        --struct_suffix _SwinIR \
        --prc_dir data/swinir_real_sr_x4_Set5/ \
        --prc_suffix x4_SwinIR \
        --output_dir results/FGResQ/SwinIR/Set5/

    # Generate 'edge' version (edge blending only)
    python generate_ensemble_fgresq.py \
        --version edge \
        --gt_dir data/SR/Set5/HR/ \
        --struct_dir data/swinir_classical_sr_x4_Set5/ \
        --struct_suffix _SwinIR \
        --prc_dir data/swinir_real_sr_x4_Set5/ \
        --prc_suffix x4_SwinIR \
        --output_dir results/FGResQ/SwinIR/Set5/
"""

import argparse
import os
import glob
import time

import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
import torchvision.transforms as transforms

from FGResQ import FGResQ





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
    """Alpha‑blend *source* and *ref* using *mask*."""
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


def _preprocess_image(img_rgb: np.ndarray):
    """Resize and normalise an RGB image for FGResQ inference."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop(224),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        ),
    ])
    try:
        img = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_LINEAR)
        image = Image.fromarray(img)
        image_tensor = transform(image).unsqueeze(0)
        return image_tensor.to("cuda")
    except Exception as e:
        print(f"Error processing image: {e}")
        return None


# ---------------------------------------------------------------------------
# Main generation pipeline
# ---------------------------------------------------------------------------
def generate(args) -> None:
    """Run the ensemble image generation pipeline."""

    struct_suffix = args.struct_suffix
    prc_suffix = args.prc_suffix
    version = args.version

    # FGResQ model (needed for face‑quality gating in 'final' mode)
    fgresq = FGResQ(model_path=args.model_path)

    # MediaPipe face detector (needed in 'final' mode)
    mp_face_detection = mp.solutions.face_detection
    face_detector = mp_face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.5,
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

        # --- Histogram matching (final only) ---
        if version == "final":
            img_gan = histogram_matching(img_gan, img_mse)
            img = img_gan
        else:
            img = img_gan  # will be overwritten by edge blending below

        # --- Edge‑guided blending (both versions) ---
        gray = cv2.cvtColor(img_mse, cv2.COLOR_BGR2GRAY)
        sharpness = calculate_sobel(gray, resize=False)
        kernel = np.ones((7, 7), np.uint8)
        sharpness = cv2.dilate(sharpness, kernel, iterations=1)
        img = merge_images(img_mse, img_gan, sharpness, reverse=True)

        # --- Face quality‑aware restoration (final only) ---
        if version == "final":
            rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = face_detector.process(rgb_image)

            if results.detections:
                h, w, _ = img.shape
                dectection_zone = img.copy()

                mask_mse = np.zeros((h, w), dtype=np.float64)
                mask_gan = np.zeros((h, w), dtype=np.float64)

                for idx, detection in enumerate(results.detections):
                    bounding_box = detection.location_data.relative_bounding_box
                    x = int(bounding_box.xmin * w)
                    y = int(bounding_box.ymin * h)
                    width = int(bounding_box.width * w)
                    height = int(bounding_box.height * h)

                    x -= height // 4
                    y -= width // 4
                    x = max(x, 0)
                    y = max(y, 0)
                    width += width // 2
                    height += height // 2
                    if width + x > w:
                        width = w - x
                    if height + y > h:
                        height = h - y

                    struct_face = img_mse[y : y + height, x : x + width].copy()
                    prc_face = img_gan[y : y + height, x : x + width].copy()
                    our1_face = img[y : y + height, x : x + width].copy()

                    struct_tensor = _preprocess_image(cv2.cvtColor(struct_face, cv2.COLOR_BGR2RGB))
                    prc_tensor = _preprocess_image(cv2.cvtColor(prc_face, cv2.COLOR_BGR2RGB))
                    our1_tensor = _preprocess_image(cv2.cvtColor(our1_face, cv2.COLOR_BGR2RGB))

                    try:
                        fgr_struct = fgresq.model(struct_tensor)[0].squeeze().item()
                        fgr_prc = fgresq.model(prc_tensor)[0].squeeze().item()
                        fgr_our1 = fgresq.model(our1_tensor)[0].squeeze().item()
                    except RuntimeError:
                        print("  face crop too small for FGResQ, skipping face")
                        fgr_struct = fgr_prc = fgr_our1 = 0

                    # Save face comparison visualisation
                    new_width, new_height = width, height
                    if width < 500:
                        new_width = 500
                        new_height = int(height * new_width / width)
                        struct_face = cv2.resize(struct_face, (new_width, new_height), interpolation=cv2.INTER_AREA)
                        prc_face = cv2.resize(prc_face, (new_width, new_height), interpolation=cv2.INTER_AREA)
                        our1_face = cv2.resize(our1_face, (new_width, new_height), interpolation=cv2.INTER_AREA)

                    canvas = np.zeros((new_height + 100, new_width * 3, 3), dtype=np.uint8)
                    canvas[:new_height, :new_width] = struct_face
                    canvas[:new_height, new_width : new_width * 2] = prc_face
                    canvas[:new_height, new_width * 2 :] = our1_face
                    cv2.putText(canvas, f"Str FGR: {fgr_struct:.5f}", (0, new_height + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(canvas, f"Prc FGR: {fgr_prc:.5f}", (new_width, new_height + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(canvas, f"Our FGR: {fgr_our1:.5f}", (new_width * 2, new_height + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imwrite(os.path.join(face_cmp_dir, f"{imgname}_face_{idx}.png"), canvas)

                    if fgr_our1 > fgr_struct and fgr_our1 > fgr_prc:
                        continue

                    # 2‑D Gaussian kernel for soft face mask
                    kernel_y = cv2.getGaussianKernel(width, width // 3)
                    kernel_x = cv2.getGaussianKernel(height, height // 3)
                    kernel_2d = np.outer(kernel_x, kernel_y.transpose()) + 0.00001

                    if fgr_struct > fgr_prc:
                        mask_mse[y : y + height, x : x + width] += kernel_2d / np.max(kernel_2d) * 255
                        mask_mse[mask_mse > 255] = 255
                    else:
                        mask_gan[y : y + height, x : x + width] += kernel_2d / np.max(kernel_2d) * 255
                        mask_gan[mask_gan > 255] = 255

                    cv2.rectangle(dectection_zone, (x, y), (x + width, y + height), 255, 2)

                # Whole‑image face comparison
                img_mse_tensor = _preprocess_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                img_copy = img.copy()

                img = merge_images(img, img_mse, mask_mse)
                img = merge_images(img, img_gan, mask_gan)

                img_tensor = _preprocess_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                try:
                    fgr_before = fgresq.model(img_mse_tensor)[0].squeeze().item()
                    fgr_after = fgresq.model(img_tensor)[0].squeeze().item()
                except (RuntimeError, Exception) as e:
                    print(f"  whole image FGResQ error: {e}")
                    fgr_before = fgr_after = 0

                canvas = np.zeros((h + 100, w * 2, 3), dtype=np.uint8)
                canvas[:h, :w] = img_copy
                canvas[:h, w:] = img
                cv2.putText(canvas, f"Before FGR: {fgr_before:.5f}", (0, h + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(canvas, f"After FGR: {fgr_after:.5f}", (w, h + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imwrite(os.path.join(face_cmp_dir, f"{imgname}_face_compare_whole.png"), canvas)

        # --- Save result ---
        cv2.imwrite(os.path.join(args.output_dir, f"{imgname}{imgext}"), img)

    print("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ensemble SR images using FGResQ-based adaptive facial restoration.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--struct_suffix", type=str, default="", help="Filename suffix for structure-oriented SR images (e.g. '_SwinIR').")
    parser.add_argument("--prc_suffix", type=str, default="", help="Filename suffix for perception-oriented SR images (e.g. '_PFT').")

    parser.add_argument(
        "--version",
        type=str,
        required=True,
        choices=["edge", "final"],
        help="'edge': edge blending only. 'final': edge blending + histogram matching + face restoration.",
    )
    parser.add_argument("--gt_dir", type=str, required=True, help="Directory containing HR ground-truth images.")
    parser.add_argument("--struct_dir", type=str, required=True, help="Directory containing structure-oriented (MSE) SR images.")
    parser.add_argument("--prc_dir", type=str, required=True, help="Directory containing perception-oriented (GAN) SR images.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output images.")
    parser.add_argument("--model_path", type=str, default="FGResQ.pth", help="Path to FGResQ model weights (default: FGResQ.pth).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate(args)
