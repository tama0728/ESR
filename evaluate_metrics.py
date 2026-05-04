"""Evaluate image quality metrics for super-resolution outputs using PyIQA.

Computes full-reference (PSNR, SSIM, LPIPS) and no-reference (NIQE) metrics
and saves per-image results with averages to an Excel file.

Dependencies:
    pip install pyiqa pandas openpyxl torch

Usage:
    # Evaluate metrics for ensemble 'final' output
    python evaluate_metrics.py \
        --version final \
        --gt_dir data/SR/Set5/HR/ \
        --struct_dir data/swinir_classical_sr_x4_Set5/ \
        --struct_suffix _SwinIR \
        --prc_dir data/swinir_real_sr_x4_Set5/ \
        --prc_suffix x4_SwinIR \
        --result_dir results/GFPGAN/SwinIR/Set5/ \
        --output_dir metrics/SwinIR/

    # Evaluate the raw structural (MSE) outputs
    python evaluate_metrics.py \
        --version mse \
        --gt_dir data/SR/Set5/HR/ \
        --struct_dir data/swinir_classical_sr_x4_Set5/ \
        --struct_suffix _SwinIR \
        --prc_dir data/swinir_real_sr_x4_Set5/ \
        --prc_suffix x4_SwinIR \
        --output_dir metrics/SwinIR/

    # Append new metrics to an existing Excel file
    python evaluate_metrics.py \
        --version final \
        --gt_dir data/SR/Set5/HR/ \
        --struct_dir data/swinir_classical_sr_x4_Set5/ \
        --struct_suffix _SwinIR \
        --prc_dir data/swinir_real_sr_x4_Set5/ \
        --prc_suffix x4_SwinIR \
        --result_dir results/GFPGAN/SwinIR/Set5/ \
        --output_dir metrics/SwinIR/ \
        --append
"""

import argparse
import os
import time
import glob
from collections import OrderedDict

import pandas as pd
import torch
import pyiqa




# ---------------------------------------------------------------------------
# Metric configuration
# ---------------------------------------------------------------------------
METRIC_CONFIG = OrderedDict([
    ("psnry",  {"column": "PSNR",  "needs_ref": True}),
    ("ssimc",  {"column": "SSIM",  "needs_ref": True}),
    ("lpips",  {"column": "LPIPS", "needs_ref": True}),
    ("niqe",   {"column": "NIQE",  "needs_ref": False}),
])




# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def create_folder(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def resolve_image_path(
    version: str,
    struct_suffix: str,
    prc_suffix: str,
    gt_dir: str,
    struct_dir: str,
    prc_dir: str,
    result_dir: str,
    imgname: str,
    imgext: str,
) -> tuple:
    """Return (image_path, gt_path) based on the evaluation version."""
    img_gt = os.path.join(gt_dir, imgname + imgext)

    if version == "mse":
        img = os.path.join(struct_dir, imgname + struct_suffix + imgext)
    elif version == "gan":
        img = os.path.join(prc_dir, imgname + prc_suffix + imgext)
    else:
        # "edge" or "final" → read from result_dir
        img = os.path.join(result_dir, imgname + imgext)

    return img, img_gt


def compute_metric(iqa_models: dict, metric_name: str, img_path: str, img_gt_path: str) -> float:
    """Compute a single IQA metric score."""
    cfg = METRIC_CONFIG[metric_name]
    if cfg["needs_ref"]:
        return iqa_models[metric_name](img_path, img_gt_path).item()
    else:
        return iqa_models[metric_name](img_path).item()


def avg_valid(values: list, invalid: float = -1) -> float:
    """Average excluding invalid sentinel values."""
    valid = [x for x in values if x != invalid]
    return sum(valid) / len(valid) if valid else invalid


# ---------------------------------------------------------------------------
# Main evaluation pipeline
# ---------------------------------------------------------------------------
def evaluate(args) -> None:
    """Run the metric evaluation pipeline."""

    struct_suffix = args.struct_suffix
    prc_suffix = args.prc_suffix
    version = args.version
    run_action = "append" if args.append else "overwrite"

    # Initialise PyIQA models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metric_names = args.metrics
    iqa_models = {m: pyiqa.create_metric(m, device=device) for m in metric_names}

    create_folder(args.output_dir)

    # Determine dataset name from gt_dir for the output filename
    dataset_name = args.dataset if args.dataset else os.path.basename(os.path.normpath(args.gt_dir))
    excel_filename = os.path.join(args.output_dir, f"{dataset_name}_{version}_metrics.xlsx")

    # ---- Load existing data (append mode) ----
    excel_data = []
    existing_images = set()
    new_metrics = []

    if run_action == "append" and os.path.exists(excel_filename):
        try:
            df_existing = pd.read_excel(excel_filename, dtype={"Image": str})
            df_existing = df_existing[df_existing["Image"] != "Average"]
            existing_images = set(df_existing["Image"])
            excel_data = df_existing.to_dict("records")

            existing_columns = set(df_existing.columns)
            for metric_name in metric_names:
                if metric_name in METRIC_CONFIG:
                    col_name = METRIC_CONFIG[metric_name]["column"]
                    if col_name not in existing_columns:
                        new_metrics.append(metric_name)

            if new_metrics:
                print(f"Detected new metrics to compute for existing images: {new_metrics}")
            print(f"Loaded {len(existing_images)} existing records from {excel_filename}")
        except Exception as e:
            print(f"Failed to read {excel_filename}: {e}")

    # ---- Compute new metrics for existing images ----
    if new_metrics and excel_data:
        print(f"Computing new metrics {new_metrics} for {len(excel_data)} existing images...")
        gt_image_map = {}
        for path in sorted(glob.glob(os.path.join(args.gt_dir, "*"))):
            name_tmp, ext_tmp = os.path.splitext(os.path.basename(path))
            gt_image_map[name_tmp] = ext_tmp

        for idx, row in enumerate(excel_data):
            imgname = row["Image"]
            if imgname not in gt_image_map:
                print(f"  WARNING: {imgname} not found in GT folder, skipping")
                for mn in new_metrics:
                    row[METRIC_CONFIG[mn]["column"]] = -1
                continue

            imgext = gt_image_map[imgname]
            img, img_gt = resolve_image_path(
                version, struct_suffix, prc_suffix, args.gt_dir, args.struct_dir, args.prc_dir,
                args.result_dir or "", imgname, imgext,
            )
            for mn in new_metrics:
                score = compute_metric(iqa_models, mn, img, img_gt)
                row[METRIC_CONFIG[mn]["column"]] = score

            if idx == 0:
                for mn in new_metrics:
                    col = METRIC_CONFIG[mn]["column"]
                    print(f"  {col}: {row[col]}")

        print("Finished computing new metrics for existing images.")

    # ---- Compute metrics for new images ----
    gt_paths = sorted(glob.glob(os.path.join(args.gt_dir, "*")))
    new_image_count = 0

    for i, path in enumerate(gt_paths):
        imgname, imgext = os.path.splitext(os.path.basename(path))

        if imgname in existing_images:
            continue

        new_image_count += 1
        img, img_gt = resolve_image_path(
            version, struct_suffix, prc_suffix, args.gt_dir, args.struct_dir, args.prc_dir,
            args.result_dir or "", imgname, imgext,
        )

        row_data = {"Image": imgname}
        for metric_name in metric_names:
            if metric_name not in METRIC_CONFIG:
                continue
            cfg = METRIC_CONFIG[metric_name]
            s = time.time()
            score = compute_metric(iqa_models, metric_name, img, img_gt)
            if new_image_count == 1:
                print(f"{cfg['column']}: {score:.4f}  ({time.time() - s:.2f}s)")
            row_data[cfg["column"]] = score

        excel_data.append(row_data)

    if new_image_count > 0:
        print(f"Computed metrics for {new_image_count} new images.")

    if not excel_data:
        print("No data to save, exiting.")
        return

    # ---- Calculate average ----
    avg_row = {"Image": "Average"}
    all_columns = set()
    for row in excel_data:
        all_columns.update(row.keys())
    all_columns.discard("Image")

    for col in all_columns:
        values = [row.get(col, -1) for row in excel_data]
        if col in ("PSNR", "SSIM"):
            avg_row[col] = sum(values) / len(values) if values else 0
        else:
            avg_row[col] = avg_valid(values)

    excel_data.append(avg_row)
    print(f"Average PSNR: {avg_row.get('PSNR', 0):.2f}")

    # ---- Save to Excel ----
    df = pd.DataFrame(excel_data)
    ordered_cols = ["Image"]
    for metric_name, cfg in METRIC_CONFIG.items():
        if cfg["column"] in df.columns:
            ordered_cols.append(cfg["column"])
    for c in df.columns:
        if c not in ordered_cols:
            ordered_cols.append(c)
    df = df[ordered_cols]

    try:
        df.to_excel(excel_filename, index=False)
        print(f"Saved results to: {excel_filename}")
    except Exception as e:
        print(f"Error saving to {excel_filename}: {e}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate IQA metrics for super-resolution outputs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--struct_suffix", type=str, default="", help="Filename suffix for structure-oriented SR images (e.g. '_SwinIR', '_PFT_classical_SRx4').")
    parser.add_argument("--prc_suffix", type=str, default="", help="Filename suffix for perception-oriented SR images (e.g. 'x4_SwinIR', 'x4').")

    parser.add_argument(
        "--version",
        type=str,
        required=True,
        choices=["mse", "gan", "edge", "final"],
        help="Which SR output to evaluate: 'mse' (structural), 'gan' (perceptual), 'edge', or 'final'.",
    )
    parser.add_argument("--gt_dir", type=str, required=True, help="Directory containing HR ground-truth images.")
    parser.add_argument("--struct_dir", type=str, required=True, help="Directory containing structure-oriented (MSE) SR images.")
    parser.add_argument("--prc_dir", type=str, required=True, help="Directory containing perception-oriented (GAN) SR images.")
    parser.add_argument(
        "--result_dir",
        type=str,
        default=None,
        help="Directory containing ensemble result images. Required when --version is 'edge' or 'final'.",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save metric results (Excel files).")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name for the output filename. If omitted, inferred from gt_dir.")
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["psnry", "ssimc", "lpips", "niqe"],
        help="IQA metrics to compute (default: psnry ssimc lpips niqe).",
    )
    parser.add_argument("--append", action="store_true", help="Append new metrics to existing Excel file instead of overwriting.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Validate: result_dir is required for edge/final
    if args.version in ("edge", "final") and not args.result_dir:
        raise ValueError("--result_dir is required when --version is 'edge' or 'final'.")

    evaluate(args)
