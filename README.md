# Post-Processing Ensemble Framework for Balancing Fidelity and Perception in Super-Resolution

This repository provides the official implementation of the **Ensemble Super-Resolution (ESR) Framework**, which combines structure-oriented and perception-oriented SR outputs through edge-guided blending and adaptive facial restoration.

## Overview

Single-image super-resolution (SR) models typically optimize for either **structural fidelity** (e.g., PSNR-oriented) or **perceptual quality** (e.g., GAN-based). Our framework ensembles both outputs to achieve a balanced result:

1. **Edge-guided blending** — Uses Sobel gradient magnitude to adaptively merge structural and perceptual SR outputs, preserving sharp edges from the structural output while retaining perceptual details elsewhere.
2. **Histogram matching** — Aligns the colour distribution of the perceptual output to the structural output via Reinhard colour transfer (L channel).
3. **Adaptive facial restoration** — Detects faces and selectively restores them using either:
   - **FGResQ** — A CLIP-based face quality assessor that gates face restoration based on quality scores.
   - **GFPGAN** — A GAN-based face restoration network for high-quality facial feature synthesis.

## Pipeline

```
Structure-oriented SR (I_struct)  ──┐
                                    ├──  Edge-guided blending  ──  Histogram matching  ──  Face restoration  ──  Output
Perception-oriented SR (I_prc)    ──┘
```

## Interactive Viewer
https://tama0728.github.io/ESR_viewer/

## Requirements

### 1. Create Conda Environment

```bash
conda env create -f environment.yml
conda activate ESR
```

### 2. Install PyTorch (CUDA-dependent — install separately)

PyTorch, torchvision, and torchaudio must be installed separately based on your CUDA version.
They are **not** included in `environment.yml`.

#### Conda install

**PyTorch 1.13.0 + CUDA 11.7:**

```bash
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia
```

## Data Preparation

Organize your data as follows:

```
data/
├── SR/
│   ├── Set5/
│   │   └── HR/                    # Ground-truth HR images
│   ├── Set14/
│   │   └── HR/
│   └── ...
├── swinir_classical_sr_x4_Set5/   # Structure-oriented SR results
├── swinir_real_sr_x4_Set5/        # Perception-oriented SR results
└── ...
```

### Filename Convention

SR output images should follow a naming convention based on the GT image name with an appended suffix:

| GT filename | struct_suffix | prc_suffix | Struct filename | Prc filename |
|---|---|---|---|---|
| `img001.png` | `_SwinIR` | `_PFT` | `img001_SwinIR.png` | `img001_PFT.png` |

The `--struct_suffix` and `--prc_suffix` arguments let you specify these suffixes for any SR model combination.

## Usage

### 1. Generate Ensemble Images

#### Using FGResQ (face quality-aware restoration)

```bash
# 'edge' version: edge-guided blending only
python generate_ensemble_fgresq.py \
    --version edge \
    --gt_dir data/SR/Set5/HR/ \
    --struct_dir data/swinir_classical_sr_x4_Set5/ \
    --struct_suffix _SwinIR \
    --prc_dir data/swinir_real_sr_x4_Set5/ \
    --prc_suffix _PFT \
    --output_dir results/FGResQ/Set5/

# 'final' version: edge blending + histogram matching + face restoration
python generate_ensemble_fgresq.py \
    --version final \
    --gt_dir data/SR/Set5/HR/ \
    --struct_dir data/swinir_classical_sr_x4_Set5/ \
    --struct_suffix _SwinIR \
    --prc_dir data/swinir_real_sr_x4_Set5/ \
    --prc_suffix _PFT \
    --output_dir results/FGResQ/Set5/ \
    --model_path FGResQ.pth
```

#### Using GFPGAN (GAN-based face restoration)

```bash
# 'final' version: edge blending + GFPGAN face restoration + histogram matching
python generate_ensemble_gfpgan.py \
    --version final \
    --gt_dir data/SR/Set5/HR/ \
    --struct_dir data/swinir_classical_sr_x4_Set5/ \
    --struct_suffix _SwinIR \
    --prc_dir data/swinir_real_sr_x4_Set5/ \
    --prc_suffix _PFT \
    --output_dir results/GFPGAN/Set5/ \
    --gfpgan_model pretrained_models/GFPGANv1.4.pth
```

### 2. Evaluate Metrics

Compute PSNR, SSIM, LPIPS, and NIQE metrics and save to Excel:

```bash
# Evaluate ensemble output
python evaluate_metrics.py \
    --version final \
    --gt_dir data/SR/Set5/HR/ \
    --struct_dir data/swinir_classical_sr_x4_Set5/ \
    --struct_suffix _SwinIR \
    --prc_dir data/swinir_real_sr_x4_Set5/ \
    --prc_suffix _PFT \
    --result_dir results/GFPGAN/Set5/ \
    --output_dir metrics/ \
    --dataset Set5

# Evaluate raw structural (MSE) SR output
python evaluate_metrics.py \
    --version mse \
    --gt_dir data/SR/Set5/HR/ \
    --struct_dir data/swinir_classical_sr_x4_Set5/ \
    --struct_suffix _SwinIR \
    --prc_dir data/swinir_real_sr_x4_Set5/ \
    --prc_suffix _PFT \
    --output_dir metrics/ \
    --dataset Set5

# Evaluate raw perceptual (GAN) SR output
python evaluate_metrics.py \
    --version gan \
    --gt_dir data/SR/Set5/HR/ \
    --struct_dir data/swinir_classical_sr_x4_Set5/ \
    --struct_suffix _SwinIR \
    --prc_dir data/swinir_real_sr_x4_Set5/ \
    --prc_suffix _PFT \
    --output_dir metrics/ \
    --dataset Set5
```

## Script Arguments

### `generate_ensemble_fgresq.py`

| Argument | Required | Description |
|---|---|---|
| `--version` | ✅ | `edge` or `final` |
| `--gt_dir` | ✅ | HR ground-truth image directory |
| `--struct_dir` | ✅ | Structure-oriented SR image directory |
| `--struct_suffix` |  | Filename suffix for structural images (e.g. `_SwinIR`) |
| `--prc_dir` | ✅ | Perception-oriented SR image directory |
| `--prc_suffix` |  | Filename suffix for perceptual images (e.g. `_PFT`) |
| `--output_dir` | ✅ | Output directory |
| `--model_path` | | FGResQ model weights (default: `FGResQ.pth`) |

### `generate_ensemble_gfpgan.py`

| Argument | Required | Description |
|---|---|---|
| `--version` | ✅ | `edge` or `final` |
| `--gt_dir` | ✅ | HR ground-truth image directory |
| `--struct_dir` | ✅ | Structure-oriented SR image directory |
| `--struct_suffix` |  | Filename suffix for structural images |
| `--prc_dir` | ✅ | Perception-oriented SR image directory |
| `--prc_suffix` |  | Filename suffix for perceptual images |
| `--output_dir` | ✅ | Output directory |
| `--gfpgan_model` | | GFPGAN model weights (default: `pretrained_models/GFPGANv1.4.pth`) |

### `evaluate_metrics.py`

| Argument | Required | Description |
|---|---|---|
| `--version` | ✅ | `mse`, `gan`, `edge`, or `final` |
| `--gt_dir` | ✅ | HR ground-truth image directory |
| `--struct_dir` | ✅ | Structure-oriented SR image directory |
| `--struct_suffix` |  | Filename suffix for structural images |
| `--prc_dir` | ✅ | Perception-oriented SR image directory |
| `--prc_suffix` |  | Filename suffix for perceptual images |
| `--output_dir` | ✅ | Directory to save Excel results |
| `--result_dir` | ✅* | Ensemble result directory (*required for `edge`/`final`) |
| `--dataset` | | Dataset name for output filename (inferred from `gt_dir` if omitted) |
| `--metrics` | | Metrics to compute (default: `psnry ssimc lpips niqe`) |
| `--append` | | Append to existing Excel file |

## Pre-trained Models

| Model | Description | Download |
|---|---|---|
| FGResQ | CLIP-based face quality assessor | [HuggingFace](https://huggingface.co/orpheus0429/FGResQ) |
| GFPGAN v1.4 | GAN-based face restoration | [GitHub](https://github.com/TencentARC/GFPGAN) |

## Acknowledgements

- [SwinIR](https://github.com/JingyunLiang/SwinIR)
- [GFPGAN](https://github.com/TencentARC/GFPGAN)
- [PyIQA](https://github.com/chaofengc/IQA-PyTorch)
- [MediaPipe](https://github.com/google/mediapipe)
