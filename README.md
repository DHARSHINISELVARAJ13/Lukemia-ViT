# Leukemia Single-Cell Classification using Vision Transformer (ViT-B/16)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: Research Use Only](https://img.shields.io/badge/License-Research%20Use%20Only-yellow.svg)](#disclaimer)

A production-ready clinical research dashboard for microscopic single-cell leukemia classification powered by a fine-tuned Vision Transformer (**ViT-B/16** at $384 \times 384$ resolution).

---

## Overview

Accurate subtype identification of white blood cells is critical for diagnosis and treatment planning in hematologic malignancies. This repository provides an end-to-end inference system for classifying single-cell microscopic images into 5 diagnostic categories:
- **ALL**: Acute Lymphoblastic Leukemia
- **AML**: Acute Myeloid Leukemia
- **CLL**: Chronic Lymphocytic Leukemia
- **CML**: Chronic Myeloid Leukemia
- **H**: Healthy / Normal Hematopoietic Control

The inference pipeline incorporates a frozen **4-view Test-Time Augmentation (TTA)** strategy, input sanitization, automated pipeline integrity verification, and interactive probability distribution visualization.

---

## Key Features

- **Intuitive Web Interface**: Built with Streamlit, supporting drag-and-drop file upload, sample testing, and native clipboard image paste (`Ctrl+V`).
- **Vision Transformer Backbone**: Pretrained ViT-B/16 fine-tuned on single-cell cytology imagery with layerwise learning rates and decision-boundary refinement.
- **Test-Time Augmentation (TTA)**: 4-view deterministic TTA (Original, Horizontal Flip, $+5^\circ$, $-5^\circ$) with mean softmax probability ensembling for stable predictions.
- **Defensive Model Verification**: Automated 8-point pre-flight checklist verifying model weights, input resolution, tensor shapes, and probability axioms.
- **Exportable Diagnostics**: Downloadable structured JSON prediction report including model lineage, confidence levels, and timestamped forensic metadata.

---

## Model Specifications

| Parameter | Specification |
| :--- | :--- |
| **Backbone Architecture** | `vit_base_patch16_224` (`timm`) |
| **Input Resolution** | $384 \times 384 \times 3$ (Bilinear interpolation) |
| **Sequence Length** | 577 tokens ($24 \times 24$ spatial patches + 1 `[CLS]` token) |
| **Training Lineage** | Phase 7 Experiment C (Decision-Boundary Refinement) |
| **Certified Test Accuracy** | **97.02%** (95% CI: [96.54%, 97.48%]) |
| **Certified Test Macro F1** | **97.04%** (95% CI: [96.55%, 97.48%]) |
| **Test Classification Errors** | **149 / 5,000** |
| **High-Confidence Errors ($\ge 80\%$)**| **0** |
| **Deployment Checkpoint** | `model/best_checkpoint.pt` (328.47 MB, tracked via Git LFS) |
| **Inference SHA-256** | `6af819a4be144a6323077083b6bdf2c7c5b03c88ef4ac0869cce2e3eec40ecad` |

For detailed architectural and training specifications, see [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md).

---

## Repository Structure

```
Lukemia-ViT/
├── app.py                      # Primary Streamlit application entry point
├── requirements.txt            # Python dependencies for inference
├── README.md                   # Project documentation
├── .gitignore                  # Exclusion rules for datasets, caches & secrets
├── .gitattributes              # Git LFS configuration for model checkpoint
│
├── model/
│   └── best_checkpoint.pt      # Production ViT-B/16 inference weights (Git LFS)
│
├── leukemia_streamlit/         # Application source package
│   ├── app.py                  # Core Streamlit UI dashboard
│   ├── config.py               # Application configuration and paths
│   ├── model_loader.py         # Defensive checkpoint loader & integrity verifier
│   ├── preprocessing.py        # 384x384 image transforms & validation
│   ├── inference.py            # TTA inference engine & result packaging
│   ├── test_inference.py       # Standalone CLI test utility
│   └── clipboard_component/    # Custom clipboard paste frontend component
│
└── docs/
    ├── MODEL_CARD.md           # Model card with architecture and certified metrics
    └── DATASET_INTEGRITY_SUMMARY.md # Summary of perceptual duplicate & leakage audit
```

---

## Installation & Setup

### 1. Prerequisites
- Python 3.10 or higher
- Git and [Git LFS](https://git-lfs.com) installed
- (Optional) NVIDIA GPU with CUDA 11.8+ for hardware acceleration

### 2. Clone the Repository
```bash
git clone https://github.com/DHARSHINISELVARAJ13/Lukemia-ViT.git
cd Lukemia-ViT
```

*Note: If you have Git LFS installed, `git clone` automatically downloads `model/best_checkpoint.pt`. If needed, manually pull the model weights with:*
```bash
git lfs pull
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Running the Application

### Web Application (Streamlit)
Launch the Streamlit interface:
```bash
streamlit run app.py
```
Or run directly via the package:
```bash
streamlit run leukemia_streamlit/app.py
```
Open your browser at `http://localhost:8501`.

### Command-Line Interface (CLI Test)
Run single-image inference directly from the terminal:
```bash
python leukemia_streamlit/test_inference.py --image path/to/cell_image.jpg
```

---

## Dataset Notice

> **Original Dataset Excluded**:
> In compliance with medical data privacy guidelines and research integrity best practices, the raw microscopic training and evaluation datasets are **NOT** hosted or redistributed in this repository.
> 
> A comprehensive data integrity and perceptual duplicate audit was performed prior to model evaluation to prevent cross-split leakage (documented in [`docs/DATASET_INTEGRITY_SUMMARY.md`](docs/DATASET_INTEGRITY_SUMMARY.md)).

---

## Medical & Clinical Disclaimer

> **RESEARCH USE ONLY (RUO)**:
> This software and model are intended strictly for computational research, algorithm benchmarking, and educational purposes. 
> 
> - The predictions produced by this software **do NOT constitute medical diagnoses**.
> - This software must **never** be used as a standalone diagnostic tool or to guide patient clinical care without supervision by a certified pathologist.
> - The authors assume no liability for decisions made based on outputs generated by this model.
