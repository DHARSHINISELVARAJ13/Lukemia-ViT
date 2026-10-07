# Model Card: Leukemia ViT-B/16 Classification Model

## 1. Model Summary
- **Architecture**: Vision Transformer (`vit_base_patch16_224`, fine-tuned at $384 \times 384$ resolution)
- **Patch Size**: $16 \times 16$
- **Sequence Length**: 577 tokens ($24 \times 24 = 576$ spatial patches + 1 `[CLS]` token)
- **Embedding Dimension**: 768
- **Transformer Depth**: 12 layers, 12 attention heads
- **Total Parameters**: ~86M
- **Lineage**: Official Phase 7 Experiment C (Decision-Boundary Refinement)
- **Deployment Artifact**: `model/best_checkpoint.pt`
- **Model Format**: PyTorch Checkpoint (`model_state_dict` + verified metadata)
- **Artifact Size**: 328.47 MB (optimized inference artifact, training optimizer/scheduler states stripped)
- **Inference Checkpoint SHA-256**: `6af819a4be144a6323077083b6bdf2c7c5b03c88ef4ac0869cce2e3eec40ecad`
- **Training Source Checkpoint SHA-256**: `a1cef719d7bb4c7cda7043301b5751c062cc3b8c86344e8db13fe46674e5509e`

---

## 2. Certified Performance Metrics
Evaluated on the certified frozen test partition (5,000 single-cell images):

| Metric | Certified Result | 95% Confidence Interval |
| :--- | :--- | :--- |
| **Test Accuracy** | **97.02%** | [96.54%, 97.48%] |
| **Test Macro F1** | **97.04%** | [96.55%, 97.48%] |
| **Validation Accuracy** | **97.05%** | — |
| **Validation Macro F1** | **96.84%** | — |
| **Test Classification Errors** | **149 / 5,000** | — |
| **High-Confidence Errors ($\ge 80\%$)** | **0** | — |

---

## 3. Clinical Target Classes
The model outputs a 5-class probability distribution:

| Index | Label | Clinical Subtype | Description |
| :---: | :---: | :--- | :--- |
| `0` | `ALL` | Acute Lymphoblastic Leukemia | Aggressive malignancy with overproduction of lymphoblasts |
| `1` | `AML` | Acute Myeloid Leukemia | Rapid clonal proliferation of abnormal myeloblasts |
| `2` | `CLL` | Chronic Lymphocytic Leukemia | Accumulation of mature-appearing abnormal lymphocytes |
| `3` | `CML` | Chronic Myeloid Leukemia | Myeloproliferative disorder with uncontrolled granulocytes |
| `4` | `H` | Healthy / Normal Control | Non-leukemic physiological hematopoietic control cells |

---

## 4. Inference Preprocessing & Pipeline
1. **Input**: Single-cell peripheral blood or bone marrow microscopic image.
2. **Resize**: Resampled to $384 \times 384$ pixels using bilinear interpolation.
3. **Normalization**: Standard ImageNet statistics:
   - Mean: `[0.485, 0.456, 0.406]`
   - Std: `[0.229, 0.224, 0.225]`
4. **Test-Time Augmentation (TTA)**: Phase 5 frozen 4-view ensemble:
   - View 1: Original image
   - View 2: Horizontal flip
   - View 3: $+5^\circ$ rotation
   - View 4: $-5^\circ$ rotation
   Probabilities are aggregated via the arithmetic mean of softmax outputs.

---

## 5. Intended Use & Clinical Disclaimer
> **RESEARCH USE ONLY (RUO)**:
> This model and application are provided strictly for research, algorithmic benchmarking, and educational purposes. This system is **not** a cleared medical device and must **never** be used as a standalone diagnostic tool or substitute for professional pathologist evaluation.
