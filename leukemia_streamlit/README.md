# Leukemia ViT Inference Application

Interactive Streamlit web application for microscopic single-cell leukemia classification using the fine-tuned Vision Transformer (ViT-B/16 at $384 \times 384$ resolution).

---

## 1. Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Run Web Application
```bash
streamlit run app.py
```
Or from the root directory:
```bash
streamlit run leukemia_streamlit/app.py
```

### CLI Testing
```bash
python test_inference.py --image path/to/image.jpg
```

---

## 2. Model Specifications & Resolution
- **Model Checkpoint**: `model/best_checkpoint.pt`
- **Resolution**: $384 \times 384$ pixels
- **Sequence Length**: 577 tokens ($24 \times 24$ spatial patches + 1 `[CLS]` token)
- **TTA Strategy**: Frozen Phase 5 4-view Test-Time Augmentation
- **Classes**: `ALL`, `AML`, `CLL`, `CML`, `H`

---

## 3. Key Application Features
1. **Side-by-Side Dual Image Input**:
   - File upload (drag & drop / file browser for PNG, JPG, JPEG, WEBP)
   - Browser-side clipboard image paste (`Ctrl+V`)
2. **Automatic Inference Trigger**: Fast evaluation upon image input.
3. **Probability Visualization**: Interactive Plotly horizontal bar charts for all 5 clinical categories.
4. **Structured JSON Export**: Downloadable timestamped prediction report with confidence metrics and model metadata.
5. **Pre-flight Integrity Verification**: Automated 8-point checklist verifying model architecture, state dict matching, and probability axioms.

---

## 4. Research Use Only Disclaimer
> **Research Use Only (RUO)**:
> This application provides model predictions from an image-classification model. The output is not a medical diagnosis and should not be used as a substitute for evaluation by a qualified healthcare professional.
