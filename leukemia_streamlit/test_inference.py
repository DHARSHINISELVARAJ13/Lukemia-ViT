# -*- coding: utf-8 -*-
"""
Leukemia ViT Inference System - CLI Test Script
================================================
Command-line utility to test and verify model loading and single-image inference
without launching the Streamlit interface.

RESTRICTION:
Uses strictly the official Phase 7 Experiment C best_checkpoint.pt.

Usage:
    python test_inference.py --image path/to/cell_image.jpg
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to sys.path if run directly
current_dir = Path(__file__).resolve().parent
if str(current_dir.parent) not in sys.path:
    sys.path.insert(0, str(current_dir.parent))
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Configure stdout for UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from leukemia_streamlit.config import (
    BEST_CHECKPOINT,
    CLASS_DESCRIPTIONS,
    DEFAULT_CHECKPOINT_NAME,
    RESEARCH_DISCLAIMER,
)
from leukemia_streamlit.inference import run_inference
from leukemia_streamlit.model_loader import (
    load_model_from_checkpoint,
    select_inference_device,
    verify_model_pipeline,
)
from leukemia_streamlit.preprocessing import validate_and_load_image


def format_ascii_bar(prob: float, bar_length: int = 24) -> str:
    """Creates a text-based progress bar for terminal output."""
    filled = int(round(prob * bar_length))
    bar = "=" * filled + " " * (bar_length - filled)
    return bar


def main():
    parser = argparse.ArgumentParser(
        description=f"Leukemia ViT Inference System - CLI Testing Tool (Strictly {DEFAULT_CHECKPOINT_NAME})"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to an input image (PNG/JPG/WEBP). Defaults to bundled sample image if available.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=f"Path to {DEFAULT_CHECKPOINT_NAME}. Must be {DEFAULT_CHECKPOINT_NAME}.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu", "auto"],
        default="auto",
        help="Inference compute device. Default is auto-detect.",
    )
    parser.add_argument(
        "--no-tta",
        action="store_true",
        help="Disable frozen Phase 5 4-view TTA and run single-view inference instead.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON results only.",
    )

    args = parser.parse_args()

    # Determine image path
    if args.image:
        image_path = Path(args.image)
    else:
        sample_path = current_dir / "sample_images" / "cell_sample_01.jpg"
        fallback_path = current_dir.parent / "scratch" / "test_cell_1.jpg"
        if sample_path.is_file():
            image_path = sample_path
        elif fallback_path.is_file():
            image_path = fallback_path
        else:
            print("Error: No image path provided and default sample image not found.")
            print("Please specify an image with: --image path/to/image.jpg")
            sys.exit(1)

    # Determine checkpoint path (strictly best_checkpoint.pt)
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint).resolve()
        if ckpt_path.name != DEFAULT_CHECKPOINT_NAME and ckpt_path.name != "best_checkpoint.pt":
            print(
                f"[ERROR] Inference is restricted to {DEFAULT_CHECKPOINT_NAME}. Rejected: '{ckpt_path.name}'",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        ckpt_path = Path(BEST_CHECKPOINT).resolve()

    # Device selection
    if args.device == "cuda":
        import torch
        if not torch.cuda.is_available():
            print("Warning: CUDA requested but not available. Falling back to CPU.")
            device = torch.device("cpu")
            device_desc = "CPU (Fallback)"
        else:
            device = torch.device("cuda")
            device_desc = f"CUDA ({torch.cuda.get_device_name(0)})"
    elif args.device == "cpu":
        import torch
        device = torch.device("cpu")
        device_desc = "CPU"
    else:
        device, device_desc = select_inference_device()

    if not args.json:
        print("=" * 68)
        print("     LEUKEMIA IMAGE CLASSIFICATION - ViT INFERENCE SYSTEM")
        print("=" * 68)
        print(f"Device:           {device_desc}")
        print(f"Checkpoint:       {ckpt_path}")
        print(f"Target Image:     {image_path}")
        print(f"Inference Mode:   {'Phase 5 Frozen 4-View TTA' if not args.no_tta else 'Single-View Deterministic'}")
        print("-" * 68)

    # 1. Load model
    try:
        model, metadata = load_model_from_checkpoint(ckpt_path, device=device)
    except Exception as e:
        print(f"\n[ERROR] Model loading failed: {e}", file=sys.stderr)
        sys.exit(1)

    # 2. Run pipeline verification
    is_valid, checks, err = verify_model_pipeline(model, metadata, device)
    if not is_valid:
        print(f"\n[ERROR] Model verification FAILED: {err}", file=sys.stderr)
        for c in checks:
            print(f"  [FAIL] {c}", file=sys.stderr)
        sys.exit(1)

    if not args.json:
        print(f"Model Architecture: {metadata.get('model_name')}")
        print(f"Training Run ID:    {metadata.get('RUN_ID')}")
        print(f"Resolution:         {metadata.get('image_size')}x{metadata.get('image_size')}")
        print(f"Model verification: PASSED ({len(checks)}/{len(checks)} checks successful)")
        print("-" * 68)

    # 3. Validate and load image
    valid_img, pil_img, img_meta, img_err = validate_and_load_image(image_path)
    if not valid_img or pil_img is None:
        print(f"\n[ERROR] Image validation failed: {img_err}", file=sys.stderr)
        sys.exit(1)

    if not args.json:
        print(f"Image Resolution:   {img_meta.get('final_width')} x {img_meta.get('final_height')} (RGB)")
        print(f"Image File Size:    {img_meta.get('file_size_kb')} KB")
        print("-" * 68)

    # 4. Run inference
    try:
        result = run_inference(model, pil_img, metadata, device, use_tta=not args.no_tta)
    except Exception as e:
        print(f"\n[ERROR] Inference failed: {e}", file=sys.stderr)
        sys.exit(1)

    # 5. Output results
    if args.json:
        print(json.dumps(result["download_payload"], indent=2))
    else:
        print("\nPREDICTION RESULT")
        print("-" * 40)
        print(f"Predicted class:    {result['predicted_class']} ({result['predicted_class_description']})")
        print(f"Confidence:         {result['confidence_percent']}")
        print(f"Class Index:        {result['predicted_class_index']}")
        print(f"Inference Mode:     {'Phase 5 Frozen 4-View TTA' if result.get('tta_enabled') else 'Single-View'}")
        print("\nCLASS PROBABILITIES")
        print("-" * 68)
        for class_name, prob, desc in result["sorted_probabilities"]:
            bar = format_ascii_bar(prob)
            marker = "<-- [Top Prediction]" if class_name == result["predicted_class"] else ""
            print(f"  {class_name:<4}  [{bar}]  {prob * 100:6.2f}%   {desc:<34} {marker}")
        print("-" * 68)
        print("\n[NOTE] " + RESEARCH_DISCLAIMER.replace("\n\n", " "))
        print("=" * 68)


if __name__ == "__main__":
    main()
