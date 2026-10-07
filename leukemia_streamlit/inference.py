# -*- coding: utf-8 -*-
"""
Leukemia ViT Inference System - Inference Engine
=================================================
Performs optimized, single-process inference using PyTorch inference mode.
Implements the official Phase 5 frozen 4-view Test-Time Augmentation (TTA)
strategy (Original, Horizontal Flip, +5° Rotation, -5° Rotation) combined via
arithmetic mean of softmax probabilities, with fallback to single-view inference.
"""

from datetime import datetime
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from PIL import Image

from .config import (
    CLASS_DESCRIPTIONS,
    CLASS_NAMES,
    DATASET_FINGERPRINT,
    IMAGE_SIZE,
    NUM_CLASSES,
)
from .preprocessing import preprocess_image


def generate_phase_5_tta_views(image: Image.Image) -> List[Image.Image]:
    """
    Generates the exact frozen Phase 5 4-view TTA transforms:
      1. Original image
      2. Horizontal flip (FLIP_LEFT_RIGHT)
      3. +5 degree rotation (BILINEAR)
      4. -5 degree rotation (BILINEAR)
    """
    return [
        image,
        image.transpose(Image.FLIP_LEFT_RIGHT),
        image.rotate(5, resample=Image.BILINEAR),
        image.rotate(-5, resample=Image.BILINEAR),
    ]


def run_inference(
    model: nn.Module,
    image: Image.Image,
    metadata: Dict[str, Any],
    device: torch.device,
    use_tta: bool = True,
) -> Dict[str, Any]:
    """
    Executes single-image inference:
      1. If use_tta=True: applies frozen Phase 5 4-view TTA (Original, H-Flip, +5°, -5°)
         and computes the arithmetic mean of softmax probabilities.
         If use_tta=False: runs standard deterministic single-view inference.
      2. Runs forward pass(es) under torch.inference_mode().
      3. Computes normalized softmax probabilities.
      4. Identifies predicted class index and confidence.
      5. Constructs sorted presentation rankings and downloadable JSON payload.
    """
    class_names: List[str] = metadata.get("class_names", CLASS_NAMES)
    img_size: int = metadata.get("image_size", IMAGE_SIZE)

    model.eval()

    if use_tta:
        # Frozen Phase 5 TTA: 4 deterministic views
        views = generate_phase_5_tta_views(image)
        batch_tensors = [preprocess_image(v, image_size=img_size, device=device) for v in views]
        input_batch = torch.cat(batch_tensors, dim=0)  # Shape: (4, 3, img_size, img_size)

        with torch.inference_mode():
            all_logits = model(input_batch)  # Shape: (4, num_classes)
            all_probs = torch.softmax(all_logits, dim=1)  # Shape: (4, num_classes)
            # Arithmetic mean of softmax probabilities
            probabilities = all_probs.mean(dim=0, keepdim=True)  # Shape: (1, num_classes)
            logits = all_logits[0:1]  # Reference logits from primary view
    else:
        # Standard Single-View Inference
        input_tensor = preprocess_image(image, image_size=img_size, device=device)
        with torch.inference_mode():
            logits = model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)

    # Extract predictions
    pred_idx: int = int(probabilities.argmax(dim=1).item())
    confidence: float = float(probabilities[0, pred_idx].item())
    predicted_class: str = class_names[pred_idx]
    predicted_description: str = CLASS_DESCRIPTIONS.get(predicted_class, "Leukemia Subtype")

    # Map probabilities across all classes in authoritative ordering
    class_probs_map: Dict[str, float] = {}
    for i, name in enumerate(class_names):
        class_probs_map[name] = float(probabilities[0, i].item())

    # Sort probabilities descending for presentation
    sorted_probs: List[Tuple[str, float, str]] = []
    for name, p_val in sorted(class_probs_map.items(), key=lambda item: item[1], reverse=True):
        desc = CLASS_DESCRIPTIONS.get(name, "")
        sorted_probs.append((name, p_val, desc))

    # Format download payload matching research requirements
    download_payload: Dict[str, Any] = {
        "predicted_class": predicted_class,
        "predicted_class_description": predicted_description,
        "predicted_class_index": pred_idx,
        "confidence": round(confidence, 6),
        "confidence_formatted": f"{confidence * 100:.2f}%",
        "class_probabilities": {k: round(v, 6) for k, v in class_probs_map.items()},
        "inference_strategy": "Phase 5 Frozen 4-View TTA (Mean Softmax)" if use_tta else "Single-View Deterministic",
        "tta_enabled": use_tta,
        "model_checkpoint": metadata.get("checkpoint_name", "best_checkpoint.pt"),
        "training_strategy": metadata.get("strategy", "Decision-Boundary Refinement (Exp C)"),
        "training_run_id": metadata.get("RUN_ID", "Phase 7 Experiment C"),
        "architecture": metadata.get("model_name", "vit_base_patch16_224"),
        "input_resolution": f"{img_size}x{img_size}",
        "dataset_fingerprint": metadata.get("dataset_fingerprint", DATASET_FINGERPRINT),
        "device": str(device),
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
    }

    return {
        "predicted_class": predicted_class,
        "predicted_class_description": predicted_description,
        "predicted_class_index": pred_idx,
        "confidence": confidence,
        "confidence_percent": f"{confidence * 100:.2f}%",
        "class_probabilities": class_probs_map,
        "sorted_probabilities": sorted_probs,
        "raw_logits": logits.cpu().numpy().tolist()[0],
        "tta_enabled": use_tta,
        "download_payload": download_payload,
    }
