# -*- coding: utf-8 -*-
"""
Leukemia ViT Inference System - Preprocessing Pipeline
======================================================
Implements deterministic image validation and preprocessing strictly matching
the validation/evaluation pipeline in vit_ddp_worker.py.
"""

import io
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
import torchvision.transforms as T
from PIL import Image, ImageOps, UnidentifiedImageError

from .config import (
    ALLOWED_EXTENSIONS,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MAX_IMAGE_DIMENSION,
    MAX_IMAGE_FILE_SIZE_BYTES,
)


def get_inference_transforms(image_size: int = IMAGE_SIZE) -> T.Compose:
    """
    Returns the exact deterministic evaluation transform pipeline used during
    training in vit_ddp_worker.py:
        1. Resize to (image_size, image_size) with BILINEAR interpolation
        2. ToTensor (scales pixel intensities from [0, 255] to [0.0, 1.0])
        3. Normalize with ImageNet mean and std
    """
    return T.Compose([
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def validate_and_load_image(
    image_source: Union[io.BytesIO, bytes, str, Path],
    filename: Optional[str] = None,
) -> Tuple[bool, Optional[Image.Image], Dict[str, Union[str, int, float]], Optional[str]]:
    """
    Thoroughly validates an uploaded image or file path:
      - Validates file size
      - Verifies file extension / format
      - Protects against malformed / corrupted images
      - Converts any color mode (Grayscale, RGBA, CMYK, Palette) into standard RGB
      - Enforces maximum dimension safety limits

    Returns:
        (is_valid, pil_rgb_image, metadata_dict, error_message)
    """
    meta: Dict[str, Union[str, int, float]] = {}

    try:
        # Determine raw bytes and size
        if isinstance(image_source, (str, Path)):
            p = Path(image_source)
            if not p.is_file():
                return False, None, meta, f"Image file not found: {p}"
            raw_bytes = p.read_bytes()
            if filename is None:
                filename = p.name
        elif isinstance(image_source, io.BytesIO):
            image_source.seek(0)
            raw_bytes = image_source.read()
            image_source.seek(0)
        elif isinstance(image_source, bytes):
            raw_bytes = image_source
        else:
            # Streamlit UploadedFile duck-typing
            if hasattr(image_source, "read") and callable(image_source.read):
                if hasattr(image_source, "seek"):
                    image_source.seek(0)
                raw_bytes = image_source.read()
                if hasattr(image_source, "seek"):
                    image_source.seek(0)
                if filename is None and hasattr(image_source, "name"):
                    filename = getattr(image_source, "name")
            else:
                return False, None, meta, f"Unsupported image source type: {type(image_source)}"

        file_size_bytes = len(raw_bytes)
        meta["file_size_bytes"] = file_size_bytes
        meta["file_size_kb"] = round(file_size_bytes / 1024, 2)
        meta["filename"] = filename or "uploaded_image"

        if file_size_bytes == 0:
            return False, None, meta, "The uploaded file is empty (0 bytes)."

        if file_size_bytes > MAX_IMAGE_FILE_SIZE_BYTES:
            max_mb = MAX_IMAGE_FILE_SIZE_BYTES / (1024 * 1024)
            return (
                False,
                None,
                meta,
                f"File size ({meta['file_size_kb']} KB) exceeds maximum limit of {max_mb:.0f} MB.",
            )

        # Check extension if filename is known
        if filename:
            ext = Path(filename).suffix.lower()
            if ext and ext not in ALLOWED_EXTENSIONS:
                return (
                    False,
                    None,
                    meta,
                    f"Unsupported format '{ext}'. Allowed image formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
                )

        # First pass: verify image integrity
        try:
            with Image.open(io.BytesIO(raw_bytes)) as test_img:
                test_img.verify()
        except Exception as verify_err:
            return (
                False,
                None,
                meta,
                f"Corrupted or invalid image file. Verification failed: {verify_err}",
            )

        # Second pass: open image for actual decoding & manipulation
        img = Image.open(io.BytesIO(raw_bytes))
        meta["original_format"] = img.format or "UNKNOWN"
        meta["original_mode"] = img.mode
        meta["original_width"] = img.width
        meta["original_height"] = img.height

        # Protect against decompression bomb / memory exhaustion
        if img.width > MAX_IMAGE_DIMENSION or img.height > MAX_IMAGE_DIMENSION:
            return (
                False,
                None,
                meta,
                f"Image dimensions ({img.width}x{img.height}) exceed maximum allowed dimension ({MAX_IMAGE_DIMENSION}px).",
            )

        # Auto-orient based on EXIF tags (e.g. mobile/microscope camera orientations)
        try:
            img = ImageOps.exif_transpose(img)
        except Exception:
            pass

        # Robust color conversion: convert to standard 3-channel RGB
        if img.mode != "RGB":
            if img.mode in ("RGBA", "LA", "P"):
                # Composite with white background if transparency exists
                rgb_canvas = Image.new("RGB", img.size, (255, 255, 255))
                if img.mode == "P":
                    img = img.convert("RGBA")
                if "A" in img.mode:
                    rgb_canvas.paste(img, mask=img.split()[-1])
                    img = rgb_canvas
                else:
                    img = img.convert("RGB")
            else:
                img = img.convert("RGB")

        meta["rgb_channels"] = 3
        meta["final_width"] = img.width
        meta["final_height"] = img.height

        return True, img, meta, None

    except UnidentifiedImageError:
        return (
            False,
            None,
            meta,
            "Unable to identify image format. Please upload a valid PNG, JPG, JPEG, or WEBP image.",
        )
    except Exception as e:
        return False, None, meta, f"Unexpected error while processing image: {str(e)}"


def preprocess_image(
    image: Image.Image,
    image_size: int = IMAGE_SIZE,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Preprocesses a validated PIL RGB image into a 4D batch tensor (1, 3, H, W)
    ready for ViT model inference.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    transform = get_inference_transforms(image_size)
    tensor = transform(image)  # Shape: (3, image_size, image_size)
    tensor = tensor.unsqueeze(0)  # Shape: (1, 3, image_size, image_size)

    if device is not None:
        tensor = tensor.to(device, non_blocking=(device.type == "cuda"))

    return tensor
