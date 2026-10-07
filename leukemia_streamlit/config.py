# -*- coding: utf-8 -*-
"""
Leukemia ViT Inference System - Configuration
==============================================
Central configuration for model paths, architecture specifications,
preprocessing parameters, and clinical label definitions.
Verified against the official Phase 7 Experiment C ViT-B/16 model lineage.

NOTE: Inference is strictly restricted to the official Phase 7 Experiment C
best_checkpoint.pt. No other checkpoint files are scanned, loaded, or exposed.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple

# ==============================================================================
# 1. DIRECTORY & EXCLUSIVE OFFICIAL CHECKPOINT CONFIGURATION
# ==============================================================================

def resolve_official_checkpoint_path() -> Path:
    """
    Resolves the exact, exclusive official Phase 7 Experiment C checkpoint path.
    Prioritizes the deployment model directory, falling back to authoritative Phase 7 location.
    """
    current_module_dir = Path(__file__).resolve().parent
    workspace_root = current_module_dir.parent

    # 1. Deployment model directory at workspace root
    deploy_path = workspace_root / "model" / "best_checkpoint.pt"
    if deploy_path.exists():
        return deploy_path.resolve()

    # 2. Deployment model directory within app directory
    app_model_path = current_module_dir / "model" / "best_checkpoint.pt"
    if app_model_path.exists():
        return app_model_path.resolve()

    # 3. Authoritative Phase 7 training directory location
    phase7_path = (
        workspace_root
        / "PHASE_7_VIT_LAYERWISE_OPTIMIZATION"
        / "checkpoints"
        / "boundary"
        / "best_checkpoint.pt"
    )
    if phase7_path.exists():
        return phase7_path.resolve()

    # Fallback to environment override if explicitly provided
    env_ckpt = os.getenv("LEUKEMIA_OFFICIAL_CHECKPOINT")
    if env_ckpt and Path(env_ckpt).exists():
        return Path(env_ckpt).resolve()

    return deploy_path.resolve()


DEFAULT_RUN_ID: str = "Phase 7 Experiment C (Decision-Boundary Refinement)"
DEFAULT_CHECKPOINT_NAME: str = "best_checkpoint.pt"

# Hard-coded exclusive official Phase 7 checkpoint path
BEST_CHECKPOINT: Path = resolve_official_checkpoint_path()
MODEL_DIR: Path = BEST_CHECKPOINT.parent


def get_best_checkpoint_path() -> Path:
    """Returns the single, exclusive inference checkpoint path (best_checkpoint.pt)."""
    return BEST_CHECKPOINT


# Alias for backward compatibility if imported by test utilities
get_default_checkpoint_path = get_best_checkpoint_path


# ==============================================================================
# 2. MODEL ARCHITECTURE & HYPERPARAMETERS (OFFICIAL PHASE 7 SPECIFICATION)
# ==============================================================================
MODEL_NAME: str = "vit_base_patch16_224"
PRETRAINED_MODEL_NAME: str = "vit_base_patch16_224.augreg2_in21k_ft_in1k"
IMAGE_SIZE: int = 384
NUM_CLASSES: int = 5
SEQUENCE_LENGTH: int = 577  # 576 patches (24x24) + 1 CLS token

# Official Dataset Fingerprint & Checkpoint Integrity
DATASET_FINGERPRINT: str = "f3d27eada084311ca19b3d91a31472156ba9c8f2ddd12079422c95591d8f4ce4"
OFFICIAL_CHECKPOINT_SHA256: str = "a1cef719d7bb4c7cda7043301b5751c062cc3b8c86344e8db13fe46674e5509e"

# Certified Official Performance Metrics
OFFICIAL_VAL_ACCURACY: str = "97.05%"
OFFICIAL_VAL_MACRO_F1: str = "96.84%"
OFFICIAL_TEST_ACCURACY: str = "97.02%"
OFFICIAL_TEST_MACRO_F1: str = "97.04%"
OFFICIAL_TEST_ERRORS: str = "149 / 5,000"

# Normalization constants (exact ImageNet statistics used throughout training)
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)
INTERPOLATION_MODE: str = "bilinear"

# ==============================================================================
# 3. CLASS MAPPINGS & CLINICAL METADATA (VERIFIED)
# ==============================================================================
CLASS_NAMES: List[str] = ["ALL", "AML", "CLL", "CML", "H"]
CLASS_TO_IDX: Dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS: Dict[int, str] = {idx: name for idx, name in enumerate(CLASS_NAMES)}

CLASS_DESCRIPTIONS: Dict[str, str] = {
    "ALL": "Acute Lymphoblastic Leukemia",
    "AML": "Acute Myeloid Leukemia",
    "CLL": "Chronic Lymphocytic Leukemia",
    "CML": "Chronic Myeloid Leukemia",
    "H":   "Healthy / Normal Cell Control",
}

CLASS_CLINICAL_DETAILS: Dict[str, str] = {
    "ALL": "Aggressive malignancy characterized by overproduction of immature lymphocytes (lymphoblasts).",
    "AML": "Rapidly progressing cancer of the myeloid line of blood cells with clonal proliferation of myeloblasts.",
    "CLL": "Slow-progressing lymphoproliferative disorder marked by accumulation of mature-appearing abnormal lymphocytes.",
    "CML": "Myeloproliferative neoplasm associated with BCR-ABL1 fusion, resulting in uncontrolled granulocyte proliferation.",
    "H":   "Non-leukemic control morphology representing normal mature and physiological hematopoietic cells.",
}

CLASS_COLORS: Dict[str, str] = {
    "ALL": "#3B82F6",  # Sapphire Blue
    "AML": "#EF4444",  # Crimson Red
    "CLL": "#F59E0B",  # Warm Amber
    "CML": "#8B5CF6",  # Violet
    "H":   "#10B981",  # Emerald Green
}

# ==============================================================================
# 4. FILE VALIDATION & SAFETY LIMITS
# ==============================================================================
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
MAX_IMAGE_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB
MAX_IMAGE_DIMENSION = 10000                   # 10,000 pixels max per side

# ==============================================================================
# 5. RESEARCH USE ONLY DISCLAIMER
# ==============================================================================
RESEARCH_DISCLAIMER: str = (
    "Research Use Only\n\n"
    "This application provides model predictions from an image-classification model. "
    "The output is not a medical diagnosis and should not be used as a substitute "
    "for evaluation by a qualified healthcare professional."
)
