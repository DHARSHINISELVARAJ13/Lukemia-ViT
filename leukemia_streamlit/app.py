# -*- coding: utf-8 -*-
"""
Leukemia ViT Inference System - Streamlit Web Application
=========================================================
Production-ready clinical research dashboard for leukemia cell classification
using the official Vision Transformer (ViT-B/16) model lineage.

RESTRICTION:
The application uses exclusively the official Phase 7 Experiment C best_checkpoint.pt.
No checkpoint selection, dropdowns, or alternate checkpoint files are permitted.

Input Modes:
  1. Standard file upload (drag & drop / file browser)
  2. Browser-side clipboard image paste (Ctrl + V & Clipboard API)
Both input modes are presented side-by-side and automatically trigger inference.
"""

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
from PIL import Image

# Ensure modules in leukemia_streamlit can be imported cleanly
current_file = Path(__file__).resolve()
app_dir = current_file.parent
if str(app_dir.parent) not in sys.path:
    sys.path.insert(0, str(app_dir.parent))
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

from leukemia_streamlit.clipboard_component import (
    clipboard_paste_box,
    decode_clipboard_image,
)
from leukemia_streamlit.config import (
    ALLOWED_EXTENSIONS,
    BEST_CHECKPOINT,
    CLASS_CLINICAL_DETAILS,
    CLASS_COLORS,
    CLASS_DESCRIPTIONS,
    CLASS_NAMES,
    DATASET_FINGERPRINT,
    DEFAULT_CHECKPOINT_NAME,
    DEFAULT_RUN_ID,
    IMAGE_SIZE,
    MODEL_DIR,
    MODEL_NAME,
    NUM_CLASSES,
    OFFICIAL_CHECKPOINT_SHA256,
    OFFICIAL_TEST_ACCURACY,
    OFFICIAL_TEST_ERRORS,
    OFFICIAL_TEST_MACRO_F1,
    OFFICIAL_VAL_ACCURACY,
    OFFICIAL_VAL_MACRO_F1,
    PRETRAINED_MODEL_NAME,
    RESEARCH_DISCLAIMER,
    SEQUENCE_LENGTH,
)
from leukemia_streamlit.inference import run_inference
from leukemia_streamlit.model_loader import (
    load_model_from_checkpoint,
    select_inference_device,
    verify_model_pipeline,
)
from leukemia_streamlit.preprocessing import validate_and_load_image

# ==============================================================================
# 1. STREAMLIT PAGE CONFIGURATION & MEDICAL CSS THEME (ZERO EMOJIS)
# ==============================================================================
st.set_page_config(
    page_title="Leukemia ViT Classification System",
    page_icon=":material/biotech:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-Aesthetic Medical Research CSS
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Header Container */
    .med-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 24px 32px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.4);
    }
    .med-title {
        color: #f8fafc;
        font-size: 26px;
        font-weight: 700;
        letter-spacing: -0.5px;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .med-subtitle {
        color: #94a3b8;
        font-size: 14px;
        margin-top: 6px;
        margin-bottom: 0;
    }
    .med-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 14px;
    }
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.3px;
        text-transform: uppercase;
    }
    .badge-primary { background: rgba(59, 130, 246, 0.15); border: 1px solid #3b82f6; color: #60a5fa; }
    .badge-purple  { background: rgba(139, 92, 246, 0.15); border: 1px solid #8b5cf6; color: #a78bfa; }
    .badge-success { background: rgba(16, 185, 129, 0.15); border: 1px solid #10b981; color: #34d399; }
    .badge-amber   { background: rgba(245, 158, 11, 0.15); border: 1px solid #f59e0b; color: #fbbf24; }

    /* Disclaimer Card */
    .disclaimer-banner {
        background: #1e1b18;
        border-left: 4px solid #f59e0b;
        border-radius: 6px;
        padding: 12px 18px;
        margin-bottom: 24px;
        font-size: 12.5px;
        color: #d1d5db;
        line-height: 1.5;
    }
    .disclaimer-title {
        font-weight: 700;
        color: #fbbf24;
        display: flex;
        align-items: center;
        gap: 6px;
        margin-bottom: 4px;
        text-transform: uppercase;
        font-size: 11px;
        letter-spacing: 0.5px;
    }

    /* Cards */
    .med-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 16px;
    }
    .card-title {
        font-size: 14px;
        font-weight: 600;
        color: #cbd5e1;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* Prediction Result Hero */
    .prediction-hero {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 2px solid #3b82f6;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    .pred-class-code {
        font-size: 42px;
        font-weight: 800;
        letter-spacing: -1px;
        line-height: 1.1;
        margin-bottom: 4px;
    }
    .pred-class-name {
        font-size: 16px;
        color: #cbd5e1;
        font-weight: 500;
        margin-bottom: 16px;
    }
    .pred-conf-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(59, 130, 246, 0.15);
        border: 1px solid #3b82f6;
        padding: 6px 18px;
        border-radius: 9999px;
        color: #60a5fa;
        font-size: 16px;
        font-weight: 700;
    }
    
    /* Metadata grid in sidebar */
    .meta-row {
        display: flex;
        justify-content: space-between;
        padding: 6px 0;
        border-bottom: 1px solid #283548;
        font-size: 12.5px;
    }
    .meta-label { color: #94a3b8; font-weight: 500; }
    .meta-value { color: #f1f5f9; font-weight: 600; font-family: monospace; font-size: 12px; }
    
    /* Source Pill */
    .source-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 600;
        background: #0f172a;
        border: 1px solid #334155;
        color: #94a3b8;
        margin-top: 10px;
    }
    .source-pill strong {
        color: #60a5fa;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==============================================================================
# 2. CACHED MODEL LOADER (LOADS ONLY OFFICIAL BEST_CHECKPOINT.PT ONCE)
# ==============================================================================
@st.cache_resource(show_spinner=False)
def get_cached_model() -> Tuple[torch.nn.Module, Dict[str, Any], torch.device, str]:
    """
    Loads and caches the trained Vision Transformer model strictly from official best_checkpoint.pt.
    No other checkpoint files are loaded, scanned, or permitted.
    """
    device, device_desc = select_inference_device()
    model, metadata = load_model_from_checkpoint(BEST_CHECKPOINT, device=device)
    return model, metadata, device, device_desc


# ==============================================================================
# 3. SIDEBAR: ARCHITECTURE, RUN & VERIFICATION (NO EMOJIS, NO SELECTOR)
# ==============================================================================
def render_sidebar() -> Tuple[Optional[torch.nn.Module], Optional[Dict[str, Any]], Optional[torch.device], bool, bool]:
    # Checkpoint existence validation
    if not BEST_CHECKPOINT.is_file():
        st.sidebar.error(
            f"**Official model checkpoint not found.**\n\n"
            f"Expected:\n`{BEST_CHECKPOINT}`\n\n"
            f"The application requires `{DEFAULT_CHECKPOINT_NAME}`.",
            icon=":material/error:",
        )
        return None, None, None, False, True

    # Attempt cached load of official best_checkpoint.pt
    model_loaded = False
    model = None
    metadata = {}
    device = None
    device_desc = "Unknown"

    try:
        with st.sidebar:
            with st.spinner("Loading official Phase 7 model checkpoint..."):
                model, metadata, device, device_desc = get_cached_model()
                model_loaded = True
    except Exception as e:
        st.sidebar.error(f"Failed to load model:\n{str(e)}", icon=":material/error:")
        return None, None, None, False, True

    # Run automated pipeline verification on official checkpoint
    is_valid, checks, err_msg = verify_model_pipeline(model, metadata, device)

    # Sidebar Header (Pure SVG Icon, Zero Emojis)
    st.sidebar.markdown(
        """
        <div style="display: flex; align-items: center; gap: 8px; font-weight: 700; font-size: 15px; color: #e2e8f0; margin-bottom: 12px;">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#60a5fa" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="12" cy="12" r="3"></circle>
                <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path>
            </svg>
            <span>OFFICIAL MODEL SPECIFICATION</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown(f"""
    <div class="meta-row"><span class="meta-label">Architecture</span><span class="meta-value">ViT-B/16</span></div>
    <div class="meta-row"><span class="meta-label">Lineage</span><span class="meta-value">Phase 7 Exp C</span></div>
    <div class="meta-row"><span class="meta-label">Strategy</span><span class="meta-value">Boundary Refinement</span></div>
    <div class="meta-row"><span class="meta-label">Checkpoint</span><span class="meta-value">{DEFAULT_CHECKPOINT_NAME}</span></div>
    <div class="meta-row"><span class="meta-label">Input Size</span><span class="meta-value">{IMAGE_SIZE} x {IMAGE_SIZE}</span></div>
    <div class="meta-row"><span class="meta-label">Tokens</span><span class="meta-value">{SEQUENCE_LENGTH} tokens</span></div>
    <div class="meta-row"><span class="meta-label">Val Macro F1</span><span class="meta-value">{OFFICIAL_VAL_MACRO_F1}</span></div>
    <div class="meta-row"><span class="meta-label">Test Macro F1</span><span class="meta-value">{OFFICIAL_TEST_MACRO_F1}</span></div>
    <div class="meta-row"><span class="meta-label">Test Accuracy</span><span class="meta-value">{OFFICIAL_TEST_ACCURACY}</span></div>
    <div class="meta-row"><span class="meta-label">Test Errors</span><span class="meta-value">{OFFICIAL_TEST_ERRORS}</span></div>
    <div class="meta-row"><span class="meta-label">Device</span><span class="meta-value">{device_desc}</span></div>
    <div class="meta-row"><span class="meta-label">Status</span><span class="meta-value" style="color: #34d399;">CERTIFIED</span></div>
    """, unsafe_allow_html=True)

    # Inference Configuration Section
    st.sidebar.markdown(
        """
        <div style="display: flex; align-items: center; gap: 8px; font-weight: 700; font-size: 14px; color: #e2e8f0; margin-top: 18px; margin-bottom: 8px;">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#60a5fa" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"></path>
            </svg>
            <span>INFERENCE STRATEGY</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    use_tta = st.sidebar.checkbox(
        "Enable Phase 5 4-View TTA",
        value=True,
        help="Applies the frozen Phase 5 4-view TTA (Original, Horizontal Flip, +5° Rotation, -5° Rotation with arithmetic mean softmax aggregation).",
    )

    # Verification Accordion
    with st.sidebar.expander("System Verification Checklist", expanded=False):
        if is_valid:
            st.success("Model verification: PASSED", icon=":material/verified:")
            for c in checks:
                st.markdown(
                    f"""
                    <div style="display: flex; align-items: center; gap: 8px; margin: 4px 0; font-size: 13px;">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                            <polyline points="20 6 9 17 4 12"></polyline>
                        </svg>
                        <span>{c}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.error(f"Model verification: FAILED\n{err_msg}", icon=":material/error:")
            for c in checks:
                st.markdown(f"- [FAIL] {c}")

    # Clinical Label Reference Accordion
    with st.sidebar.expander("Leukemia Class Reference", expanded=False):
        for code in CLASS_NAMES:
            color = CLASS_COLORS.get(code, "#3b82f6")
            st.markdown(
                f"<div style='border-left: 3px solid {color}; padding-left: 8px; margin-bottom: 12px;'>"
                f"<strong style='color: {color};'>{code}</strong>: {CLASS_DESCRIPTIONS.get(code, '')}<br>"
                f"<span style='font-size: 11px; color: #94a3b8;'>{CLASS_CLINICAL_DETAILS.get(code, '')}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    return model, metadata, device, model_loaded, use_tta


# ==============================================================================
# 4. MAIN USER INTERFACE
# ==============================================================================
def main():
    # Render Sidebar and load model
    model, metadata, device, model_loaded, use_tta = render_sidebar()

    # Top Header Banner (Pure SVG Medical Icon, Zero Emojis)
    st.markdown(
        f"""
        <div class="med-header">
            <h1 class="med-title">
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#60a5fa" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M6 18h8"></path>
                    <path d="M3 22h18"></path>
                    <path d="M14 22a7 7 0 1 0 0-14h-1"></path>
                    <path d="M9 14h2"></path>
                    <path d="M9 12a2 2 0 0 1-2-2V6h6v4a2 2 0 0 1-2 2Z"></path>
                    <path d="M12 6V3a1 1 0 0 0-1-1H9a1 1 0 0 0-1 1v3"></path>
                </svg>
                <span>LEUKEMIA CELL CLASSIFICATION</span>
            </h1>
            <p class="med-subtitle">
                Official Vision Transformer (ViT-B/16) Inference System for Microscopic Cell Image Analysis
            </p>
            <div class="med-badges">
                <span class="badge badge-primary">Model: ViT-B/16 ({MODEL_NAME})</span>
                <span class="badge badge-purple">Lineage: Phase 7 Exp C</span>
                <span class="badge badge-success">5 Classes (ALL, AML, CLL, CML, H)</span>
                <span class="badge badge-amber">Resolution: {IMAGE_SIZE}x{IMAGE_SIZE} ({SEQUENCE_LENGTH} tokens)</span>
                <span class="badge badge-primary" style="background: rgba(16, 185, 129, 0.2); border: 1px solid #10b981; color: #34d399;">Official Test: {OFFICIAL_TEST_ACCURACY} Acc | {OFFICIAL_TEST_MACRO_F1} F1</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Research Use Only Disclaimer (SVG Warning Triangle, Zero Emojis)
    st.markdown(
        f"""
        <div class="disclaimer-banner">
            <div class="disclaimer-title">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#fbbf24" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"></path>
                    <line x1="12" y1="9" x2="12" y2="13"></line>
                    <line x1="12" y1="17" x2="12.01" y2="17"></line>
                </svg>
                <span>Research Use Only</span>
            </div>
            {RESEARCH_DISCLAIMER.replace(chr(10) + chr(10), '<br>')}
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not model_loaded or model is None:
        st.error(
            f"**Official model checkpoint not found.**\n\n"
            f"Expected:\n`{BEST_CHECKPOINT}`\n\n"
            f"The application requires `{DEFAULT_CHECKPOINT_NAME}`.",
            icon=":material/error:",
        )
        return

    # ==========================================================================
    # SIDE-BY-SIDE IMAGE INPUT SECTION (UPLOAD FILE & PASTE FROM CLIPBOARD)
    # ==========================================================================
    st.markdown(
        """
        <div class="card-title">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#60a5fa" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                <circle cx="8.5" cy="8.5" r="1.5"></circle>
                <polyline points="21 15 16 10 5 21"></polyline>
            </svg>
            <span>Image Input</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_upload, col_paste = st.columns(2, gap="medium")

    with col_upload:
        st.markdown(
            """
            <div style="font-size: 13px; font-weight: 600; color: #cbd5e1; margin-bottom: 8px; display: flex; align-items: center; gap: 8px;">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#60a5fa" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                    <polyline points="17 8 12 3 7 8"></polyline>
                    <line x1="12" y1="3" x2="12" y2="15"></line>
                </svg>
                <span>Upload File</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader(
            "Upload cell image (PNG, JPG, JPEG, WEBP):",
            type=["png", "jpg", "jpeg", "webp"],
            label_visibility="collapsed",
            help="Upload a microscopic blood or bone marrow cell smear image.",
        )

    with col_paste:
        st.markdown(
            """
            <div style="font-size: 13px; font-weight: 600; color: #cbd5e1; margin-bottom: 8px; display: flex; align-items: center; gap: 8px;">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#60a5fa" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"></path>
                    <rect x="8" y="2" width="8" height="4" rx="1" ry="1"></rect>
                </svg>
                <span>Paste from Clipboard</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        clipboard_payload = clipboard_paste_box(key="clipboard_input")

    # --------------------------------------------------------------------------
    # INPUT SOURCE ARBITRATION (DETERMINE ACTIVE IMAGE & PRESERVE RECENCY)
    # --------------------------------------------------------------------------
    upload_event = False
    current_upload_id = None
    upload_bytes = None
    upload_name = None

    if uploaded_file is not None:
        upload_bytes = uploaded_file.getvalue()
        upload_name = uploaded_file.name
        current_upload_id = f"upload_{hashlib.sha256(upload_bytes).hexdigest()[:16]}"
        if current_upload_id != st.session_state.get("last_uploaded_file_id"):
            st.session_state["last_uploaded_file_id"] = current_upload_id
            upload_event = True
    else:
        if st.session_state.get("last_uploaded_file_id") is not None:
            st.session_state["last_uploaded_file_id"] = None
            if st.session_state.get("active_source") == "Uploaded File":
                st.session_state["active_image"] = None
                st.session_state["active_image_id"] = None
                st.session_state["active_source"] = "None"
                st.session_state["active_filename"] = "cell_image"
                st.session_state["image_error"] = None
                st.session_state.pop("prediction_result", None)
                st.session_state.pop("prediction_error", None)
                st.session_state.pop("last_predicted_image_id", None)

    # Process clipboard input payload safely without slicing non-string objects
    paste_event = False
    paste_raw_bytes = None
    paste_filename = "clipboard_image.png"
    paste_error = None
    current_paste_id = None
    paste_ts = None

    if clipboard_payload is not None:
        if isinstance(clipboard_payload, dict):
            status = clipboard_payload.get("status")
            paste_ts = clipboard_payload.get("timestamp")
            if status == "error":
                paste_error = clipboard_payload.get("message", "Clipboard image could not be accessed.")
            elif status == "success" or "data_url" in clipboard_payload or "image" in clipboard_payload:
                raw_bytes, fname, err = decode_clipboard_image(clipboard_payload)
                if err:
                    paste_error = err
                elif raw_bytes:
                    paste_raw_bytes = raw_bytes
                    paste_filename = fname or "clipboard_image.png"
                    current_paste_id = f"paste_{hashlib.sha256(raw_bytes).hexdigest()[:16]}"
                else:
                    paste_error = "The clipboard did not contain valid image data."
            else:
                paste_error = "Empty or unrecognized clipboard payload received."
        elif isinstance(clipboard_payload, (str, bytes)):
            raw_bytes, fname, err = decode_clipboard_image(clipboard_payload)
            if err:
                paste_error = err
            elif raw_bytes:
                paste_raw_bytes = raw_bytes
                paste_filename = fname or "clipboard_image.png"
                current_paste_id = f"paste_{hashlib.sha256(raw_bytes).hexdigest()[:16]}"
        else:
            paste_error = f"Unsupported clipboard payload type: {type(clipboard_payload).__name__}"

        # Determine whether a new paste event occurred
        is_new_paste = False
        if paste_ts is not None:
            if paste_ts != st.session_state.get("last_paste_ts"):
                st.session_state["last_paste_ts"] = paste_ts
                is_new_paste = True
        elif current_paste_id is not None:
            if current_paste_id != st.session_state.get("last_paste_id"):
                is_new_paste = True
        elif paste_error is not None:
            if paste_error != st.session_state.get("last_paste_error"):
                is_new_paste = True

        if is_new_paste:
            paste_event = True
            st.session_state["last_paste_id"] = current_paste_id
            st.session_state["last_paste_error"] = paste_error

    # Recency arbitration:
    # 1. Clipboard paste has priority if a new paste event just arrived
    # 2. Otherwise a new file upload takes priority
    if paste_event:
        if paste_error:
            st.session_state["image_error"] = paste_error
            st.session_state["active_image"] = None
            st.session_state["active_image_id"] = None
            st.session_state["active_source"] = "Clipboard Paste"
            st.session_state["active_filename"] = paste_filename
            st.session_state.pop("prediction_result", None)
            st.session_state.pop("prediction_error", None)
            st.session_state.pop("last_predicted_image_id", None)
        elif paste_raw_bytes is not None:
            is_valid, pil_img, meta, load_err = validate_and_load_image(
                paste_raw_bytes, filename=paste_filename
            )
            if is_valid and pil_img is not None:
                st.session_state["active_image"] = pil_img
                st.session_state["active_image_id"] = current_paste_id
                st.session_state["active_source"] = "Clipboard Paste"
                st.session_state["active_filename"] = paste_filename
                st.session_state["image_error"] = None
            else:
                st.session_state["active_image"] = None
                st.session_state["active_image_id"] = None
                st.session_state["active_source"] = "Clipboard Paste"
                st.session_state["active_filename"] = paste_filename
                st.session_state["image_error"] = load_err or "Invalid clipboard image format."
                st.session_state.pop("prediction_result", None)
                st.session_state.pop("prediction_error", None)
                st.session_state.pop("last_predicted_image_id", None)
    elif upload_event and upload_bytes is not None:
        is_valid, pil_img, meta, load_err = validate_and_load_image(
            upload_bytes, filename=upload_name
        )
        if is_valid and pil_img is not None:
            st.session_state["active_image"] = pil_img
            st.session_state["active_image_id"] = current_upload_id
            st.session_state["active_source"] = "Uploaded File"
            st.session_state["active_filename"] = upload_name
            st.session_state["image_error"] = None
        else:
            st.session_state["active_image"] = None
            st.session_state["active_image_id"] = None
            st.session_state["active_source"] = "Uploaded File"
            st.session_state["active_filename"] = upload_name
            st.session_state["image_error"] = load_err or "Invalid uploaded file."
            st.session_state.pop("prediction_result", None)
            st.session_state.pop("prediction_error", None)
            st.session_state.pop("last_predicted_image_id", None)

    if st.session_state.get("image_error"):
        st.error(f"Image validation failed: {st.session_state['image_error']}", icon=":material/error:")

    # --------------------------------------------------------------------------
    # AUTOMATIC INFERENCE TRIGGER
    # --------------------------------------------------------------------------
    active_image = st.session_state.get("active_image")
    active_image_id = st.session_state.get("active_image_id")
    active_source = st.session_state.get("active_source", "Unknown")
    active_filename = st.session_state.get("active_filename", "cell_image")

    if active_image is not None and active_image_id is not None:
        # Re-run inference if image changed or TTA setting changed
        tta_key = f"tta_{use_tta}"
        if (
            st.session_state.get("last_predicted_image_id") != active_image_id
            or st.session_state.get("last_predicted_tta_setting") != tta_key
            or "prediction_result" not in st.session_state
        ):
            try:
                with st.spinner("Executing official ViT-B/16 inference pipeline..."):
                    result = run_inference(model, active_image, metadata, device, use_tta=use_tta)
                st.session_state["prediction_result"] = result
                st.session_state["last_predicted_image_id"] = active_image_id
                st.session_state["last_predicted_tta_setting"] = tta_key
                st.session_state["prediction_error"] = None
            except Exception as e:
                st.session_state["prediction_result"] = None
                st.session_state["prediction_error"] = str(e)

    # --------------------------------------------------------------------------
    # DISPLAY: IMAGE PREVIEW & CLASSIFICATION RESULT
    # --------------------------------------------------------------------------
    if active_image is None:
        st.markdown("<div style='margin-top: 16px;'></div>", unsafe_allow_html=True)
        st.info(
            "Upload a cell image or paste an image from your clipboard above. Prediction will run automatically.",
            icon=":material/info:",
        )
    else:
        st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)
        col_preview, col_result = st.columns([1, 1], gap="large")

        # LEFT COLUMN: Image Preview
        with col_preview:
            st.markdown(
                """
                <div class="card-title">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#60a5fa" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                        <circle cx="8.5" cy="8.5" r="1.5"></circle>
                        <polyline points="21 15 16 10 5 21"></polyline>
                    </svg>
                    <span>Image Preview</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.image(
                active_image,
                caption=active_filename,
                use_container_width=True,
            )

            st.markdown(
                f"""
                <div class="source-pill">
                    <span>Source:</span> <strong>{active_source}</strong>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # RIGHT COLUMN: Classification Result
        with col_result:
            st.markdown(
                """
                <div class="card-title">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#60a5fa" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <line x1="18" y1="20" x2="18" y2="10"></line>
                        <line x1="12" y1="20" x2="12" y2="4"></line>
                        <line x1="6" y1="20" x2="6" y2="14"></line>
                    </svg>
                    <span>Classification Result</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            prediction_result = st.session_state.get("prediction_result")
            prediction_error = st.session_state.get("prediction_error")

            if prediction_error:
                st.error(f"Inference execution failed: {prediction_error}", icon=":material/error:")
            elif prediction_result is not None:
                pred_class = prediction_result["predicted_class"]
                pred_desc = prediction_result["predicted_class_description"]
                confidence_pct = prediction_result["confidence_percent"]
                class_color = CLASS_COLORS.get(pred_class, "#3B82F6")
                is_tta = prediction_result.get("tta_enabled", True)
                strategy_label = "Phase 5 Frozen 4-View TTA" if is_tta else "Single-View Deterministic"

                # Hero Prediction Card
                st.markdown(
                    f"""
                    <div class="prediction-hero" style="border-color: {class_color};">
                        <span style="font-size: 12px; text-transform: uppercase; letter-spacing: 1px; color: #94a3b8; font-weight: 600;">
                            PREDICTED LEUKEMIA CLASS
                        </span>
                        <div class="pred-class-code" style="color: {class_color};">{pred_class}</div>
                        <div class="pred-class-name">{pred_desc}</div>
                        <div style="display: flex; justify-content: center; gap: 8px; flex-wrap: wrap;">
                            <div class="pred-conf-pill" style="border-color: {class_color}; color: {class_color};">
                                <span>Confidence:</span> <strong>{confidence_pct}</strong>
                            </div>
                            <div class="source-pill" style="margin-top: 0; background: rgba(15, 23, 42, 0.8);">
                                <span>Strategy:</span> <strong>{strategy_label}</strong>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Class Probabilities Horizontal Bar Chart (Plotly)
                st.markdown("#### Class Probability Distribution")

                # Format data for Plotly (ascending for horizontal bar chart display)
                sorted_items = list(prediction_result["sorted_probabilities"])
                chart_classes = [item[0] for item in reversed(sorted_items)]
                chart_probs = [item[1] * 100 for item in reversed(sorted_items)]
                chart_labels = [f"{CLASS_DESCRIPTIONS.get(c, c)} ({c})" for c in chart_classes]
                chart_colors = [CLASS_COLORS.get(c, "#3b82f6") for c in chart_classes]
                chart_text = [f"{p:.2f}%" for p in chart_probs]

                fig = go.Figure(
                    go.Bar(
                        x=chart_probs,
                        y=chart_labels,
                        orientation="h",
                        marker=dict(
                            color=chart_colors,
                            line=dict(color="#1e293b", width=1.5),
                        ),
                        text=chart_text,
                        textposition="outside",
                        textfont=dict(color="#f8fafc", size=12, family="Inter"),
                        cliponaxis=False,
                    )
                )

                fig.update_layout(
                    margin=dict(l=10, r=40, t=10, b=30),
                    height=280,
                    xaxis=dict(
                        title="Probability (%)",
                        range=[0, max(max(chart_probs) * 1.15, 10.0)],
                        showgrid=True,
                        gridcolor="#334155",
                        color="#94a3b8",
                        zeroline=False,
                    ),
                    yaxis=dict(
                        showgrid=False,
                        color="#f1f5f9",
                        tickfont=dict(size=12),
                    ),
                    plot_bgcolor="rgba(15, 23, 42, 0.4)",
                    paper_bgcolor="rgba(0, 0, 0, 0)",
                )

                st.plotly_chart(fig, use_container_width=True)

                # Tabular Probability Breakdown
                st.markdown("#### Detailed Probability Breakdown")
                prob_data = []
                for code, prob_val, desc in prediction_result["sorted_probabilities"]:
                    prob_data.append({
                        "Class Code": code,
                        "Leukemia Subtype / Description": desc,
                        "Probability": f"{prob_val * 100:.2f}%",
                        "Raw Score": round(prob_val, 6),
                    })

                prob_df = pd.DataFrame(prob_data)
                st.dataframe(
                    prob_df[["Class Code", "Leukemia Subtype / Description", "Probability"]],
                    use_container_width=True,
                    hide_index=True,
                )

                # Downloadable Prediction Result (JSON)
                st.markdown("---")
                json_payload_str = json.dumps(prediction_result["download_payload"], indent=2)
                st.download_button(
                    label="Download Prediction Result (JSON)",
                    icon=":material/download:",
                    data=json_payload_str,
                    file_name=f"leukemia_prediction_{pred_class}.json",
                    mime="application/json",
                    use_container_width=True,
                )


if __name__ == "__main__":
    main()
