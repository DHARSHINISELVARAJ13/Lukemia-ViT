# -*- coding: utf-8 -*-
"""
Leukemia ViT Inference System - Clipboard Paste Component
==========================================================
Browser-side custom Streamlit component capturing clipboard image data via
the HTML5 Clipboard API and Ctrl+V paste events.
"""

import base64
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import streamlit.components.v1 as components

_frontend_dir = Path(__file__).resolve().parent / "frontend"

_component_func = components.declare_component(
    "clipboard_paste_box",
    path=str(_frontend_dir),
)


def clipboard_paste_box(key: str = "clipboard_paste_widget") -> Optional[Dict[str, Any]]:
    """
    Renders the browser-side clipboard paste zone.
    Returns dictionary payload from JavaScript or None if no action yet.
    """
    return _component_func(key=key, default=None)


def decode_clipboard_image(
    payload: Any,
) -> Tuple[Optional[bytes], Optional[str], Optional[str]]:
    """
    Decodes the component payload into raw image bytes.
    Accepts dictionary from JS component, data URL string, raw bytes, or None.

    Returns:
        (raw_image_bytes, filename, error_message)
    """
    if payload is None:
        return None, None, None

    if isinstance(payload, bytes):
        return payload, "clipboard_image.png", None

    if isinstance(payload, str):
        clean_str = payload.strip()
        if not clean_str:
            return None, None, None
        try:
            if "," in clean_str:
                _, encoded = clean_str.split(",", 1)
            else:
                encoded = clean_str
            raw_bytes = base64.b64decode(encoded)
            return raw_bytes, "clipboard_image.png", None
        except Exception as e:
            return None, None, f"Failed to decode clipboard image data: {str(e)}"

    if not isinstance(payload, dict):
        return None, None, f"Unsupported clipboard payload type: {type(payload).__name__}"

    status = payload.get("status")
    if status == "error":
        return None, None, payload.get("message", "Clipboard image could not be accessed.")

    data_url = payload.get("data_url") or payload.get("image") or payload.get("data") or ""
    if not data_url or not isinstance(data_url, str):
        if status == "success":
            return (
                None,
                None,
                "The clipboard payload did not contain valid image data.",
            )
        return None, None, None

    try:
        # Format: data:image/png;base64,<encoded_data>
        if "," in data_url:
            _, encoded = data_url.split(",", 1)
        else:
            encoded = data_url
        raw_bytes = base64.b64decode(encoded)
        filename = payload.get("filename") or "clipboard_image.png"
        return raw_bytes, filename, None
    except Exception as e:
        return (
            None,
            None,
            f"Failed to decode clipboard image data: {str(e)}",
        )
