# -*- coding: utf-8 -*-
"""
Leukemia ViT Inference System - Streamlit Web Application
=========================================================
Clinical research dashboard for leukemia single-cell classification
using the official Vision Transformer (ViT-B/16) model lineage.

Usage:
    streamlit run app.py
"""

import runpy
import sys
from pathlib import Path

# Add repository root and package paths
repo_root = Path(__file__).resolve().parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

app_path = repo_root / "leukemia_streamlit" / "app.py"
runpy.run_path(str(app_path), run_name="__main__")
