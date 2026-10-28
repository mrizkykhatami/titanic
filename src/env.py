"""Deteksi "sedang jalan di mana" + path standar.

ENV bernilai salah satu: "kaggle" | "colab" | "local".
- kaggle : notebook di server Kaggle
- colab  : runtime Colab (di browser ATAU lewat ekstensi Colab di VS Code)
- local  : laptop kamu (kernel "Python (kaggle)")
"""
from __future__ import annotations

import os
from pathlib import Path


def detect_env() -> str:
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
        return "kaggle"
    try:
        import google.colab  # noqa: F401
        return "colab"
    except ImportError:
        pass
    if "COLAB_RELEASE_TAG" in os.environ:
        return "colab"
    return "local"


ENV = detect_env()
ROOT = Path(__file__).resolve().parents[1]          # folder repo (titanic/)
DATA_DIR = ROOT / "data"                           # data lokal (tidak di-commit)
OUTPUT_DIR = Path("/kaggle/working") if ENV == "kaggle" else ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
