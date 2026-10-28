"""Mencari & memuat data kompetisi, di mana pun notebook dijalankan.

Urutan pencarian:
  1. data/ di repo            (laptop: setelah `kaggle competitions download`)
  2. /kaggle/input/**         (Kaggle: data lomba yang di-"Add Input")
  3. kagglehub (download)     (Colab / laptop tanpa data/; butuh token Kaggle)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .env import DATA_DIR, ENV

COMPETITION = "titanic"


def find_data_dir(competition: str = COMPETITION) -> Path:
    if (DATA_DIR / "train.csv").exists():
        return DATA_DIR

    if ENV == "kaggle":
        hits = sorted(Path("/kaggle/input").rglob("train.csv"))
        for h in hits:
            if competition in str(h):
                return h.parent
        if hits:
            return hits[0].parent

    try:
        import kagglehub

        return Path(kagglehub.competition_download(competition))
    except Exception as e:  # token belum ada / belum join lomba / offline
        raise FileNotFoundError(
            f"Data '{competition}' tidak ditemukan dan gagal di-download ({e}).\n"
            "Cek: (1) sudah klik 'Join Competition' di Kaggle, "
            "(2) token Kaggle ada di ~/.kaggle/access_token atau env KAGGLE_API_TOKEN, "
            "atau (3) download manual: kaggle competitions download -c titanic -p data"
        ) from e


def load(competition: str = COMPETITION) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train, test) sebagai DataFrame."""
    d = find_data_dir(competition)
    train = pd.read_csv(d / "train.csv")
    test = pd.read_csv(d / "test.csv")
    print(f"[data] {d} | train={train.shape} test={test.shape}")
    return train, test
