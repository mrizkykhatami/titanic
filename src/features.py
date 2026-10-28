"""Feature engineering untuk Titanic.

Feature = informasi yang dipakai model untuk menebak.
Semua transformasi ada di sini supaya train & test diperlakukan sama persis.
"""
from __future__ import annotations

import pandas as pd

NUM_COLS = ["Age", "Fare", "FarePerPerson", "SibSp", "Parch", "FamilySize", "IsAlone", "HasCabin"]
CAT_COLS = ["Pclass", "Sex", "Embarked", "Title", "Deck"]
COMMON_TITLES = {"Mr", "Mrs", "Miss", "Master"}


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Gelar dari nama: "Braund, Mr. Owen Harris" -> "Mr"
    title = df["Name"].str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()
    title = title.replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    df["Title"] = title.where(title.isin(COMMON_TITLES), "Rare")

    # Ukuran keluarga di kapal
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    df["FarePerPerson"] = df["Fare"] / df["FamilySize"]

    # Kabin: punya data kabin? dek berapa (huruf pertama)?
    df["HasCabin"] = df["Cabin"].notna().astype(int)
    df["Deck"] = df["Cabin"].str[0].fillna("U")

    # Kolom kategori -> tipe object (aman untuk OneHotEncoder)
    df["Pclass"] = df["Pclass"].astype(str)
    for c in CAT_COLS:
        df[c] = df[c].astype(object)
    return df
