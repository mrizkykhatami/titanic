# Titanic — project latihan Kaggle

Tujuan: menebak penumpang Titanic yang selamat. Project ini sekaligus latihan alur kerja:
**VS Code (per sel) → GitHub → Kaggle**, dibantu Claude Code.

## Struktur
```
titanic/
├── CLAUDE.md              # aturan & konteks untuk Claude Code
├── configs/baseline.yaml  # pengaturan eksperimen
├── src/                   # kode inti (dipakai notebook & Kaggle)
├── notebooks/01_eda.ipynb # eksplorasi per sel
├── kernels/train/         # notebook yang dikirim ke Kaggle
├── scripts/               # alat bantu (kaggle_chain.py)
├── data/                  # (tidak di-commit) data lomba
└── outputs/               # (tidak di-commit) hasil eksperimen
```

## Jalankan pertama kali (di laptop)
```powershell
conda activate kaggle
cd "$HOME\kaggle\titanic"
kaggle competitions download -c titanic -p data
Expand-Archive data\titanic.zip -DestinationPath data -Force
python -m src.train
```
Hasilnya: skor CV di terminal + `outputs/baseline/submission.csv` (nama folder = `name` di config).

## Submit
```powershell
kaggle competitions submit -c titanic -f outputs\baseline\submission.csv -m "baseline hgb"
kaggle competitions submissions -c titanic
```

## Notebook
Buka `notebooks/01_eda.ipynb` → **Select Kernel**:
- **Python (kaggle)** untuk jalan di laptop, atau
- **Colab → New Colab Server** untuk jalan di server Google (repo GitHub harus sudah di-push).

## Training di server Kaggle
1. Ganti `KAGGLE_USERNAME` di `kernels/train/kernel-metadata.json` dengan username Kaggle-mu.
2. `kaggle kernels push -p kernels/train`
3. `kaggle kernels status <username>/titanic-train`
