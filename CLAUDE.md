# CLAUDE.md — catatan project untuk Claude Code

## Tentang user & cara menjelaskan
- User baru belajar data science & Kaggle. Jelaskan perubahan dengan bahasa sederhana,
  Bahasa Indonesia, istilah teknis tetap English. Sertakan alasan singkat ("kenapa").
- OS: Windows 11. Python ada di conda env `kaggle` (kernel Jupyter: "Python (kaggle)").
  Jalankan Python dengan `conda run -n kaggle python ...` kalau env belum aktif.

## Struktur
- `src/env.py`       deteksi tempat jalan (local / colab / kaggle) + path standar
- `src/data.py`      cari & load data (data/ → /kaggle/input → kagglehub)
- `src/features.py`  feature engineering (dipakai sama persis untuk train & test)
- `src/train.py`     CV (StratifiedKFold) + OOF + submission + log eksperimen
- `src/time_budget_train.py`  loop PyTorch sadar-waktu 12 jam + auto-resume (untuk lomba DL nanti)
- `configs/*.yaml`   satu file = satu eksperimen
- `notebooks/`       notebook eksplorasi (per sel)
- `kernels/train/`   notebook + kernel-metadata.json yang di-push ke Kaggle
- `scripts/kaggle_chain.py`  orkestrasi beberapa sesi Kaggle (untuk training > 12 jam)
- `scripts/push_kernel.py`   rapikan train.ipynb (kernelspec python3 + id sel) lalu push ke Kaggle

## Aturan
- Kode yang dipakai berulang → `src/*.py`. Notebook hanya eksplorasi & memanggil fungsi dari `src`.
- JANGAN membaca, menampilkan, atau meng-commit token: `~/.kaggle/access_token`, `kaggle.json`, `.env`.
- `data/` dan `outputs/` tidak di-commit (sudah di .gitignore).
- Validasi selalu pakai CV (default 5-fold, seed 42). Laporkan skor CV, jangan hanya leaderboard.
- Saat mengedit `.ipynb`: ubah hanya sel yang diminta; jangan hapus sel/output lain.
- Setelah mengubah `src/`, ingatkan user menjalankan ulang sel SETUP (atau autoreload sudah aktif).
- Minta konfirmasi user sebelum: `git push`, `kaggle kernels push`, `kaggle competitions submit`.
- Push ke Kaggle SELALU lewat `python scripts/push_kernel.py`, jangan `kaggle kernels push` langsung
  (kernelspec "kaggle" lokal bikin run gagal NoSuchKernel). Kalau script mengubah notebook, commit dulu.

## Perintah yang sering dipakai
- Download data lokal: `kaggle competitions download -c titanic -p data` lalu
  `tar -xf data/titanic.zip -C data` (atau Expand-Archive di PowerShell)
- CV lokal: `python -m src.train` | model lain: `python -m src.train --model rf`
- Push training ke Kaggle: `conda run -n kaggle python scripts/push_kernel.py`
  (cek saja tanpa push: tambah `--dry-run`)
- Cek status: `kaggle kernels status mrizkykhatami/titanic-train`
- Ambil output: `kaggle kernels output mrizkykhatami/titanic-train -p outputs/kaggle`
- Submit: `kaggle competitions submit -c titanic -f outputs/<nama>/submission.csv -m "<pesan>"`
- Lihat skor submission: `kaggle competitions submissions -c titanic`
