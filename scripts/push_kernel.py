"""
push_kernel.py — rapikan kernels/train/train.ipynb lalu push ke Kaggle.

Jalankan dari laptop:
    conda run -n kaggle python scripts/push_kernel.py
    conda run -n kaggle python scripts/push_kernel.py --dry-run   # rapikan saja, tanpa push

Yang dilakukan:
  1. kernelspec = python3. Kernel "kaggle" (conda env lokal) tidak ada di server Kaggle
     -> run gagal dengan NoSuchKernel.
  2. Tambah `id` ke sel yang belum punya (nbformat >= 4.5) -> MissingIDFieldWarning hilang.
  3. `kaggle kernels push -p kernels/train`.
"""
import argparse
import json
import subprocess
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNEL_DIR = ROOT / "kernels" / "train"
NOTEBOOK = KERNEL_DIR / "train.ipynb"
KERNELSPEC = {"display_name": "Python 3", "language": "python", "name": "python3"}


def fix_notebook(path: Path) -> list[str]:
    nb = json.loads(path.read_text(encoding="utf-8"))
    changes = []

    if nb.get("metadata", {}).get("kernelspec") != KERNELSPEC:
        nb.setdefault("metadata", {})["kernelspec"] = KERNELSPEC
        changes.append("kernelspec -> python3")

    if (nb.get("nbformat"), nb.get("nbformat_minor", 0)) < (4, 5):
        nb["nbformat"], nb["nbformat_minor"] = 4, 5  # field `id` baru ada sejak nbformat 4.5
        changes.append("nbformat -> 4.5")

    used = {c["id"] for c in nb["cells"] if c.get("id")}
    added = 0
    for cell in nb["cells"]:
        if not cell.get("id"):
            new_id = uuid.uuid4().hex[:8]
            while new_id in used:
                new_id = uuid.uuid4().hex[:8]
            cell["id"] = new_id
            used.add(new_id)
            added += 1
    if added:
        changes.append(f"id ditambahkan ke {added} sel")

    if changes:
        # newline="\n": di Windows tetap LF (sama seperti file di repo)
        path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    return changes


def main():
    p = argparse.ArgumentParser(description="Rapikan train.ipynb lalu push ke Kaggle")
    p.add_argument("--dry-run", action="store_true", help="rapikan notebook saja, tanpa push")
    a = p.parse_args()

    changes = fix_notebook(NOTEBOOK)
    print(f"[notebook] {', '.join(changes) if changes else 'sudah rapi, tidak ada perubahan'}")
    if changes:
        print("[notebook] catatan: commit perubahan ini supaya repo sama dengan yang di-push")

    if a.dry_run:
        print("[push] --dry-run: dilewati")
        return
    r = subprocess.run(["kaggle", "kernels", "push", "-p", str(KERNEL_DIR)])
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
