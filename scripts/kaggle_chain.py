"""
kaggle_chain.py — orchestrator "push -> tunggu -> cek status -> push lagi" untuk Kaggle.

Jalankan dari laptop / GitHub Actions (butuh `pip install kaggle` + token di ~/.kaggle/).
Pola PING-PONG dua kernel (A dan B): run A menulis last.pt ke output-nya; run B meng-attach
output A sebagai input (kernel_sources) lalu time_budget_train.find_latest_checkpoint()
otomatis menemukannya di /kaggle/input/**/last.pt. Begitu seterusnya A <- B <- A ...

Struktur folder lokal:
    kernels/A/  train.ipynb  kernel-metadata.json   (id: USER/comp-train-a)
    kernels/B/  train.ipynb  kernel-metadata.json   (id: USER/comp-train-b)
    (isi notebook A dan B identik; cukup copy)

Catatan:
  - Setiap sesi memakan weekly GPU quota; 30 jam/minggu ≈ 2-3 sesi 11.5 jam.
  - Kaggle Secrets (diatur via UI) tidak terbawa saat push lewat CLI -> pakai WANDB_MODE=offline.
  - Cek `kaggle kernels push --help` di versi CLI-mu untuk opsi pemilihan accelerator (T4x2 vs P100).
"""
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

USER = "your-kaggle-username"
SLOTS = {"A": Path("kernels/A"), "B": Path("kernels/B")}
MAX_SESSIONS = 3
POLL_SEC = 300


def sh(*args) -> str:
    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(args)}\n{r.stderr}")
    return r.stdout


def kernel_id(slot: str) -> str:
    return json.loads((SLOTS[slot] / "kernel-metadata.json").read_text())["id"]


def set_source(slot: str, prev_kernel: str | None):
    meta_p = SLOTS[slot] / "kernel-metadata.json"
    meta = json.loads(meta_p.read_text())
    meta["kernel_sources"] = [prev_kernel] if prev_kernel else []
    meta_p.write_text(json.dumps(meta, indent=2))


def wait_until_finished(kid: str) -> str:
    while True:
        out = sh("kaggle", "kernels", "status", kid).lower()
        print(time.strftime("%H:%M:%S"), out.strip())
        for s in ("complete", "error", "cancel"):
            if s in out:
                return s
        time.sleep(POLL_SEC)


def read_status(kid: str) -> dict:
    with tempfile.TemporaryDirectory() as d:
        # unduh output; status.json kecil (last.pt bisa besar — kalau lambat, pakai --file-pattern di CLI versi baru)
        sh("kaggle", "kernels", "output", kid, "-p", d)
        p = Path(d) / "status.json"
        return json.loads(p.read_text()) if p.exists() else {"status": "UNKNOWN"}


def main():
    prev, slot = None, "A"
    for n in range(MAX_SESSIONS):
        set_source(slot, prev)
        kid = kernel_id(slot)
        print(f"\n=== session {n + 1}: push {kid} (resume dari {prev}) ===")
        sh("kaggle", "kernels", "push", "-p", str(SLOTS[slot]))
        time.sleep(60)
        result = wait_until_finished(kid)
        st = read_status(kid)
        print(f"[{kid}] run={result} status={st}")
        if st.get("status") == "DONE":
            print("Training selesai.")
            return
        if st.get("status") != "RESUME_NEEDED":
            sys.exit("Run gagal sebelum sempat menulis checkpoint — cek log di Kaggle.")
        prev, slot = kid, ("B" if slot == "A" else "A")
    print("MAX_SESSIONS tercapai; lanjutkan minggu depan / naikkan limit.")


if __name__ == "__main__":
    main()
