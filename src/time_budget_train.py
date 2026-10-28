"""
time_budget_train.py
====================
Training loop PyTorch yang "sadar waktu session" untuk Kaggle (12 jam GPU/CPU, 9 jam TPU)
dan Colab (~12 jam, bisa lebih cepat diputus).

Ide utama:
  1. Ukur waktu sejak session dimulai (SESSION_T0) -> hitung sisa budget.
  2. Estimasi durasi epoch & durasi save (EMA). Sebelum mulai epoch baru, cek:
         sisa >= est_epoch * 1.1 + reserve   ? lanjut : save + keluar rapi
  3. Di tengah epoch, cek tiap `check_every` step: kalau sisa < reserve -> save mid-epoch.
  4. Mid-epoch resume akurat: sampler deterministik (seed + epoch), lanjut dari batch ke-k
     tanpa harus me-load ulang batch yang sudah dilewati.
  5. Save atomik (tmp -> os.replace) supaya checkpoint tidak korup kalau kernel mati saat menulis.
  6. Saat start, cari checkpoint terbaru di OUT_DIR dan di /kaggle/input/** (output run sebelumnya
     yang di-attach sebagai input) -> resume otomatis.
  7. Tulis status.json: {"status": "RESUME_NEEDED" | "DONE", ...} supaya orchestrator
     (kaggle_chain.py) tahu harus push ulang atau berhenti.

Pemakaian di notebook (sel PALING ATAS, sebelum pip install dsb):
    import os, time; os.environ["SESSION_T0"] = str(time.time())
lalu di sel training:
    from time_budget_train import fit
    fit(model, optimizer, scheduler, train_ds, valid_fn, epochs=20, batch_size=32)
"""
from __future__ import annotations

import glob
import json
import os
import random
import signal
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler


# ----------------------------------------------------------------------------- environment
def detect_env() -> str:
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
        return "kaggle"
    try:
        import google.colab  # noqa: F401
        return "colab"
    except ImportError:
        return "local"


ENV = detect_env()
OUT_DIR = os.environ.get(
    "OUT_DIR",
    {"kaggle": "/kaggle/working", "colab": "/content/drive/MyDrive/ckpt", "local": "./ckpt"}[ENV],
)
os.makedirs(OUT_DIR, exist_ok=True)


def session_start_time() -> float:
    """Waktu mulai session. Prioritas: env SESSION_T0 > waktu start PID 1 (container init) > sekarang."""
    if "SESSION_T0" in os.environ:
        return float(os.environ["SESSION_T0"])
    try:
        import psutil
        return psutil.Process(1).create_time()  # di container, PID 1 start ~ saat session start
    except Exception:
        return time.time()


# ----------------------------------------------------------------------------- time budget
class TimeBudget:
    def __init__(self, budget_h: float | None = None, safety_min: float = 15.0):
        default_h = {"kaggle": 11.5, "colab": 11.0, "local": 1e6}[ENV]
        self.budget = float(os.environ.get("TIME_BUDGET_H", budget_h or default_h)) * 3600
        self.t0 = session_start_time()
        self.safety = safety_min * 60
        self.ema_epoch = None   # detik per epoch
        self.ema_save = 60.0    # tebakan awal durasi save
        self.stop_requested = False
        signal.signal(signal.SIGTERM, self._on_sigterm)  # kalau platform kirim SIGTERM

    def _on_sigterm(self, *_):
        self.stop_requested = True

    @staticmethod
    def _ema(old, new, a=0.3):
        return new if old is None else (1 - a) * old + a * new

    def elapsed(self) -> float:
        return time.time() - self.t0

    def remaining(self) -> float:
        return self.budget - self.elapsed()

    def reserve(self) -> float:
        return 2 * self.ema_save + self.safety

    def update_epoch(self, sec: float):
        self.ema_epoch = self._ema(self.ema_epoch, sec)

    def update_save(self, sec: float):
        self.ema_save = self._ema(self.ema_save, sec)

    def can_start_epoch(self) -> bool:
        if self.stop_requested:
            return False
        if self.ema_epoch is None:
            return self.remaining() > self.reserve()
        return self.remaining() > 1.1 * self.ema_epoch + self.reserve()

    def must_stop_now(self) -> bool:
        return self.stop_requested or self.remaining() < self.reserve()

    def __repr__(self):
        h = lambda s: f"{s / 3600:.2f}h"
        return f"[budget] elapsed={h(self.elapsed())} remaining={h(self.remaining())} reserve={h(self.reserve())}"


# ----------------------------------------------------------------------------- resumable sampler
class ResumableRandomSampler(Sampler):
    """Permutasi deterministik per epoch (seed + epoch); bisa mulai dari batch ke-k."""

    def __init__(self, n: int, batch_size: int, seed: int = 42):
        self.n, self.bs, self.seed = n, batch_size, seed
        self.epoch, self.start_batch = 0, 0

    def set_epoch(self, epoch: int, start_batch: int = 0):
        self.epoch, self.start_batch = epoch, start_batch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        perm = torch.randperm(self.n, generator=g).tolist()
        return iter(perm[self.start_batch * self.bs:])

    def __len__(self):
        return max(0, self.n - self.start_batch * self.bs)


# ----------------------------------------------------------------------------- checkpoint io
def atomic_save(obj, path: str) -> float:
    t = time.time()
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)  # atomik di filesystem yang sama
    return time.time() - t


def find_latest_checkpoint() -> str | None:
    cands = [os.path.join(OUT_DIR, "last.pt")]
    if ENV == "kaggle":
        cands += glob.glob("/kaggle/input/**/last.pt", recursive=True)
    best, best_step = None, -1
    for p in cands:
        if not os.path.exists(p):
            continue
        try:
            step = torch.load(p, map_location="cpu", weights_only=False)["global_step"]
        except Exception as e:  # checkpoint rusak -> lewati
            print(f"[ckpt] skip {p}: {e}")
            continue
        if step > best_step:
            best, best_step = p, step
    return best


def write_status(status: str, **kw):
    with open(os.path.join(OUT_DIR, "status.json"), "w") as f:
        json.dump({"status": status, "env": ENV, "time": time.time(), **kw}, f, indent=2)


def rng_state():
    st = {"py": random.getstate(), "np": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def set_rng_state(st):
    random.setstate(st["py"])
    np.random.set_state(st["np"])
    torch.set_rng_state(st["torch"])
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])


# ----------------------------------------------------------------------------- main loop
def fit(
    model,
    optimizer,
    scheduler,
    train_ds,
    valid_fn,                 # callable(model) -> float (semakin besar semakin bagus)
    epochs: int,
    batch_size: int = 32,
    accum_steps: int = 1,
    seed: int = 42,
    check_every: int = 50,    # cek budget tiap N step
    save_every_min: float = 30.0,  # periodic save (proteksi crash/disconnect tanpa sinyal)
    collate_fn=None,
    num_workers: int = 2,
    loss_fn=None,             # callable(model, batch) -> loss; default: model(**batch).loss
    budget: TimeBudget | None = None,
):
    budget = budget or TimeBudget()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    loss_fn = loss_fn or (lambda m, b: m(**b).loss)

    sampler = ResumableRandomSampler(len(train_ds), batch_size, seed)
    start_epoch, start_batch, global_step, best = 0, 0, 0, -float("inf")

    # --- resume
    ck_path = find_latest_checkpoint()
    if ck_path:
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        if scheduler is not None and ck.get("scheduler"):
            scheduler.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        set_rng_state(ck["rng"])
        start_epoch, start_batch = ck["epoch"], ck["batch_in_epoch"]
        global_step, best = ck["global_step"], ck["best"]
        budget.ema_epoch = ck.get("ema_epoch")
        print(f"[resume] {ck_path} -> epoch={start_epoch} batch={start_batch} step={global_step} best={best:.5f}")
    else:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    def save_last(epoch, batch_in_epoch):
        state = {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict(), "rng": rng_state(),
            "epoch": epoch, "batch_in_epoch": batch_in_epoch,
            "global_step": global_step, "best": best, "ema_epoch": budget.ema_epoch,
        }
        budget.update_save(atomic_save(state, os.path.join(OUT_DIR, "last.pt")))

    def graceful_exit(epoch, batch_in_epoch):
        save_last(epoch, batch_in_epoch)
        write_status("RESUME_NEEDED", epoch=epoch, batch=batch_in_epoch, step=global_step, best=best)
        print(f"[stop] budget hampir habis, checkpoint disimpan. {budget}")

    last_periodic = time.time()
    for epoch in range(start_epoch, epochs):
        if not budget.can_start_epoch():
            graceful_exit(epoch, 0)
            return "RESUME_NEEDED"

        sb = start_batch if epoch == start_epoch else 0
        sampler.set_epoch(epoch, sb)
        loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn,
                            num_workers=num_workers, pin_memory=(device == "cuda"), drop_last=True)
        model.train()
        t_epoch = time.time()
        n_batches_full = len(train_ds) // batch_size

        for i, batch in enumerate(loader, start=sb):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=(device == "cuda")):
                loss = loss_fn(model, batch) / accum_steps
            scaler.scale(loss).backward()
            if (i + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
                global_step += 1

            # cek budget hanya di batas accumulation supaya state optimizer konsisten
            if (i + 1) % accum_steps == 0 and global_step % check_every == 0:
                if budget.must_stop_now():
                    graceful_exit(epoch, i + 1)
                    return "RESUME_NEEDED"
                if time.time() - last_periodic > save_every_min * 60:
                    save_last(epoch, i + 1)
                    last_periodic = time.time()

        # epoch selesai -> estimasi waktu epoch (dinormalisasi kalau epoch ini parsial)
        frac = max(1e-6, (n_batches_full - sb) / max(1, n_batches_full))
        budget.update_epoch((time.time() - t_epoch) / frac)

        score = valid_fn(model)
        if score > best:
            best = score
            torch.save({k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()},
                       os.path.join(OUT_DIR, "best_model_fp16.pt"))  # ringan, untuk inference
        save_last(epoch + 1, 0)
        last_periodic = time.time()
        print(f"[epoch {epoch}] score={score:.5f} best={best:.5f} {budget}")

    write_status("DONE", step=global_step, best=best)
    print(f"[done] training selesai. best={best:.5f}")
    return "DONE"
