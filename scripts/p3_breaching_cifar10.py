#!/usr/bin/env python
"""v0.16 P3: breaching as the primary gradient inversion attack on CIFAR-10 with a small CNN.

Pre-registration: docs/preregistration/v0.16-p3.md (committed before this file). Runbook:
docs/v0.16-p3-colab-runbook.md. Every parameter below that the pre-registration fixes is a module
constant and is not a flag. Only `--quick` (tiny sizes for a CPU smoke test) changes sizes, and a
quick run never produces verdicts.

What it does
  1. Loads CIFAR-10 (train split) and trains or loads one small CNN (shared by all seeds).
  2. Tunes the total variation setting of both attacks on tuning seeds 9000 to 9002 (grid fixed
     in the pre-registration, criterion highest mean PSNR per view).
  3. For each evaluation seed simulates four clients with 8 records each, one full batch SGD step,
     and the views: plaintext, SecAgg+ aggregate (upstream quantiser arithmetic), n-1 collusion.
  4. Runs breaching (primary) and the repository attack (secondary) on every view, and attaches
     the label-only floor (class average image of the public pool) to every row.
  5. Writes one JSON per seed, atomically, and skips seeds already present (resumable).
  6. Writes summary.json with seed-level bootstrap intervals and the pre-registered verdicts.

Colab (Tesla T4, torch and torchvision preinstalled, do not reinstall them)
    pip install -q breaching==0.1.2 hydra-core==1.3.8 omegaconf==2.3.0 lpips==0.1.4 tqdm pyarrow pillow scipy

Usage
    python p3_breaching_cifar10.py --out /content/drive/MyDrive/p3_out --seeds 20
    python p3_breaching_cifar10.py --out /tmp/p3_quick --quick --seeds 2 --device cpu --synthetic-standin

Data. `--data-source auto` tries a cached npz, then torchvision, then the official tarball
(MD5 checked), then the Hugging Face parquet export `uoft-cs/cifar10` (SHA-256 checked, needs
pyarrow and pillow). `--synthetic-standin` builds a fake dataset with the same shapes. It is only
for the CPU smoke test, it is refused without `--quick`, and every JSON it writes is labelled.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import pickle
import sys
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

SCRIPT_VERSION = "p3-1"
SCHEMA_VERSION = 1

# ---- pre-registered constants (docs/preregistration/v0.16-p3.md) ---------------------------------
N_CLIENTS = 4
RECORDS_PER_CLIENT = 8
LOCAL_LR = 0.15
CLIP_RANGE = 8.0
QUANT_RANGE = 2 ** 22
MAX_WEIGHT = 1000.0
MAX_WEIGHT_SECONDARY = 16.0
EVAL_SEEDS_REQUIRED = 20
TUNING_SEEDS = (9000, 9001, 9002)
TV_GRID_BREACHING = (0.2, 0.05, 0.01, 0.001)
TV_GRID_REPO = (0.0005, 0.05, 0.1, 0.5)
BREACHING_ITERATIONS = 4000
BREACHING_LIBRARY_DEFAULT_ITERATIONS = 24000
BREACHING_LIBRARY_DEFAULT_TV = 0.2
REPO_ITERATIONS = 3000
REPO_ATTACK_LR = 0.1
CONTROL_RECORDS = 4
SELECTION_SEED_BASE = 1000
TRAIN_SUBSET_SEED = 12345
TRAIN_SUBSET_SIZE = 20000
TRAIN_EPOCHS = 10
TRAIN_LR = 1e-3
TRAIN_BATCH = 128
HELD_OUT_SIZE = 5000
RECOGNISABLE_MSE = 0.05
BOOT_RESAMPLES = 10000
BOOT_SEED = 0
MARGIN_PLAINTEXT = 1.0     # dB, R1 and R3 (lower limit must exceed it)
MARGIN_AGGREGATE = 1.0     # dB, R2 (upper limit must not exceed it)
MARGIN_CONTROL = 2.0       # dB, K0 (lower limit must reach it)
VACUOUS_BELOW = -1.0       # dB, R2 is vacuous when the upper limit is below it
B1_MAX_RATIO = 1.5
CIFAR_TAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR_TAR_MD5 = "c58f30108f718f92721af3b95e74349a"
HF_PARQUET_URL = "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/train-00000-of-00001.parquet"
HF_PARQUET_SHA256 = "8428b53a88a11ac374111006708df51469e315a22ac6d66470afd9c78d2ae883"

# Sizes that --quick replaces. Nothing else changes.
QUICK = {"standin_images": 2000, "train_subset": 400, "epochs": 1, "held_out": 200, "breaching_iterations": 8,
         "library_default_iterations": 12, "repo_iterations": 8, "tuning_seeds": (9000,)}

VIEWS = ("plaintext", "aggregate", "collusion")


# =================================================================================================
# small utilities
# =================================================================================================
def log(msg: str) -> None:
    print(f"[p3 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def atomic_write_json(path: Path, obj) -> None:
    """Write JSON to a temporary file in the same directory, flush, then rename over the target."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(obj, fh, indent=1)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def read_json(path: Path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# =================================================================================================
# data
# =================================================================================================
def make_standin(n: int = 2000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic stand-in with CIFAR-10 shapes (n, 32, 32, 3) uint8. Smoke test only, never a result."""
    rng = np.random.default_rng(seed)
    y = np.arange(n) % 10
    rng.shuffle(y)
    grid = np.linspace(0, 1, 32)
    yy, xx = np.meshgrid(grid, grid, indexing="ij")
    base = np.empty((10, 32, 32, 3))
    for c in range(10):
        f = rng.uniform(1, 4, size=(3, 2))
        ph = rng.uniform(0, 6.28, size=(3, 2))
        for ch in range(3):
            base[c, :, :, ch] = 0.5 + 0.35 * np.sin(2 * np.pi * f[ch, 0] * xx + ph[ch, 0]) * np.cos(2 * np.pi * f[ch, 1] * yy + ph[ch, 1])
    x = base[y] + rng.normal(0, 0.08, size=(n, 32, 32, 3))
    return (np.clip(x, 0, 1) * 255).round().astype(np.uint8), y.astype(np.int64)


def _download(url: str, timeout: float = 600.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "p3-breaching-cifar10"})
    with urllib.request.urlopen(req, timeout=timeout) as r:   # noqa: S310 (fixed https URLs)
        return r.read()


def _load_torchvision(root: Path):
    from torchvision.datasets import CIFAR10   # lazy, can fail on a mismatched build
    ds = CIFAR10(root=str(root), train=True, download=True)
    return np.asarray(ds.data, dtype=np.uint8), np.asarray(ds.targets, dtype=np.int64), "torchvision"


def _load_tarball(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    path = root / "cifar-10-python.tar.gz"
    if not path.is_file() or hashlib.md5(path.read_bytes()).hexdigest() != CIFAR_TAR_MD5:
        data = _download(CIFAR_TAR_URL)
        if hashlib.md5(data).hexdigest() != CIFAR_TAR_MD5:
            raise ValueError("CIFAR-10 tarball MD5 does not match, not used")
        path.write_bytes(data)
    xs, ys = [], []
    with tarfile.open(path) as tf:
        for i in range(1, 6):
            member = tf.extractfile(f"cifar-10-batches-py/data_batch_{i}")
            d = pickle.load(member, encoding="bytes")   # noqa: S301 (MD5 checked archive)
            xs.append(np.asarray(d[b"data"], dtype=np.uint8).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1))
            ys.append(np.asarray(d[b"labels"], dtype=np.int64))
    return np.concatenate(xs), np.concatenate(ys), "tarball"


def _load_hf(root: Path, allow_hash_mismatch: bool = False):
    import pyarrow.parquet as pq
    from PIL import Image
    root.mkdir(parents=True, exist_ok=True)
    path = root / "cifar10_train.parquet"
    if not path.is_file():
        path.write_bytes(_download(HF_PARQUET_URL))
    digest = sha256_bytes(path.read_bytes())
    if digest != HF_PARQUET_SHA256 and not allow_hash_mismatch:
        raise ValueError(f"parquet SHA-256 {digest} differs from the pinned value, pass --allow-data-hash-mismatch to accept")
    t = pq.read_table(path).to_pydict()
    x = np.stack([np.asarray(Image.open(io.BytesIO(d["bytes"])).convert("RGB"), dtype=np.uint8) for d in t["img"]])
    return x, np.asarray(t["label"], dtype=np.int64), "hf-parquet"


def validate_cifar(x: np.ndarray, y: np.ndarray) -> None:
    if x.shape != (50000, 32, 32, 3) or x.dtype != np.uint8 or y.shape != (50000,):
        raise ValueError(f"unexpected CIFAR-10 train shape {x.shape} {x.dtype} {y.shape}")
    if not (np.bincount(y, minlength=10) == 5000).all():
        raise ValueError("CIFAR-10 class counts are not 5000 each")


def load_data(data_dir: Path, source: str, *, standin: bool, quick: bool, allow_hash_mismatch: bool = False):
    """Returns (x uint8 (n,32,32,3), y int64, source name). Real data only unless `standin` (quick only)."""
    if standin:
        if not quick:
            raise SystemExit("--synthetic-standin is only allowed together with --quick (smoke test)")
        x, y = make_standin(QUICK["standin_images"])
        return x, y, "SYNTHETIC_STANDIN"
    data_dir.mkdir(parents=True, exist_ok=True)
    cache = data_dir / "cifar10_train_uint8.npz"
    if source in ("auto", "cache") and cache.is_file():
        with np.load(cache) as d:
            x, y = d["x"], d["y"]
            src = str(d["source"])
        validate_cifar(x, y)
        return x, y, src + "+cache"
    loaders = {"torchvision": lambda: _load_torchvision(data_dir), "tarball": lambda: _load_tarball(data_dir),
               "hf": lambda: _load_hf(data_dir, allow_hash_mismatch)}
    order = ["torchvision", "tarball", "hf"] if source == "auto" else [source]
    errors = []
    for name in order:
        try:
            log(f"loading CIFAR-10 through {name}")
            x, y, src = loaders[name]()
            validate_cifar(x, y)
            np.savez_compressed(cache, x=x, y=y, source=np.array(src))
            return x, y, src
        except Exception as exc:   # noqa: BLE001 (every loader may fail in its own way)
            errors.append(f"{name}: {type(exc).__name__}: {str(exc)[:160]}")
            log(f"loader {name} failed: {errors[-1]}")
    raise SystemExit("could not load CIFAR-10 from any source:\n  " + "\n  ".join(errors))


# =================================================================================================
# model
# =================================================================================================
class SmallCNN(nn.Module):
    """conv 3-16, ReLU, pool, conv 16-32, ReLU, pool, linear 2048-10. 25,578 parameters, no buffers."""

    def __init__(self, classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                      nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.classifier = nn.Linear(32 * 8 * 8, classes)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


def param_names(model: nn.Module) -> list[str]:
    return [n for n, _ in model.named_parameters()]


def state_sha256(state: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(state):
        h.update(k.encode())
        h.update(state[k].detach().cpu().numpy().astype(np.float32).tobytes())
    return h.hexdigest()


def to_float(x_uint8: np.ndarray) -> np.ndarray:
    """uint8 (n,32,32,3) to float32 (n,3,32,32) in [0, 1]."""
    return (x_uint8.astype(np.float32) / 255.0).transpose(0, 3, 1, 2).copy()


class Split:
    """Fixed index sets. Training subset M, eligible client records, held out accuracy set."""

    def __init__(self, n_total: int, train_size: int, held_out: int):
        perm = np.random.default_rng(TRAIN_SUBSET_SEED).permutation(n_total)
        self.train_idx = np.sort(perm[:train_size])
        self.eligible = np.setdiff1d(np.arange(n_total), self.train_idx)
        self.held_out_idx = self.eligible[:held_out]

    def client_records(self, seed: int) -> np.ndarray:
        k = N_CLIENTS * RECORDS_PER_CLIENT
        order = np.random.default_rng(SELECTION_SEED_BASE + seed).permutation(self.eligible)
        if len(order) < k:
            raise ValueError("not enough eligible records")
        return order[:k]


def train_or_load_model(out_dir: Path, x: np.ndarray, y: np.ndarray, split: Split, *, epochs: int, source: str,
                        device: str, quick: bool) -> tuple[dict, dict]:
    """Returns (state dict on CPU, metadata). Loads out/model.pt when its config matches, else trains."""
    path = out_dir / "model.pt"
    config = {"train_subset": int(len(split.train_idx)), "epochs": epochs, "lr": TRAIN_LR, "batch": TRAIN_BATCH,
              "data_source": source.replace("+cache", ""), "quick": quick, "subset_seed": TRAIN_SUBSET_SEED}
    if path.is_file():
        blob = torch.load(path, map_location="cpu", weights_only=False)
        if blob["config"] == config:
            log(f"loaded model {path} sha256 {blob['meta']['state_sha256'][:12]}")
            return blob["state"], blob["meta"]
        raise SystemExit(f"{path} exists with a different training config {blob['config']}, remove it or use another --out")
    log(f"training the global CNN on {len(split.train_idx)} images for {epochs} epochs on {device}")
    torch.manual_seed(0)
    model = SmallCNN().to(device)
    xt = torch.from_numpy(to_float(x[split.train_idx])).to(device)
    yt = torch.from_numpy(y[split.train_idx]).long().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=TRAIN_LR)
    gen = torch.Generator().manual_seed(0)
    for ep in range(epochs):
        order = torch.randperm(len(yt), generator=gen).to(device)
        total = 0.0
        for i in range(0, len(yt), TRAIN_BATCH):
            b = order[i:i + TRAIN_BATCH]
            opt.zero_grad()
            loss = F.cross_entropy(model(xt[b]), yt[b])
            loss.backward()
            opt.step()
            total += float(loss) * len(b)
        log(f"  epoch {ep + 1}/{epochs} train loss {total / len(yt):.4f}")
    model.eval()
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        xh = torch.from_numpy(to_float(x[split.held_out_idx])).to(device)
        acc = float((model(xh).argmax(1).cpu().numpy() == y[split.held_out_idx]).mean())
        xa = xt[:2000]
        train_acc = float((model(xa).argmax(1) == yt[:2000]).float().mean())
    meta = {"state_sha256": state_sha256(state), "held_out_accuracy": acc, "held_out_size": int(len(split.held_out_idx)),
            "train_accuracy_first_2000": train_acc, "parameters": int(sum(v.numel() for v in state.values()))}
    log(f"model held out accuracy {acc:.3f} ({meta['parameters']} parameters)")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state": state, "meta": meta, "config": config}, path)
    return state, meta


# =================================================================================================
# federated simulation: one local step, SecAgg+ quantiser arithmetic, collusion
# =================================================================================================
def flatten(state: dict) -> np.ndarray:
    return np.concatenate([v.detach().cpu().numpy().astype(np.float64).ravel() for v in state.values()])


def unflatten(vec: np.ndarray, like: dict) -> dict:
    out, i = {}, 0
    for k, v in like.items():
        n = v.numel()
        out[k] = torch.tensor(vec[i:i + n].reshape(tuple(v.shape)), dtype=v.dtype)
        i += n
    return out


def client_step(init: dict, x: np.ndarray, y: np.ndarray, *, lr: float = LOCAL_LR) -> dict:
    """One full batch SGD step on CPU. Returns the updated state dict (parameters only)."""
    model = SmallCNN()
    model.load_state_dict(init)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    opt.zero_grad()
    F.cross_entropy(model(torch.from_numpy(x)), torch.from_numpy(y).long()).backward()
    opt.step()
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _stochastic_round(arr: np.ndarray) -> np.ndarray:
    """Copy of flwr.common.secure_aggregation.quantization._stochastic_round (uses the global numpy RNG)."""
    ret = np.ceil(arr).astype(np.int32)
    rand_arr = np.random.rand(*ret.shape)
    ret[rand_arr < ret - arr] -= 1
    return ret


def quantize(arr: np.ndarray, clipping_range: float, target_range: int) -> np.ndarray:
    quantizer = target_range / (2 * clipping_range)
    return _stochastic_round((np.clip(arr, -clipping_range, clipping_range) + clipping_range) * quantizer)


def dequantize(arr: np.ndarray, clipping_range: float, target_range: int) -> np.ndarray:
    return arr.astype(float) * ((2 * clipping_range) / target_range) - clipping_range


def secagg_aggregate(vectors: list[np.ndarray], sizes: list[int], *, max_weight: float = MAX_WEIGHT,
                     rng_seed: int = 0) -> np.ndarray:
    """Weighted mean the server recovers from SecAgg+ (masks cancel, quantiser arithmetic remains).
    Same steps as `feddisclosure.attacks.view_study.secagg_aggregate`."""
    np.random.seed(rng_seed)
    total_q, summed = 0, None
    for v, n in zip(vectors, sizes):
        q_ratio = round(n / max_weight * QUANT_RANGE)
        dq_ratio = q_ratio / QUANT_RANGE
        q = quantize(v * dq_ratio, CLIP_RANGE, QUANT_RANGE).astype(np.int64)
        summed = q if summed is None else summed + q
        total_q += q_ratio
    agg = dequantize(summed, CLIP_RANGE, QUANT_RANGE)
    return (agg - (len(vectors) - 1) * CLIP_RANGE) * (QUANT_RANGE / total_q)


def exact_mean(vectors: list[np.ndarray], sizes: list[int]) -> np.ndarray:
    w = np.asarray(sizes, dtype=np.float64)
    return sum(v * wi for v, wi in zip(vectors, w)) / w.sum()


def collusion_view(aggregate: np.ndarray, plain: list[np.ndarray], sizes: list[int]) -> np.ndarray:
    """What the server and clients 1..n-1 compute for client 0. Same as `scaling_study.collusion_view`."""
    if len(plain) != len(sizes) or len(sizes) < 2:
        raise ValueError("need one state per client and at least two clients")
    total = float(sum(sizes))
    others = sum(plain[j] * sizes[j] / total for j in range(1, len(sizes)))
    return (aggregate - others) * total / sizes[0]


def gradient_of(init: dict, state: dict, lr: float = LOCAL_LR) -> list[torch.Tensor]:
    """Gradient the attacker derives from an observed model: (init minus final) over lr, in parameter order."""
    return [((init[k] - state[k]) / lr).float() for k in init]


def build_views(init: dict, cx: np.ndarray, cy: np.ndarray, seed: int) -> dict:
    """All observer views for one seed. cx (N,B,3,32,32) float32, cy (N,B).

    Returns {"grads": {view: list of tensors}, "labels": {view: (n,) int}, "truth": {view: images}}."""
    n, b = cx.shape[0], cx.shape[1]
    sizes = [b] * n
    states = [client_step(init, cx[i], cy[i]) for i in range(n)]
    vecs = [flatten(s) for s in states]
    views = {"plaintext": gradient_of(init, states[0])}
    out_states = {}
    for mw, tag in ((MAX_WEIGHT, ""), (MAX_WEIGHT_SECONDARY, "_maxw16")):
        agg_state = unflatten(secagg_aggregate(vecs, sizes, max_weight=mw, rng_seed=seed), init)
        col_state = unflatten(collusion_view(flatten(agg_state), vecs, sizes), init)
        out_states["aggregate" + tag] = agg_state
        out_states["collusion" + tag] = col_state
    out_states["aggregate_exact"] = unflatten(exact_mean(vecs, sizes), init)
    for name, st in out_states.items():
        views[name] = gradient_of(init, st)
    all_y, all_x = cy.reshape(-1), cx.reshape(-1, *cx.shape[2:])
    labels = {v: (all_y if v.startswith("aggregate") else cy[0]) for v in views}
    truth = {v: (all_x if v.startswith("aggregate") else cx[0]) for v in views}
    return {"grads": views, "labels": labels, "truth": truth, "states": states}


# =================================================================================================
# metrics and the label-only floor
# =================================================================================================
def psnr_from_mse(mse) -> np.ndarray:
    return -10.0 * np.log10(np.maximum(np.asarray(mse, dtype=np.float64), 1e-10))


def _gauss_window(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    ax = torch.arange(size, dtype=torch.float64) - (size - 1) / 2
    g = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return torch.outer(g, g)


def ssim_per_image(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """SSIM of image pairs (n,3,H,W) in [0, 1]. Gaussian window 11, sigma 1.5, valid region, mean over channels."""
    ta, tb = torch.from_numpy(np.asarray(a, dtype=np.float64)), torch.from_numpy(np.asarray(b, dtype=np.float64))
    c = ta.shape[1]
    w = _gauss_window()[None, None].repeat(c, 1, 1, 1)

    def conv(t):
        return F.conv2d(t, w, groups=c)

    mu_a, mu_b = conv(ta), conv(tb)
    saa, sbb, sab = conv(ta * ta) - mu_a ** 2, conv(tb * tb) - mu_b ** 2, conv(ta * tb) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mu_a * mu_b + c1) * (2 * sab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (saa + sbb + c2))
    return s.mean(dim=(1, 2, 3)).numpy()


def per_image_mse(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return ((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2).reshape(len(a), -1).mean(axis=1)


def match_to_originals(originals: np.ndarray, recon: np.ndarray) -> np.ndarray:
    """Optimal one to one assignment by MSE (evaluator privilege, as `score_victim`). Row i of the result
    is the reconstruction assigned to original i."""
    from scipy.optimize import linear_sum_assignment
    o = np.asarray(originals, np.float64).reshape(len(originals), -1)
    r = np.asarray(recon, np.float64).reshape(len(recon), -1)
    d = (o ** 2).mean(1)[:, None] + (r ** 2).mean(1)[None, :] - 2.0 * (o @ r.T) / o.shape[1]
    rows, cols = linear_sum_assignment(d)
    out = np.empty_like(recon)
    out[rows] = recon[cols]
    return out


def metrics(originals: np.ndarray, recon_matched: np.ndarray) -> dict:
    mse = per_image_mse(originals, recon_matched)
    return {"per_image_mse": mse.tolist(), "per_image_psnr": psnr_from_mse(mse).tolist(),
            "per_image_ssim": ssim_per_image(originals, recon_matched).tolist()}


class ClassStats:
    """Per class pixel sums of the full train split, so the public pool mean of any seed is a subtraction."""

    def __init__(self, x_uint8: np.ndarray, y: np.ndarray):
        self.sums = np.zeros((10, 3, 32, 32), dtype=np.float64)
        self.counts = np.bincount(y, minlength=10).astype(np.float64)
        for c in range(10):
            idx = np.flatnonzero(y == c)
            for i in range(0, len(idx), 2500):
                self.sums[c] += to_float(x_uint8[idx[i:i + 2500]]).astype(np.float64).sum(axis=0)

    def pool_means(self, client_x: np.ndarray, client_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Class average images and the overall mean image of the public pool (all images except the client records)."""
        sums, counts = self.sums.copy(), self.counts.copy()
        for xi, yi in zip(client_x, client_y):
            sums[int(yi)] -= xi.astype(np.float64)
            counts[int(yi)] -= 1
        if (counts <= 0).any():
            raise ValueError("public pool has no record of some class")
        return sums / counts[:, None, None, None], sums.sum(axis=0) / counts.sum()


def label_floor(victim_x: np.ndarray, victim_y: np.ndarray, class_avg: np.ndarray, mean_img: np.ndarray) -> dict:
    """Label-only floor. Every victim record is rebuilt as the class average image of the public pool.
    No matching for the headline floor. The matched floor and the mean image floor are for context."""
    recon = class_avg[np.asarray(victim_y, dtype=int)].astype(np.float32)
    out = metrics(victim_x, recon)
    out["matched_per_image_mse"] = per_image_mse(victim_x, match_to_originals(victim_x, recon)).tolist()
    out["mean_image_per_image_mse"] = per_image_mse(victim_x, np.broadcast_to(mean_img.astype(np.float32), victim_x.shape)).tolist()
    return out


# =================================================================================================
# attacks
# =================================================================================================
def _device_str(device) -> str:
    return str(device)


def breaching_attack(init: dict, grads: list[torch.Tensor], labels: np.ndarray, *, tv_scale: float | None,
                     iterations: int | None, device: str, seed: int) -> tuple[np.ndarray, dict]:
    """breaching 0.1.2 `invertinggradients` configuration, called through `prepare_attack`.

    tv_scale and iterations None means the shipped configuration unchanged. Labels are passed in, so the library
    label recovery is not used. Returns (n,3,32,32) float32 reconstructions and a small info dict."""
    import breaching
    from breaching.attacks import prepare_attack
    from omegaconf import OmegaConf

    overrides = []
    if iterations is not None:
        overrides.append(f"optim.max_iterations={int(iterations)}")
    if tv_scale is not None:
        overrides.append(f"regularization.total_variation.scale={float(tv_scale)}")
    with contextlib.redirect_stdout(io.StringIO()):
        cfg = breaching.get_attack_config("invertinggradients", overrides=overrides)
    dev = torch.device(device)
    model = SmallCNN()
    model.load_state_dict(init)
    model.to(dev).eval()
    n = int(len(labels))
    meta = OmegaConf.create({"shape": [3, 32, 32], "modality": "vision", "mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]})
    payload = [{"metadata": meta, "parameters": [p.detach().clone() for p in model.parameters()], "buffers": None}]
    shared = [{"gradients": [g.detach().clone().to(dev) for g in grads], "buffers": None,
               "metadata": {"num_data_points": n, "labels": torch.as_tensor(np.asarray(labels), dtype=torch.long, device=dev),
                            "local_hyperparams": None}}]
    attacker = prepare_attack(model, nn.CrossEntropyLoss(), cfg, setup=dict(dtype=torch.float, device=dev))
    torch.manual_seed(seed)
    rec, stats = attacker.reconstruct(payload, shared, server_secrets={})
    out = rec["data"].detach().cpu().numpy().astype(np.float32)
    invalid = bool((not np.isfinite(out).all()) or (not np.any(out)))
    if not np.isfinite(out).all():
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out, {"invalid": invalid, "iterations": int(cfg.optim.max_iterations), "tv_scale": float(cfg.regularization.total_variation.scale),
                 "final_objective": float(stats.get("opt_value", float("nan")))}


def repo_attack(init: dict, grads: list[torch.Tensor], labels: np.ndarray, *, tv_weight: float, iterations: int,
                device: str, seed: int, attack_lr: float = REPO_ATTACK_LR) -> tuple[np.ndarray, dict]:
    """The repository attack (Adam, squared L2 gradient matching, uniform [0.2, 0.8] start, clamp, light
    total variation) ported from feddisclosure.attacks.gradient_inversion to 3x32x32. Sees only the initial
    model, the observed gradient and the labels."""
    dev = torch.device(device)
    model = SmallCNN()
    model.load_state_dict(init)
    model.to(dev).eval()
    params = tuple(model.parameters())
    targets = [g.detach().to(dev) for g in grads]
    y = torch.as_tensor(np.asarray(labels), dtype=torch.long, device=dev)
    gen = torch.Generator().manual_seed(seed)
    dummy = (torch.rand((len(y), 3, 32, 32), generator=gen) * 0.6 + 0.2).to(dev).requires_grad_(True)
    opt = torch.optim.Adam([dummy], lr=attack_lr)
    for _ in range(iterations):
        opt.zero_grad()
        sim = torch.autograd.grad(F.cross_entropy(model(dummy), y), params, create_graph=True)
        match = sum((a - b).square().sum() for a, b in zip(sim, targets))
        tv = (dummy[:, :, 1:, :] - dummy[:, :, :-1, :]).abs().mean() + (dummy[:, :, :, 1:] - dummy[:, :, :, :-1]).abs().mean()
        (match + tv_weight * tv).backward()
        opt.step()
        with torch.no_grad():
            dummy.clamp_(0, 1)
    out = dummy.detach().cpu().numpy().astype(np.float32)
    return out, {"invalid": bool(not np.isfinite(out).all()), "iterations": int(iterations), "tv_weight": float(tv_weight),
                 "final_gradient_match": float(match.detach())}


# =================================================================================================
# one seed
# =================================================================================================
class Context:
    """Everything that is shared by the seeds of one run."""

    def __init__(self, args, x, y, source, split, init, model_meta, device):
        self.args, self.x, self.y, self.source, self.split = args, x, y, source, split
        self.init, self.model_meta, self.device = init, model_meta, device
        self.quick = bool(args.quick)
        self.stats = ClassStats(x, y)
        self.breaching_iters = QUICK["breaching_iterations"] if self.quick else BREACHING_ITERATIONS
        self.library_iters = QUICK["library_default_iterations"] if self.quick else BREACHING_LIBRARY_DEFAULT_ITERATIONS
        self.repo_iters = QUICK["repo_iterations"] if self.quick else REPO_ITERATIONS
        self.tuning_seeds = QUICK["tuning_seeds"] if self.quick else TUNING_SEEDS

    def seed_data(self, seed: int) -> dict:
        idx = self.split.client_records(seed)
        cx = to_float(self.x[idx]).reshape(N_CLIENTS, RECORDS_PER_CLIENT, 3, 32, 32)
        cy = self.y[idx].reshape(N_CLIENTS, RECORDS_PER_CLIENT)
        class_avg, mean_img = self.stats.pool_means(cx.reshape(-1, 3, 32, 32), cy.reshape(-1))
        floor = label_floor(cx[0], cy[0], class_avg, mean_img)
        return {"idx": idx, "cx": cx, "cy": cy, "floor": floor, "pool_size": int(len(self.y) - len(idx)),
                "views": build_views(self.init, cx, cy, seed)}


def score_row(attack: str, view: str, recon: np.ndarray, truth: np.ndarray, floor: dict, victim_count: int,
              info: dict, seconds: float, floor_slice: slice | None = None) -> dict:
    """Match, score the victim rows and attach the floor of the same images."""
    matched = match_to_originals(truth, recon) if len(truth) > 1 else recon
    m = metrics(truth[:victim_count], matched[:victim_count])
    fs = floor_slice or slice(0, victim_count)
    fl = {k: floor[k][fs] for k in ("per_image_mse", "per_image_psnr", "per_image_ssim", "matched_per_image_mse", "mean_image_per_image_mse")}
    return {"attack": attack, "view": view, **m, "floor": fl, "n_images": int(len(m["per_image_mse"])),
            "seconds": round(seconds, 2), "invalid_reconstructions": int(bool(info.get("invalid"))), "attack_info": info}


def run_one(ctx: Context, attack: str, view: str, grads, labels, truth, floor, victim_count, *, tv: float | None,
            iters: int | None, seed: int, floor_slice=None, label: str = "") -> dict:
    t = time.perf_counter()
    if attack == "breaching":
        rec, info = breaching_attack(ctx.init, grads, labels, tv_scale=tv, iterations=iters, device=ctx.device, seed=seed)
    else:
        rec, info = repo_attack(ctx.init, grads, labels, tv_weight=float(tv), iterations=int(iters), device=ctx.device, seed=seed)
    row = score_row(attack, view, rec, truth, floor, victim_count, info, time.perf_counter() - t, floor_slice)
    row["_recon"] = rec
    return row


def settings_hash(ctx: Context, tuning: dict) -> str:
    blob = {"quick": ctx.quick, "script": SCRIPT_VERSION, "model": ctx.model_meta["state_sha256"], "source": ctx.source,
            "tv": tuning["selected"], "iters": [ctx.breaching_iters, ctx.library_iters, ctx.repo_iters],
            "skip_library_default": bool(ctx.args.skip_library_default)}
    return sha256_bytes(json.dumps(blob, sort_keys=True).encode())[:16]


def run_seed(ctx: Context, seed: int, tuning: dict) -> tuple[dict, dict]:
    """All rows for one evaluation seed. Returns (json dict, images dict for the optional npz)."""
    sel = tuning["selected"]
    sd = ctx.seed_data(seed)
    views, floor = sd["views"], sd["floor"]
    rows, images = {}, {"originals_victim": (sd["cx"][0] * 255).round().astype(np.uint8)}
    att_seed = seed + 1
    plan = [("breaching", "plaintext", "plaintext"), ("breaching", "aggregate", "aggregate"), ("breaching", "collusion", "collusion"),
            ("breaching", "collusion_maxw16", "collusion"), ("breaching", "aggregate_exact", "aggregate"),
            ("repo", "plaintext", "plaintext"), ("repo", "aggregate", "aggregate"), ("repo", "collusion", "collusion"),
            ("repo", "collusion_maxw16", "collusion")]
    t0 = time.perf_counter()
    # control: batch of 1, plaintext, the first CONTROL_RECORDS victim records one at a time
    ctrl = []
    for k in range(CONTROL_RECORDS):
        st = client_step(ctx.init, sd["cx"][0][k:k + 1], sd["cy"][0][k:k + 1])
        r = run_one(ctx, "breaching", "ctrl_plaintext_b1", gradient_of(ctx.init, st), sd["cy"][0][k:k + 1], sd["cx"][0][k:k + 1],
                    floor, 1, tv=sel["breaching"]["plaintext"], iters=ctx.breaching_iters, seed=att_seed + 100 * (k + 1),
                    floor_slice=slice(k, k + 1))
        ctrl.append(r)
    row = {"attack": "breaching", "view": "ctrl_plaintext_b1", "floor": {}, "seconds": 0.0, "invalid_reconstructions": 0,
           "attack_info": {"tv_scale": sel["breaching"]["plaintext"], "iterations": ctx.breaching_iters, "single_record_attacks": CONTROL_RECORDS}}
    for key in ("per_image_mse", "per_image_psnr", "per_image_ssim"):
        row[key] = [v for r in ctrl for v in r[key]]
    for key in ("per_image_mse", "per_image_psnr", "per_image_ssim", "matched_per_image_mse", "mean_image_per_image_mse"):
        row["floor"][key] = [v for r in ctrl for v in r["floor"][key]]
    row["seconds"] = round(sum(r["seconds"] for r in ctrl), 2)
    row["invalid_reconstructions"] = int(sum(r["invalid_reconstructions"] for r in ctrl))
    row["n_images"] = len(row["per_image_mse"])
    rows["breaching/ctrl_plaintext_b1"] = row
    images["breaching/ctrl_plaintext_b1"] = np.concatenate([r["_recon"] for r in ctrl])
    log(f"seed {seed} control done ({row['seconds']}s)")
    for attack, view, family in plan:
        tv = sel[attack][family]
        iters = ctx.breaching_iters if attack == "breaching" else ctx.repo_iters
        r = run_one(ctx, attack, view, views["grads"][view], views["labels"][view], views["truth"][view], floor, RECORDS_PER_CLIENT,
                    tv=tv, iters=iters, seed=att_seed)
        images[f"{attack}/{view}"] = r.pop("_recon")
        rows[f"{attack}/{view}"] = r
        log(f"seed {seed} {attack}/{view} psnr {np.mean(r['per_image_psnr']):.2f} floor {np.mean(r['floor']['per_image_psnr']):.2f} ({r['seconds']}s)")
    if not ctx.args.skip_library_default:
        r = run_one(ctx, "breaching", "plaintext_ig_default", views["grads"]["plaintext"], views["labels"]["plaintext"],
                    views["truth"]["plaintext"], floor, RECORDS_PER_CLIENT, tv=None, iters=None if not ctx.quick else ctx.library_iters,
                    seed=att_seed)
        images["breaching/plaintext_ig_default"] = r.pop("_recon")
        rows["breaching/plaintext_ig_default"] = r
        log(f"seed {seed} breaching/plaintext_ig_default psnr {np.mean(r['per_image_psnr']):.2f} ({r['seconds']}s)")
    out = {"schema_version": SCHEMA_VERSION, "script_version": SCRIPT_VERSION, "seed": seed, "settings_hash": settings_hash(ctx, tuning),
           "quick": ctx.quick, "synthetic_standin": ctx.source == "SYNTHETIC_STANDIN", "data_source": ctx.source,
           "device": ctx.device, "versions": versions(), "model": ctx.model_meta,
           "config": {"clients": N_CLIENTS, "records_per_client": RECORDS_PER_CLIENT, "lr": LOCAL_LR, "clip_range": CLIP_RANGE,
                      "quantization_range": QUANT_RANGE, "max_weight": MAX_WEIGHT, "max_weight_secondary": MAX_WEIGHT_SECONDARY,
                      "breaching_iterations": ctx.breaching_iters, "repo_iterations": ctx.repo_iters,
                      "library_default_iterations": ctx.library_iters, "selected_tv": sel},
           "public_pool_size": sd["pool_size"], "victim_records": [int(i) for i in sd["idx"][:RECORDS_PER_CLIENT]],
           "victim_labels": [int(v) for v in sd["cy"][0]], "floor": floor, "rows": rows,
           "seconds_total": round(time.perf_counter() - t0, 1)}
    return out, images


def versions() -> dict:
    v = {"python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__, "cuda": torch.version.cuda,
         "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    for name in ("breaching", "scipy", "torchvision", "hydra-core", "omegaconf", "lpips"):
        try:
            from importlib import metadata
            v[name] = metadata.version(name)
        except Exception:   # noqa: BLE001
            v[name] = None
    return v


# =================================================================================================
# tuning
# =================================================================================================
def tune_seed(ctx: Context, seed: int) -> dict:
    """PSNR of every grid point on one tuning seed, for both attacks and the three tuned views."""
    sd = ctx.seed_data(seed)
    views, res = sd["views"], {"seed": seed, "breaching": {}, "repo": {}}
    for attack, grid in (("breaching", TV_GRID_BREACHING), ("repo", TV_GRID_REPO)):
        iters = ctx.breaching_iters if attack == "breaching" else ctx.repo_iters
        for view in VIEWS:
            for tv in grid:
                r = run_one(ctx, attack, view, views["grads"][view], views["labels"][view], views["truth"][view], sd["floor"],
                            RECORDS_PER_CLIENT, tv=tv, iters=iters, seed=seed + 1)
                res[attack].setdefault(view, {})[repr(tv)] = r["per_image_psnr"]
                log(f"tuning seed {seed} {attack}/{view} tv {tv}: psnr {np.mean(r['per_image_psnr']):.2f} ({r['seconds']}s)")
    return res


def select_tuning(per_seed: list[dict]) -> dict:
    """Highest mean PSNR over all tuning images, ties to the earlier grid entry."""
    selected, table = {"breaching": {}, "repo": {}}, {"breaching": {}, "repo": {}}
    for attack, grid in (("breaching", TV_GRID_BREACHING), ("repo", TV_GRID_REPO)):
        for view in VIEWS:
            means = {}
            for tv in grid:
                vals = [v for r in per_seed for v in r[attack][view][repr(tv)]]
                means[repr(tv)] = float(np.mean(vals))
            best = max(grid, key=lambda tv: (means[repr(tv)], -grid.index(tv)))
            selected[attack][view] = best
            table[attack][view] = means
    return {"selected": selected, "mean_psnr": table, "seeds": [r["seed"] for r in per_seed],
            "criterion": "highest mean PSNR over tuning images, ties to the earlier grid entry",
            "grids": {"breaching": list(TV_GRID_BREACHING), "repo": list(TV_GRID_REPO)}}


def run_tuning(ctx: Context, out_dir: Path) -> dict:
    final = out_dir / "tuning.json"
    cached = read_json(final)
    if cached and cached.get("quick") == ctx.quick and cached.get("model_sha256") == ctx.model_meta["state_sha256"]:
        log("tuning.json present, reusing it")
        return cached
    per_seed = []
    for s in ctx.tuning_seeds:
        f = out_dir / f"tuning_seed{s}.json"
        r = read_json(f)
        if r is None:
            r = tune_seed(ctx, s)
            atomic_write_json(f, r)
        else:
            log(f"tuning seed {s} present, skipped")
        per_seed.append(r)
    sel = select_tuning(per_seed)
    sel.update({"quick": ctx.quick, "model_sha256": ctx.model_meta["state_sha256"], "script_version": SCRIPT_VERSION,
                "iterations": {"breaching": ctx.breaching_iters, "repo": ctx.repo_iters}})
    atomic_write_json(final, sel)
    log("tuning selected: " + json.dumps(sel["selected"]))
    return sel


# =================================================================================================
# statistics and verdicts
# =================================================================================================
def wilson(k: int, n: int, z: float = 1.959964) -> list[float]:
    if n <= 0:
        return [float("nan"), float("nan")]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def boot_indices(n_seeds: int) -> np.ndarray:
    return np.random.default_rng(BOOT_SEED).integers(0, n_seeds, size=(BOOT_RESAMPLES, n_seeds))


def boot_ci(per_seed: np.ndarray, idx: np.ndarray) -> dict:
    v = np.asarray(per_seed, dtype=float)
    means = v[idx].mean(axis=1)
    return {"mean": float(v.mean()), "lo": float(np.quantile(means, 0.025)), "hi": float(np.quantile(means, 0.975)),
            "sd_seed": float(v.std(ddof=1)) if len(v) > 1 else float("nan")}


def verdict_lower(ci: dict, margin: float, *, inclusive: bool = False) -> dict:
    """Pass when the lower limit is above the margin (at or above when inclusive)."""
    ok = ci["lo"] >= margin if inclusive else ci["lo"] > margin
    label = "PASS" if ok else ("FAIL, underpowered" if ci["mean"] > margin else "FAIL")
    return {"verdict": label, "rule": f"lower limit {'>=' if inclusive else '>'} {margin:+.1f} dB", "mean": ci["mean"],
            "lo": ci["lo"], "hi": ci["hi"]}


def verdict_upper(ci: dict, margin: float, vacuous_below: float) -> dict:
    """Pass when the upper limit does not exceed the margin. Vacuous when it is below `vacuous_below`."""
    ok = ci["hi"] <= margin
    label = "PASS" if ok else ("FAIL, underpowered" if ci["mean"] <= margin else "FAIL")
    return {"verdict": label, "rule": f"upper limit <= {margin:+.1f} dB", "mean": ci["mean"], "lo": ci["lo"], "hi": ci["hi"],
            "vacuous": bool(ok and ci["hi"] < vacuous_below)}


def seed_arrays(runs: list[dict], key: str) -> dict:
    """Per seed means of the paired quantities of one row."""
    psnr = np.array([np.mean(np.array(r["rows"][key]["per_image_psnr"]) - np.array(r["rows"][key]["floor"]["per_image_psnr"])) for r in runs])
    ssim = np.array([np.mean(np.array(r["rows"][key]["per_image_ssim"]) - np.array(r["rows"][key]["floor"]["per_image_ssim"])) for r in runs])
    mse = np.array([np.mean(r["rows"][key]["per_image_mse"]) for r in runs])
    fmse = np.array([np.mean(r["rows"][key]["floor"]["per_image_mse"]) for r in runs])
    return {"gain_psnr": psnr, "gain_ssim": ssim, "mse": mse, "floor_mse": fmse}


def summarise(runs: list[dict]) -> dict:
    """Seed level bootstrap intervals for every row and the pre-registered verdicts."""
    runs = sorted(runs, key=lambda r: r["seed"])
    n = len(runs)
    quick = any(r["quick"] for r in runs)
    standin = any(r["synthetic_standin"] for r in runs)
    reasons = []
    if n != EVAL_SEEDS_REQUIRED:
        reasons.append(f"{n} seeds present, the pre-registered size is {EVAL_SEEDS_REQUIRED}")
    if quick:
        reasons.append("quick run")
    if standin:
        reasons.append("synthetic stand-in data")
    if [r["seed"] for r in runs] != list(range(n)):
        reasons.append("seeds are not 0..n-1")
    idx = boot_indices(n)
    keys = list(runs[0]["rows"])
    rows, arrs = {}, {}
    for key in keys:
        a = seed_arrays(runs, key)
        arrs[key] = a
        mse_all = np.array([v for r in runs for v in r["rows"][key]["per_image_mse"]])
        fl_all = np.array([v for r in runs for v in r["rows"][key]["floor"]["per_image_mse"]])
        k, tot = int((mse_all < RECOGNISABLE_MSE).sum()), len(mse_all)
        kf = int((fl_all < RECOGNISABLE_MSE).sum())
        rows[key] = {
            "n_images": tot, "gain_psnr_db": boot_ci(a["gain_psnr"], idx), "gain_ssim": boot_ci(a["gain_ssim"], idx),
            "attack_mean_psnr": float(np.mean([v for r in runs for v in r["rows"][key]["per_image_psnr"]])),
            "floor_mean_psnr": float(np.mean([v for r in runs for v in r["rows"][key]["floor"]["per_image_psnr"]])),
            "attack_mean_mse": float(mse_all.mean()), "floor_mean_mse": float(fl_all.mean()),
            "image_win_rate_vs_floor": float((mse_all < fl_all).mean()),
            "recognisable_mse_below_0_05": {"rate": k / tot, "wilson_optimistic": wilson(k, tot), "floor_rate": kf / tot},
            "seconds_mean_per_seed": float(np.mean([r["rows"][key]["seconds"] for r in runs])),
            "invalid_reconstructions": int(sum(r["rows"][key]["invalid_reconstructions"] for r in runs))}
    v = {}
    g = lambda k: rows[k]["gain_psnr_db"]   # noqa: E731
    v["K0"] = verdict_lower(g("breaching/ctrl_plaintext_b1"), MARGIN_CONTROL, inclusive=True)
    v["R1"] = verdict_lower(g("breaching/plaintext"), MARGIN_PLAINTEXT)
    v["R2"] = verdict_upper(g("breaching/aggregate"), MARGIN_AGGREGATE, VACUOUS_BELOW)
    v["R3"] = verdict_lower(g("breaching/collusion"), MARGIN_PLAINTEXT)
    v["R1r"] = verdict_lower(g("repo/plaintext"), MARGIN_PLAINTEXT)
    v["R2r"] = verdict_upper(g("repo/aggregate"), MARGIN_AGGREGATE, VACUOUS_BELOW)
    v["R3r"] = verdict_lower(g("repo/collusion"), MARGIN_PLAINTEXT)
    ratio_seed_num, ratio_seed_den = arrs["repo/plaintext"]["mse"], arrs["breaching/plaintext"]["mse"]
    ratios = ratio_seed_num[idx].mean(axis=1) / ratio_seed_den[idx].mean(axis=1)
    ratio = float(ratio_seed_num.mean() / ratio_seed_den.mean())
    v["B1c"] = {"verdict": "PASS" if ratio <= B1_MAX_RATIO else "FAIL", "rule": f"repo mean MSE / breaching mean MSE <= {B1_MAX_RATIO}",
                "ratio": ratio, "ratio_ci95": [float(np.quantile(ratios, 0.025)), float(np.quantile(ratios, 0.975))]}

    def paired(a, b, name, rule):
        d = boot_ci(arrs[a]["gain_psnr"] - arrs[b]["gain_psnr"], idx)
        return {"verdict": "PASS" if d["lo"] > 0 else ("FAIL, underpowered" if d["mean"] > 0 else "FAIL"), "rule": rule,
                "mean": d["mean"], "lo": d["lo"], "hi": d["hi"]}

    v["O1"] = paired("breaching/plaintext", "breaching/collusion", "O1", "gain(plaintext) minus gain(collusion), lower limit > 0")
    v["O2"] = paired("breaching/collusion", "breaching/aggregate", "O2", "gain(collusion) minus gain(aggregate), lower limit > 0")
    diffs = {}
    for view in ("plaintext", "aggregate", "collusion"):
        diffs[view] = boot_ci(arrs[f"breaching/{view}"]["gain_psnr"] - arrs[f"repo/{view}"]["gain_psnr"], idx)
    sign_note = [k for k in rows if np.sign(rows[k]["gain_psnr_db"]["mean"]) != np.sign(rows[k]["gain_ssim"]["mean"])]
    sds = [rows[k]["gain_psnr_db"]["sd_seed"] for k in ("breaching/plaintext", "breaching/aggregate", "breaching/collusion")]
    out = {"schema_version": SCHEMA_VERSION, "script_version": SCRIPT_VERSION, "n_seeds": n,
           "status": "verdicts computed" if not reasons else "no verdict: " + "; ".join(reasons),
           "bootstrap": {"resamples": BOOT_RESAMPLES, "seed": BOOT_SEED, "unit": "evaluation seed"},
           "data_source": runs[0]["data_source"], "model": runs[0]["model"], "device": runs[0]["device"],
           "rows": rows, "verdicts": v if not reasons else {"withheld": v}, "breaching_minus_repo_gain_db": diffs,
           "rows_where_psnr_and_ssim_gain_disagree_in_sign": sign_note,
           "observed_seed_sd_of_gain_db": {"breaching_plaintext_aggregate_collusion": sds, "power_assumption_db": 1.5,
                                           "power_void_if_above_db": 3.0}}
    return out


def print_summary(s: dict) -> None:
    log(f"summary status: {s['status']}")
    for k, r in s["rows"].items():
        g = r["gain_psnr_db"]
        log(f"  {k:36s} gain {g['mean']:+6.2f} dB [{g['lo']:+6.2f}, {g['hi']:+6.2f}]  attack {r['attack_mean_psnr']:5.2f} floor {r['floor_mean_psnr']:5.2f}")
    vs = s["verdicts"].get("withheld", s["verdicts"])
    for k, x in vs.items():
        log(f"  {k:4s} {x['verdict']}" + (" (vacuous)" if x.get("vacuous") else ""))


# =================================================================================================
# main
# =================================================================================================
def pick_device(arg: str) -> str:
    if arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if arg == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is not available")
    return arg


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", type=Path, required=True, help="output directory (seed JSON files, tuning, model, summary)")
    p.add_argument("--seeds", type=int, default=EVAL_SEEDS_REQUIRED, help="number of evaluation seeds, 0..N-1 (pre-registered: 20)")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--quick", action="store_true", help="tiny sizes for a CPU smoke test, never produces verdicts")
    p.add_argument("--stage", default="all", choices=["all", "tune", "eval", "summarise"])
    p.add_argument("--data-dir", type=Path, default=Path("p3_data"))
    p.add_argument("--data-source", default="auto", choices=["auto", "torchvision", "tarball", "hf"])
    p.add_argument("--allow-data-hash-mismatch", action="store_true")
    p.add_argument("--synthetic-standin", action="store_true", help="fake data, only with --quick (smoke test)")
    p.add_argument("--skip-library-default", action="store_true", help="skip the 24000 iteration library default row (secondary)")
    p.add_argument("--no-images", action="store_true", help="do not save reconstructions for seeds 0 and 1")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.seeds < 1:
        raise SystemExit("--seeds must be at least 1")
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    log(f"device {device}, torch {torch.__version__}, quick {args.quick}")
    if args.stage == "summarise":
        return write_summary(out_dir, args.seeds)
    x, y, source = load_data(args.data_dir, args.data_source, standin=args.synthetic_standin, quick=args.quick,
                             allow_hash_mismatch=args.allow_data_hash_mismatch)
    if source == "SYNTHETIC_STANDIN":
        log("*** SYNTHETIC STAND-IN DATA. Smoke test only. Nothing from this run is a result. ***")
    split = Split(len(y), QUICK["train_subset"] if args.quick else TRAIN_SUBSET_SIZE, QUICK["held_out"] if args.quick else HELD_OUT_SIZE)
    init, model_meta = train_or_load_model(out_dir, x, y, split, epochs=QUICK["epochs"] if args.quick else TRAIN_EPOCHS, source=source,
                                           device=device, quick=args.quick)
    ctx = Context(args, x, y, source, split, init, model_meta, device)
    tuning = run_tuning(ctx, out_dir)
    if args.stage == "tune":
        return 0
    expected_hash = settings_hash(ctx, tuning)
    for seed in range(args.seeds):
        f = out_dir / f"seed{seed}.json"
        done = read_json(f)
        if done is not None and done.get("schema_version") == SCHEMA_VERSION and "rows" in done:
            if done.get("settings_hash") != expected_hash:
                raise SystemExit(f"{f} was written with different settings ({done.get('settings_hash')} vs {expected_hash}), use a new --out")
            log(f"seed {seed} present, skipped")
            continue
        t = time.perf_counter()
        result, images = run_seed(ctx, seed, tuning)
        if seed < 2 and not args.no_images:
            np.savez_compressed(out_dir / f"images_seed{seed}.npz", **{k.replace("/", "__"): v.astype(np.float16) if v.dtype != np.uint8 else v for k, v in images.items()})
        atomic_write_json(f, result)
        log(f"seed {seed} written ({time.perf_counter() - t:.1f}s)")
    return write_summary(out_dir, args.seeds)


def write_summary(out_dir: Path, n_seeds: int) -> int:
    runs = []
    for seed in range(n_seeds):
        r = read_json(out_dir / f"seed{seed}.json")
        if r is not None and "rows" in r:
            runs.append(r)
    if len(runs) < 2:
        log("fewer than two seed files, no summary written")
        return 0
    s = summarise(runs)
    atomic_write_json(out_dir / "summary.json", s)
    print_summary(s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
