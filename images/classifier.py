#!/usr/bin/env python3
"""
Healthy vs. unhealthy laryngeal-endoscopy image classifier — one self-contained file.

WHAT IT DOES
  * reads every image under  Healthy/  and  Unhealthy/  (any sub-folders, any case)
  * groups frames by imaging session (parsed from the file name) so the same
    session never lands in both train and test  -> honest accuracy
  * fixes the ~20:1 class imbalance by oversampling the healthy class and
    heavy train-time augmentation (crops / flips / rotation / colour jitter)
  * trains a ResNet-18 that was pre-trained on ImageNet (transfer learning)
  * runs grouped k-fold cross-validation, prints image-level AND session-level
    scores, then refits on ALL the data and saves  model.pt
  * with  --predict PATH  it just loads  model.pt  and scores new images

HOW TO RUN  (from this folder:  ~/Desktop/Stenosis/images)
  python3 classifier.py                  # train + cross-validate + save model.pt
  python3 classifier.py --quick          # fast smoke run (2 folds, few epochs)
  python3 classifier.py --crop 0.8       # leakage check: ignore image borders
  python3 classifier.py --mode finetune  # train the whole network, not just the head
  python3 classifier.py --predict some/new/folder     # score new images
  python3 classifier.py --predict pic1.jpg pic2.jpg

Everything is set up for this Mac already: it uses the Apple GPU (MPS) if
available and never spawns DataLoader workers (they deadlock on macOS).

Needs:  pip install torch torchvision pillow numpy scikit-learn
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFile
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader, Dataset

try:                                            # progress bars are optional
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **_):
        return x

ImageFile.LOAD_TRUNCATED_IMAGES = True          # some exports are slightly corrupt
SEED = 1234
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLASS_NAMES = ("healthy", "unhealthy")           # index 0 / 1

_COPY_RE = re.compile(r"^(?:коп[ияей]+|copy of|копия)\s+", re.IGNORECASE)
_DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})")
_CASE_RE = re.compile(r"^(\d{1,4})s?", re.IGNORECASE)


# --------------------------------------------------------------------------- data
def session_key(stem: str) -> str:
    """One imaging session per calendar date (2025-02-17_...) or per case
    number (001, 001S, 111 -> case001). Falls back to the whole name."""
    stem = _COPY_RE.sub("", stem.strip())
    m = _DATE_RE.match(stem)
    if m:
        return m.group(1)
    m = _CASE_RE.match(stem)
    if m:
        return f"case{int(m.group(1)):03d}"
    return stem


def find_data(root: Path) -> list[dict]:
    """Return [{path, label, session}, ...].  Prefers images_small/ if present."""
    small = root / "images_small"
    if small.is_dir():
        root = small
    kids = {c.name.lower(): c for c in root.iterdir() if c.is_dir()}
    items: list[dict] = []
    for label, name in enumerate(CLASS_NAMES):
        folder = kids.get(name)
        if folder is None:
            sys.exit(f"could not find a '{name}' folder under {root}")
        for p in sorted(folder.rglob("*")):
            if p.suffix.lower() in IMG_EXT:
                items.append(dict(path=str(p), label=label,
                                  session=f"{name}:{session_key(p.stem)}"))
    if not items:
        sys.exit(f"no images found under {root}")
    print(f"data root: {root}")
    return items


def audit(items: list[dict]) -> None:
    n = len(items)
    per_class = Counter(CLASS_NAMES[it["label"]] for it in items)
    sess = defaultdict(set)
    for it in items:
        sess[CLASS_NAMES[it["label"]]].add(it["session"])
    print("-" * 64)
    print(f"images            : {n}")
    print(f"  healthy         : {per_class['healthy']:>5}   "
          f"({len(sess['healthy'])} sessions)")
    print(f"  unhealthy       : {per_class['unhealthy']:>5}   "
          f"({len(sess['unhealthy'])} sessions)")
    if len(sess["healthy"]) < 15:
        print(f"NOTE: only {len(sess['healthy'])} healthy sessions — every score "
              "below has a wide error bar. Treat results as a rough estimate,\n"
              "      and run  --crop 0.8  to check the model isn't just reading "
              "image borders / overlays.")
    print("-" * 64)


def session_holdout(pool_idx: np.ndarray, items: list[dict], frac: float,
                    seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Carve a validation set out of ``pool_idx`` by whole sessions (never split
    a session), aiming for ~frac of the images and at least one healthy +
    one unhealthy session in validation."""
    rng = np.random.default_rng(seed)
    by_sess: dict[str, list[int]] = defaultdict(list)
    for i in pool_idx:
        by_sess[items[int(i)]["session"]].append(int(i))
    sessions = list(by_sess)
    rng.shuffle(sessions)
    target = max(1, int(frac * len(pool_idx)))
    val: list[int] = []
    have = set()
    for s in sessions:
        if len(val) >= target and len(have) >= 2:
            break
        lbl = items[by_sess[s][0]]["label"]
        if len(val) >= target and lbl in have:
            continue
        val.extend(by_sess[s])
        have.add(lbl)
    val_set = set(val)
    train = np.array([int(i) for i in pool_idx if int(i) not in val_set])
    return train, np.array(sorted(val_set))


def grouped_folds(items: list[dict], k: int, seed: int) -> list[np.ndarray]:
    """Assign a fold id to every item so that a session is wholly inside one
    fold and each class's sessions are spread round-robin across all k folds
    (guarantees every fold has both classes even with very few healthy sessions)."""
    rng = np.random.default_rng(seed)
    fold_of_session: dict[str, int] = {}
    by_class: dict[int, list[str]] = defaultdict(list)
    for it in items:
        if it["session"] not in fold_of_session:
            by_class[it["label"]].append(it["session"])
    for label, sessions in by_class.items():
        sessions = list(dict.fromkeys(sessions))
        rng.shuffle(sessions)
        start = int(rng.integers(0, k))
        for i, s in enumerate(sessions):
            fold_of_session[s] = (start + i) % k
    fold = np.array([fold_of_session[it["session"]] for it in items])
    return [np.where(fold == f)[0] for f in range(k)]


class EndoDataset(Dataset):
    def __init__(self, items, transform):
        self.items, self.transform = items, transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        with Image.open(it["path"]) as im:
            x = self.transform(im.convert("RGB"))
        return x, torch.tensor(float(it["label"])), i


def make_transforms(size: int, train: bool, crop_frac: float | None):
    from torchvision import transforms as T
    pre = []
    if crop_frac:                       # strip scope letterboxing / timestamps / overlays
        pre.append(T.Lambda(lambda im: T.functional.center_crop(
            im, [int(min(im.size) * crop_frac)] * 2)))
    if train:
        body = [
            T.RandomResizedCrop(size, scale=(0.7, 1.0), ratio=(0.85, 1.18)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(25),
            T.ColorJitter(0.25, 0.25, 0.20, 0.04),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            T.RandomErasing(p=0.25, scale=(0.02, 0.12)),
        ]
    else:
        body = [T.Resize(int(size * 1.14)), T.CenterCrop(size),
                T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(pre + body)


def oversample(train_items: list[dict], seed: int) -> list[dict]:
    """Duplicate the minority class up to parity. Because every epoch re-applies
    random augmentation, the copies are seen under different crops/rotations —
    i.e. this is augmentation-based rebalancing, not plain repetition."""
    rng = np.random.default_rng(seed)
    by_class = defaultdict(list)
    for it in train_items:
        by_class[it["label"]].append(it)
    n_max = max(len(v) for v in by_class.values())
    out = []
    for v in by_class.values():
        out.extend(v)
        if len(v) < n_max:
            extra = rng.choice(len(v), size=n_max - len(v), replace=True)
            out.extend(v[j] for j in extra)
    rng.shuffle(out)
    return out


# -------------------------------------------------------------------------- model
def build_model(mode: str) -> nn.Module:
    from torchvision import models
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    if mode == "linear":                       # freeze the backbone, train the head only
        for p in m.parameters():
            p.requires_grad = False
    m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.fc.in_features, 1))
    return m


def pick_device(explicit: str | None) -> torch.device:
    if explicit:
        return torch.device(explicit)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def predict_probs(model, loader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    P, Y, I = [], [], []
    for x, y, idx in loader:
        logits = model(x.to(device)).squeeze(1)
        P.append(torch.sigmoid(logits).cpu().numpy())
        Y.append(y.numpy())
        I.append(idx.numpy())
    return np.concatenate(P), np.concatenate(Y), np.concatenate(I)


def train_model(train_items, val_items, *, mode, size, crop, epochs, batch,
                lr, device, patience, verbose=True) -> nn.Module:
    tr = EndoDataset(oversample(train_items, SEED),
                     make_transforms(size, True, crop))
    va = EndoDataset(val_items, make_transforms(size, False, crop))
    tl = DataLoader(tr, batch_size=batch, shuffle=True, num_workers=0, drop_last=True)
    vl = DataLoader(va, batch_size=batch, shuffle=False, num_workers=0)

    model = build_model(mode).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    best_state, best_val, bad = None, np.inf, 0
    for ep in range(1, epochs + 1):
        model.train()
        run = 0.0
        bar = tqdm(tl, desc=f"  epoch {ep:2d}/{epochs}", leave=False, ncols=78,
                   disable=not verbose)
        for x, y, _ in bar:
            opt.zero_grad()
            out = model(x.to(device)).squeeze(1)
            loss = loss_fn(out, y.to(device))
            loss.backward()
            opt.step()
            run += loss.item() * len(y)
            if hasattr(bar, "set_postfix"):
                bar.set_postfix(loss=f"{loss.item():.3f}")
        run /= len(tl.dataset)

        vp, vy, _ = predict_probs(model, vl, device)
        vloss = nn.functional.binary_cross_entropy(
            torch.tensor(vp).clamp(1e-6, 1 - 1e-6), torch.tensor(vy)).item()
        try:
            vauc = roc_auc_score(vy, vp) if len(set(vy)) > 1 else float("nan")
        except ValueError:
            vauc = float("nan")
        if verbose:
            print(f"  epoch {ep:2d}/{epochs}  train_loss {run:.3f}  "
                  f"val_loss {vloss:.3f}  val_auc {vauc:.3f}")

        if vloss < best_val - 1e-4:
            best_val, best_state, bad = vloss, {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"  early stop at epoch {ep}")
                break
    if best_state:
        model.load_state_dict(best_state)
    return model


# ------------------------------------------------------------------------ metrics
def report(name: str, y: np.ndarray, p: np.ndarray) -> dict:
    yhat = (p >= 0.5).astype(int)
    auc = roc_auc_score(y, p) if len(set(y)) > 1 else float("nan")
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else float("nan")   # unhealthy caught
    spec = tn / (tn + fp) if (tn + fp) else float("nan")   # healthy caught
    acc = (tp + tn) / len(y)
    print(f"  {name:16s}  AUC {auc:.3f}   acc {acc:.3f}   "
          f"sensitivity {sens:.3f}   specificity {spec:.3f}")
    return dict(auc=auc, acc=acc, sens=sens, spec=spec)


def session_rollup(items, idx, p):
    """Mean probability per session -> one prediction per imaging session."""
    bag = defaultdict(list)
    lab = {}
    for j, prob in zip(idx, p):
        s = items[j]["session"]
        bag[s].append(prob)
        lab[s] = items[j]["label"]
    sess = sorted(bag)
    y = np.array([lab[s] for s in sess])
    pm = np.array([np.mean(bag[s]) for s in sess])
    return y, pm


# --------------------------------------------------------------------------- main
def run_cv(items, args, device):
    folds = grouped_folds(items, args.folds, SEED)
    oof_p = np.full(len(items), np.nan)
    img_aucs, sess_aucs = [], []
    for f, test_idx in enumerate(folds):
        rest = np.concatenate([folds[j] for j in range(args.folds) if j != f])
        train_idx, val_idx = session_holdout(rest, items, 0.15, SEED + f)

        print(f"\n=== fold {f + 1}/{args.folds}  "
              f"(train {len(train_idx)} img / test {len(test_idx)} img) ===")
        model = train_model(
            [items[i] for i in train_idx], [items[i] for i in val_idx],
            mode=args.mode, size=args.size, crop=args.crop, epochs=args.epochs,
            batch=args.batch, lr=args.lr, device=device, patience=args.patience,
            verbose=args.verbose)

        te = EndoDataset([items[i] for i in test_idx],
                         make_transforms(args.size, False, args.crop))
        tp, ty, ti = predict_probs(
            model, DataLoader(te, batch_size=args.batch, num_workers=0), device)
        oof_p[test_idx] = tp
        m = report(f"fold {f + 1} image", ty, tp)
        img_aucs.append(m["auc"])
        sy, sp = session_rollup(items, test_idx, tp)
        if len(set(sy)) > 1:
            sess_aucs.append(report(f"fold {f + 1} session", sy, sp)["auc"])

    y = np.array([it["label"] for it in items])
    ok = ~np.isnan(oof_p)
    sy, sp = session_rollup(items, np.where(ok)[0], oof_p[ok])

    def _ms(a):
        a = [v for v in a if not np.isnan(v)]
        return (float(np.mean(a)), float(np.std(a))) if a else (float("nan"), 0.0)

    ia_m, ia_s = _ms(img_aucs)
    sa_m, sa_s = _ms(sess_aucs)
    print("\n" + "=" * 64)
    print(f"CROSS-VALIDATED RESULT  ({args.folds} folds, sessions never shared)")
    print("=" * 64)
    print(f"  image-level   AUC : {ia_m:.3f} +/- {ia_s:.3f}   (mean +/- SD over folds)")
    print(f"  session-level AUC : {sa_m:.3f} +/- {sa_s:.3f}")
    print(f"  per-fold image AUC : {', '.join(f'{v:.2f}' for v in img_aucs)}")
    if sess_aucs:
        print(f"  per-fold sess  AUC : {', '.join(f'{v:.2f}' for v in sess_aucs)}")
    print("  at threshold 0.5, pooling every held-out prediction:")
    report("  image level", y[ok], oof_p[ok])
    report("  session level", sy, sp)
    print("\nSession-level is the number that means something: it asks whether a"
          "\nwhole unseen session is called correctly, not each near-duplicate frame."
          "\nA wide +/- SD or a low per-fold value means the estimate is fragile"
          "\n(expected here — 7 healthy sessions).")


def refit_and_save(items, args, device, out_path):
    print("\n" + "=" * 64)
    print(f"refitting on ALL {len(items)} images and writing {out_path}")
    print("=" * 64)
    train_idx, val_idx = session_holdout(np.arange(len(items)), items, 0.12, SEED)
    train_items = [items[i] for i in train_idx]
    val_items = [items[i] for i in val_idx]
    model = train_model(train_items, val_items, mode=args.mode, size=args.size,
                        crop=args.crop, epochs=args.epochs, batch=args.batch,
                        lr=args.lr, device=device, patience=args.patience,
                        verbose=args.verbose)
    torch.save(dict(
        state_dict={k: v.cpu() for k, v in model.state_dict().items()},
        arch="resnet18", mode=args.mode, size=args.size, crop=args.crop,
        class_names=CLASS_NAMES, mean=IMAGENET_MEAN, std=IMAGENET_STD,
    ), out_path)
    print(f"saved {out_path}")


def do_predict(paths, ckpt_path, device, threshold=0.5):
    ck = torch.load(ckpt_path, map_location="cpu")
    model = build_model(ck["mode"])
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    tf = make_transforms(ck["size"], False, ck.get("crop"))

    files: list[Path] = []
    for item in paths:
        p = Path(item)
        if p.is_dir():
            files += [q for q in sorted(p.rglob("*")) if q.suffix.lower() in IMG_EXT]
        elif p.suffix.lower() in IMG_EXT:
            files.append(p)
    if not files:
        sys.exit("no images found at the given path(s)")

    names = ck["class_names"]
    per_session = defaultdict(list)
    print(f"{'image':<50} {'P(unhealthy)':>13}  prediction")
    print("-" * 80)
    with torch.no_grad():
        for q in files:
            try:
                with Image.open(q) as im:
                    x = tf(im.convert("RGB")).unsqueeze(0).to(device)
                prob = torch.sigmoid(model(x).squeeze()).item()
            except Exception as e:                                    # noqa: BLE001
                print(f"{q.name:<50}  skipped ({e})")
                continue
            label = names[int(prob >= threshold)]
            per_session[session_key(q.stem)].append(prob)
            print(f"{q.name[:50]:<50} {prob:>13.3f}  {label}")

    if len(per_session) > 1 or any(len(v) > 1 for v in per_session.values()):
        print("\nper session (mean probability):")
        print("-" * 80)
        for s in sorted(per_session):
            m = float(np.mean(per_session[s]))
            print(f"  {s:<30} {m:>8.3f}   {names[int(m >= threshold)]}"
                  f"   ({len(per_session[s])} frames)")


def main():
    ap = argparse.ArgumentParser(
        description="Train / cross-validate / use a healthy-vs-unhealthy image classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--data", default=".",
                    help="folder containing Healthy/ and Unhealthy/ (or images_small/)")
    ap.add_argument("--mode", choices=["linear", "finetune"], default="linear",
                    help="'linear' trains only the classifier head (best for small data); "
                         "'finetune' trains the whole network")
    ap.add_argument("--folds", type=int, default=5, help="cross-validation folds")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--size", type=int, default=224, help="input image size")
    ap.add_argument("--lr", type=float, default=None,
                    help="learning rate (default 1e-3 linear / 1e-4 finetune)")
    ap.add_argument("--patience", type=int, default=6, help="early-stopping patience")
    ap.add_argument("--crop", type=float, default=None, metavar="FRAC",
                    help="keep only the central FRAC of each image, e.g. 0.8 "
                         "(border-leakage check)")
    ap.add_argument("--model-out", default="model.pt")
    ap.add_argument("--device", default=None, help="cpu / mps / cuda (auto by default)")
    ap.add_argument("--quick", action="store_true",
                    help="fast run: 2 folds, 8 epochs (for checking it works)")
    ap.add_argument("--no-save", action="store_true",
                    help="cross-validate only, don't refit/save model.pt")
    ap.add_argument("--quiet", dest="verbose", action="store_false",
                    help="hide per-epoch lines")
    ap.add_argument("--predict", nargs="+", metavar="PATH",
                    help="score images with an existing model.pt and exit")
    args = ap.parse_args()

    try:                        # show prints live in editors that buffer stdout
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:           # noqa: BLE001
        pass

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = pick_device(args.device)
    print(f"device: {device}")

    if args.predict:
        do_predict(args.predict, args.model_out, device)
        return

    if args.quick:
        args.folds, args.epochs = 2, 8
    if args.lr is None:
        args.lr = 1e-3 if args.mode == "linear" else 1e-4

    items = find_data(Path(args.data))
    audit(items)
    run_cv(items, args, device)
    if not args.no_save:
        refit_and_save(items, args, device, args.model_out)
    print("\ndone. Use the model on new images with:")
    print(f"  python3 {Path(sys.argv[0]).name} --predict <folder-or-images>")


if __name__ == "__main__":
    main()
