"""Backbone comparison for healthy vs unhealthy laryngeal endoscopy.

Patient-grouped repeated stratified CV, transfer learning, image-level and
patient-level metrics, Nadeau-Bengio corrected comparison, Grad-CAM.

    python3 img_benchmark.py --root images --repeats 2 --folds 5
    python3 img_benchmark.py --root images --backbones resnet50 convnext_tiny
    python3 img_benchmark.py --root images --mode linear_probe
    python3 img_benchmark.py --smoke            # random weights, synthetic images
"""
import argparse
import json
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             brier_score_loss, confusion_matrix, roc_auc_score)
from torch.utils.data import DataLoader

from img_data import (EndoscopyDataset, audit, build_index, build_transforms,
                      oversample_to_balance, patient_splits, print_audit)

warnings.filterwarnings("ignore")
SEED = 20240813

DEFAULT_BACKBONES = [
    "resnet18", "resnet50", "efficientnet_b0",
    "densenet121", "convnext_tiny", "vit_small_patch16_224",
]


def make_model(name, pretrained=True, mode="finetune", dropout=0.3):
    """mode: 'linear_probe' freezes the backbone (right choice for small n),
    'finetune' unfreezes everything at a low LR, 'last_block' unfreezes the tail."""
    import timm
    m = timm.create_model(name, pretrained=pretrained, num_classes=1,
                          drop_rate=dropout)
    if mode == "linear_probe":
        for p in m.parameters():
            p.requires_grad = False
        for p in m.get_classifier().parameters():
            p.requires_grad = True
    elif mode == "last_block":
        for p in m.parameters():
            p.requires_grad = False
        groups = [c for c in m.children()]
        for c in groups[-3:]:
            for p in c.parameters():
                p.requires_grad = True
    return m


def train_one(model, tr_loader, va_loader, device, epochs=30, lr=3e-4,
              weight_decay=1e-4, patience=7, pos_weight=1.0):
    model.to(device)
    crit = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best, best_state, bad = -np.inf, None, 0
    for ep in range(epochs):
        model.train()
        for x, y, _ in tr_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x).squeeze(-1), y)
            loss.backward()
            nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
        sched.step()
        p, yv, _ = infer(model, va_loader, device)
        auc = roc_auc_score(yv, p) if len(np.unique(yv)) > 1 else 0.5
        if auc > best + 1e-4:
            best, bad = auc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    return model, best


@torch.no_grad()
def infer(model, loader, device):
    model.eval()
    ps, ys, idx = [], [], []
    for x, y, i in loader:
        p = torch.sigmoid(model(x.to(device)).squeeze(-1)).cpu().numpy()
        ps.append(p)
        ys.append(y.numpy())
        idx.append(i.numpy())
    return np.concatenate(ps), np.concatenate(ys), np.concatenate(idx)


def metrics(y, p, prefix=""):
    yhat = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return {
        f"{prefix}auc": roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan,
        f"{prefix}ap": average_precision_score(y, p),
        f"{prefix}bacc": balanced_accuracy_score(y, yhat),
        f"{prefix}sens": tp / (tp + fn) if tp + fn else np.nan,
        f"{prefix}spec": tn / (tn + fp) if tn + fp else np.nan,
        f"{prefix}brier": brier_score_loss(y, p),
    }


def patient_level(df, test_idx, p, y):
    """Aggregate frame predictions to one prediction per patient (mean prob).
    This is the number that matters clinically -- you diagnose children, not frames."""
    sub = df.iloc[test_idx].copy()
    sub["p"], sub["y"] = p, y
    g = sub.groupby("patient_id").agg(p=("p", "mean"), y=("y", "first"))
    return metrics(g["y"].values, g["p"].values, prefix="pat_")


def corrected_ttest(diff, n_train, n_test):
    from scipy import stats
    d = np.asarray(diff, float)
    d = d[~np.isnan(d)]
    if len(d) < 2:
        return np.nan, np.nan
    var = d.var(ddof=1) * (1 / len(d) + n_test / n_train)
    if var <= 0:
        return np.nan, np.nan
    t = d.mean() / np.sqrt(var)
    return t, 2 * stats.t.sf(abs(t), df=len(d) - 1)


def run(df, backbones, repeats, folds, mode, size, epochs, batch, lr,
        pretrained, center_crop_frac, device, workers, oversample=False):
    splits = patient_splits(df, folds, repeats, SEED)
    rows = []
    for name in backbones:
        t0 = time.time()
        for rep, k, tr, te in splits:
            torch.manual_seed(SEED + k)
            tr_df, te_df = df.iloc[tr], df.iloc[te]
            # inner validation split, also patient-grouped, for early stopping.
            # 4 folds -> ~25% of the (already scarce) healthy sessions held out
            # for validation while still leaving >=1 per side.
            inner = patient_splits(tr_df.reset_index(drop=True), 4, 1, SEED + k)[0]
            fit_df = tr_df.reset_index(drop=True).iloc[inner[2]]
            val_df = tr_df.reset_index(drop=True).iloc[inner[3]]

            # class rebalancing on the training split only
            raw_pos = max(int(fit_df["label"].sum()), 1)
            raw_neg = max(len(fit_df) - int(fit_df["label"].sum()), 1)
            if oversample:
                fit_df = oversample_to_balance(fit_df, seed=SEED + k)
                pos_weight = 1.0            # classes already even after resampling
            else:
                pos_weight = raw_neg / raw_pos   # cost-sensitive loss instead

            mk = lambda d, train: DataLoader(
                EndoscopyDataset(d, build_transforms(size, train, center_crop_frac)),
                batch_size=batch, shuffle=train, num_workers=workers,
                drop_last=train and len(d) > batch)

            model = make_model(name, pretrained, mode)
            model, _ = train_one(model, mk(fit_df, True), mk(val_df, False), device,
                                 epochs=epochs, lr=lr, pos_weight=pos_weight)

            p, y, idx = infer(model, mk(te_df, False), device)
            r = dict(backbone=name, repeat=rep, fold=k,
                     n_test_img=len(te_df), n_test_pat=te_df["patient_id"].nunique())
            r.update(metrics(y, p))
            r.update(patient_level(df, te, p, y))
            rows.append(r)
            del model
            print(f"    {name:22s} rep{rep} fold{k}  img_auc={r['auc']:.3f} "
                  f"pat_auc={r['pat_auc']:.3f}", flush=True)
        print(f"  {name:22s} done in {time.time()-t0:.0f}s", flush=True)
    return pd.DataFrame(rows)


def summarise(res, n_pat, folds, key="pat_auc"):
    n_test = n_pat / folds
    n_train = n_pat - n_test
    out = []
    for name, g in res.groupby("backbone", sort=False):
        a = g[key].dropna().values
        se = np.sqrt(a.var(ddof=1) * (1 / len(a) + n_test / n_train))
        row = dict(backbone=name, **{key: a.mean(),
                                     "lo": a.mean() - 1.96 * se, "hi": a.mean() + 1.96 * se})
        for c in ["auc", "pat_auc", "pat_sens", "pat_spec", "pat_bacc", "pat_brier"]:
            row[c] = g[c].mean()
        out.append(row)
    s = pd.DataFrame(out).sort_values(key, ascending=False).reset_index(drop=True)
    best = s.loc[0, "backbone"]
    piv = res.pivot_table(index=["repeat", "fold"], columns="backbone", values=key)
    comp = [dict(backbone=b,
                 delta=piv[best].mean() - piv[b].mean(),
                 p=corrected_ttest(piv[best] - piv[b], n_train, n_test)[1])
            for b in s["backbone"]]
    return s, pd.DataFrame(comp), best


def gradcam(model, dataset, device, out_png, n=8, layer=None):
    """Overlay class activation on sample images. If the heat sits on borders,
    timestamps or letterboxing rather than the airway, the model is reading
    acquisition source and the accuracy is not real."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.to(device).eval()
    if layer is None:
        convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
        if not convs:
            print("no conv layer found (transformer backbone) -- use attention rollout")
            return
        layer = convs[-1]

    feats, grads = {}, {}
    h1 = layer.register_forward_hook(lambda m, i, o: feats.__setitem__("a", o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("g", go[0]))

    idxs = np.linspace(0, len(dataset) - 1, n).astype(int)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(3 * ((n + 1) // 2), 6))
    for ax, i in zip(axes.ravel(), idxs):
        x, y, _ = dataset[i]
        x = x.unsqueeze(0).to(device)
        out = model(x).squeeze()
        model.zero_grad()
        out.backward()
        a, g = feats["a"][0], grads["g"][0]
        cam = torch.relu((a * g.mean(dim=(1, 2), keepdim=True)).sum(0)).detach().cpu().numpy()
        cam = (cam - cam.min()) / (np.ptp(cam) + 1e-8)
        img = x[0].cpu().permute(1, 2, 0).numpy()
        img = np.clip(img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]), 0, 1)
        ax.imshow(img)
        ax.imshow(np.kron(cam, np.ones((img.shape[0] // cam.shape[0] + 1,) * 2))
                  [:img.shape[0], :img.shape[1]], cmap="jet", alpha=0.42)
        ax.set_title(f"true={int(y)} p={torch.sigmoid(out).item():.2f}", fontsize=9)
        ax.axis("off")
    h1.remove(); h2.remove()
    plt.tight_layout()
    plt.savefig(out_png, dpi=130)
    print("saved", out_png)


def make_smoke_data(tmp="/tmp/smoke_images", n_pat=24, per_pat=6):
    """Synthetic images with a faint class signal, for code-path testing only."""
    from pathlib import Path
    from PIL import Image
    rng = np.random.default_rng(0)
    for lab in ["healthy", "unhealthy"]:
        Path(f"{tmp}/{lab}").mkdir(parents=True, exist_ok=True)
    for lab_i, lab in enumerate(["healthy", "unhealthy"]):
        for pid in range(n_pat):
            base = rng.integers(60, 190, 3)
            for f in range(per_pat):
                a = np.clip(base + rng.normal(0, 18, (96, 96, 3)), 0, 255)
                yy, xx = np.mgrid[0:96, 0:96]
                r = np.sqrt((yy - 48) ** 2 + (xx - 48) ** 2)
                a[..., 0] += (28 if lab_i else 0) * np.exp(-(r ** 2) / 320)
                Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).save(
                    f"{tmp}/{lab}/p{pid:03d}_{f}.png")
    return tmp


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root"); ap.add_argument("--csv")
    ap.add_argument("--backbones", nargs="+", default=DEFAULT_BACKBONES)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--mode", default="finetune",
                    choices=["finetune", "linear_probe", "last_block"])
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--center-crop-frac", type=float, default=None)
    ap.add_argument("--oversample", action="store_true",
                    help="resample the scarce class up to parity on each training "
                         "split (augmentation-based rebalancing); otherwise a "
                         "cost-sensitive loss is used")
    ap.add_argument("--device", default=None,
                    help="cuda | mps | cpu (default: autodetect)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--audit-only", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="img_results")
    ap.add_argument("--save-final", metavar="PATH",
                    help="after CV, refit one model on ALL data (with the same "
                         "rebalancing) and write it here as a .pt checkpoint, "
                         "plus a Grad-CAM panel alongside it")
    a = ap.parse_args()

    if a.smoke:
        a.root = make_smoke_data()
        a.backbones = ["resnet18"]
        a.repeats, a.folds, a.epochs, a.size, a.batch, a.workers = 1, 3, 2, 64, 16, 0
        pretrained = False
        print("SMOKE TEST -- random weights, synthetic images, numbers meaningless\n")
    else:
        pretrained = True

    df = build_index(a.root, a.csv)
    print_audit(audit(df))
    if a.audit_only:
        raise SystemExit(0)

    if a.device:
        device = a.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"device: {device} | mode: {a.mode} | oversample: {a.oversample} | "
          f"backbones: {a.backbones}\n")

    res = run(df, a.backbones, a.repeats, a.folds, a.mode, a.size, a.epochs,
              a.batch, a.lr, pretrained, a.center_crop_frac, device, a.workers,
              oversample=a.oversample)
    res.to_csv(f"{a.out}_folds.csv", index=False)
    s, comp, best = summarise(res, df["patient_id"].nunique(), a.folds)
    s.to_csv(f"{a.out}_summary.csv", index=False)

    print("\n=== patient-level AUC (mean, 95% CI, corrected variance) ===")
    for _, r in s.iterrows():
        print(f"{r['backbone']:24s} {r['pat_auc']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]  "
              f"sens={r['pat_sens']:.3f} spec={r['pat_spec']:.3f} "
              f"| image-level AUC={r['auc']:.3f}")
    print(f"\n=== corrected resampled t-test vs {best} ===")
    for _, r in comp.iterrows():
        if r["backbone"] != best:
            print(f"{r['backbone']:24s} d={r['delta']:+.3f}  p={r['p']:.3f}")
    print("\nNext: run Grad-CAM on the winning fold before believing any of this.")

    if a.save_final:
        from pathlib import Path

        from img_data import IMAGENET_MEAN, IMAGENET_STD
        name = best if len(a.backbones) > 1 else a.backbones[0]
        print(f"\n=== refitting {name} on all {len(df)} images -> {a.save_final} ===")
        inner = patient_splits(df.reset_index(drop=True), 4, 1, SEED)[0]
        fit_df = df.reset_index(drop=True).iloc[inner[2]]
        val_df = df.reset_index(drop=True).iloc[inner[3]]
        if a.oversample:
            fit_df = oversample_to_balance(fit_df, seed=SEED)
            pw = 1.0
        else:
            pw = max(len(fit_df) - fit_df["label"].sum(), 1) / max(fit_df["label"].sum(), 1)
        mk = lambda d, tr: DataLoader(
            EndoscopyDataset(d, build_transforms(a.size, tr, a.center_crop_frac)),
            batch_size=a.batch, shuffle=tr, num_workers=a.workers,
            drop_last=tr and len(d) > a.batch)
        model = make_model(name, pretrained, a.mode)
        model, best_auc = train_one(model, mk(fit_df, True), mk(val_df, False),
                                    device, epochs=a.epochs, lr=a.lr, pos_weight=pw)
        Path(a.save_final).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "backbone": name,
                    "mode": a.mode, "size": a.size,
                    "center_crop_frac": a.center_crop_frac,
                    "mean": IMAGENET_MEAN, "std": IMAGENET_STD,
                    "classes": {0: "healthy", 1: "unhealthy"},
                    "val_auc": float(best_auc)}, a.save_final)
        print(f"saved checkpoint (inner-val AUC {best_auc:.3f})")
        try:
            cam_png = str(Path(a.save_final).with_suffix(".gradcam.png"))
            gradcam(model, EndoscopyDataset(
                df.sample(min(len(df), 12), random_state=SEED),
                build_transforms(a.size, False, a.center_crop_frac)),
                device, cam_png, n=8)
        except Exception as e:  # noqa: BLE001
            print(f"grad-cam skipped: {e}")
