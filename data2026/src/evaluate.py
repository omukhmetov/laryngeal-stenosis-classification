"""Baseline: normal / mild (1-2) / severe (3-4) from frozen features.

Features -> StandardScaler -> PCA(128) -> logistic regression (balanced class weights).
Repeated StratifiedGroupKFold on `split_group` (honest), plus a frame-level random
split for comparison (leaky). Controls: HSV colour histogram only, and the
field-of-view diameter only.

Usage: python src/evaluate.py --manifest .../manifest.csv --work ./work [--repeats 5]
"""
import os, warnings
import numpy as np, pandas as pd
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from common import parse_args, selected_frames, frame_path
from metrics import metrics, session_level

warnings.filterwarnings("ignore")
FEATURES = [("efficientnet_b0", "EfficientNet-B0"), ("resnet50", "ResNet-50"), ("vit_base_patch16_224", "ViT-B/16")]
LEAKY_DEMO = ("EfficientNet-B0", "ViT-B/16")


def color_hist(img, work):
    p = os.path.join(work, "features", "emb_colorhist.npy")
    if os.path.exists(p):
        return np.load(p)
    F = []
    for i in img.idx:
        a = np.asarray(Image.open(frame_path(work, i)).convert("HSV").resize((128, 128)))
        mask = a[..., 2] > 20                                   # ignore the black border
        F.append(np.concatenate([np.histogram(a[..., c][mask], bins=16, range=(0, 255), density=True)[0] for c in range(3)]))
    F = np.array(F)
    np.save(p, F)
    return F


def model(dim):
    steps = [StandardScaler()] + ([PCA(128, random_state=0)] if dim > 256 else [])
    return make_pipeline(*steps, LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000))


if __name__ == "__main__":
    a = parse_args(lambda ap: (ap.add_argument("--repeats", type=int, default=5), ap.add_argument("--folds", type=int, default=5)))
    img, sel = selected_frames(a.manifest, a.work)
    d = img[sel].reset_index(drop=True)
    y, groups, sess = d.y.values, d.split_group.values, d.session_id.values

    feats = {"Colour/brightness (control)": color_hist(img, a.work),
             "FOV diameter (control)": img[["fov_diam"]].values.astype(float)}
    for n, lab in FEATURES:
        p = os.path.join(a.work, "features", f"emb_{n}.npy")
        if os.path.exists(p):
            feats[lab] = np.load(p)

    rows, cms = [], {}
    for lab, X0 in feats.items():
        X = X0[sel]
        for split in ("group", "random"):
            if split == "random" and lab not in LEAKY_DEMO:
                continue
            for rep in range(a.repeats):
                cv = (StratifiedGroupKFold(a.folds, shuffle=True, random_state=rep) if split == "group"
                      else StratifiedKFold(a.folds, shuffle=True, random_state=rep))
                P = np.zeros((len(y), 3))
                for tr, te in cv.split(X, y, groups):
                    P[te] = model(X.shape[1]).fit(X[tr], y[tr]).predict_proba(X[te])
                ys, Ps, _ = session_level(P, y, sess)
                rows += [dict(features=lab, split=split, repeat=rep, level="frame", **metrics(y, P)),
                         dict(features=lab, split=split, repeat=rep, level="session", **metrics(ys, Ps))]
                if split == "group":
                    cms[lab] = cms.get(lab, 0) + confusion_matrix(ys, Ps.argmax(1), labels=[0, 1, 2])
        print(lab, "done", flush=True)

    res = pd.DataFrame(rows)
    cols = ["bal_acc", "macro_f1", "qwk", "auc_macro", "auc_norm_vs_sten", "auc_mild_vs_severe"]
    summ = res.groupby(["features", "split", "level"])[cols].agg(["mean", "std"])
    out = os.path.join(a.work, "results")
    res.to_csv(os.path.join(out, "cv_all_repeats.csv"), index=False)
    summ.to_csv(os.path.join(out, "cv_summary.csv"))
    pd.set_option("display.width", 250)
    print(summ.round(3).to_string())
    for k, v in cms.items():
        print(k, "(session-level confusion, summed over repeats)\n", v)
