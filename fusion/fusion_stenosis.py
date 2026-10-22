#!/usr/bin/env python3
"""
Step 2: image + clinical fusion for stenosis severity (grade 3-4 vs 1-2).

Question: do endoscopic images add information about severity beyond the
clinical risk factors in DATA.xlsx? Four models are compared on the SAME
patients with the SAME patient-level folds:

  A  clinical   CatBoost on the DATA.xlsx variables
  B  image      logistic regression on PCA of frozen ResNet-18 embeddings
                (frames averaged per patient)
  C  late       average of A and B probabilities
  D  early      CatBoost on clinical variables + image PCs

Everything that learns from data (imputation medians, PCA, scaler, models) is
fitted inside each training fold only. One row = one patient, so a stratified
split of rows is a patient-level split.

Reported: AUC, balanced accuracy, Brier per fold (repeated stratified k-fold),
Nadeau-Bengio corrected t-test for each model vs clinical-only,
a label-permutation test for image-only, out-of-fold predictions, ROC plot,
and feature importance of the fused model.

Run from the Stenosis folder:
    python3 fusion/fusion_stenosis.py                       # needs fusion/link.xlsx
    python3 fusion/fusion_stenosis.py --crop 0.8            # border-crop sensitivity
    python3 fusion/fusion_stenosis.py --demo                # smoke test, fake links (numbers meaningless)
Needs: pip install torch torchvision catboost scikit-learn pandas openpyxl matplotlib pillow scipy
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
SEED = 42
ID_COL = "Patient №№"
TARGET = "Grade of stenosis"
DROP_COLS = ["№№", "Patient №№", "Year", TARGET]
MISSING = {"", "no data", "not reported", "nan", "none", "<na>"}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MODELS = ["clinical", "image", "late", "early"]


# ----------------------------------------------------------------- clinical data
def load_clinical(path: str) -> pd.DataFrame:
    """Same cleaning rules as cat/catboost_stenosis.py."""
    df = pd.read_excel(path, header=1)
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET]).copy()
    df[TARGET] = df[TARGET].astype(int)
    df[ID_COL] = df[ID_COL].astype(str).str.strip()

    def age(v):
        s = str(v).strip().lower()
        if s == "up to 1":
            return 0.5
        try:
            return float(s)
        except ValueError:
            return np.nan

    df["Age at primary intubation (months)"] = df["Age at primary intubation (months)"].map(age)
    for c in ["re-intubation", "mother's pregnancy №", "childbirth №",
              "Birth weight, gramm", "Height at birth, cm"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in df.columns:
        if df[c].dtype == object or str(df[c].dtype).startswith("string"):
            s = df[c].astype("string").str.strip()
            df[c] = s.mask(s.str.lower().isin(MISSING), pd.NA)
    df[ID_COL] = df[ID_COL].astype(str)
    return df


def split_features(df: pd.DataFrame):
    X = df.drop(columns=[c for c in DROP_COLS if c in df])
    cat = [c for c in X if not pd.api.types.is_numeric_dtype(X[c])]
    num = [c for c in X if c not in cat]
    return X, cat, num


def impute(Xtr: pd.DataFrame, Xte: pd.DataFrame, cat, num):
    Xtr, Xte = Xtr.copy(), Xte.copy()
    med = Xtr[num].median()
    Xtr[num], Xte[num] = Xtr[num].fillna(med), Xte[num].fillna(med)
    for c in cat:
        Xtr[c] = Xtr[c].astype(object).where(Xtr[c].notna(), "missing").astype(str)
        Xte[c] = Xte[c].astype(object).where(Xte[c].notna(), "missing").astype(str)
    return Xtr, Xte


# ----------------------------------------------------------------- linking
def load_links(link_path: str) -> pd.DataFrame:
    """Returns image path (relative to images root) -> patient_id."""
    ex = pd.read_excel(link_path, sheet_name="exams", dtype=str)
    im = pd.read_excel(link_path, sheet_name="images", dtype=str)
    ex["patient_id"] = ex["patient_id"].fillna("").str.strip()
    ex = ex[ex.patient_id != ""]
    return im.merge(ex[["exam_id", "patient_id"]], on="exam_id")


# ----------------------------------------------------------------- image embeddings
def embed_images(paths: list[Path], weights: str | None, crop: float | None,
                 cache: Path) -> dict[str, np.ndarray]:
    import torch
    import torch.nn as nn
    from PIL import Image, ImageFile
    from torchvision import models, transforms as T
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    done: dict[str, np.ndarray] = {}
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        done = dict(zip(z["paths"].tolist(), z["emb"]))
    todo = [p for p in paths if str(p) not in done]
    if todo:
        dev = ("cuda" if torch.cuda.is_available() else
               "mps" if torch.backends.mps.is_available() else "cpu")
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        if weights:
            ck = torch.load(weights, map_location="cpu")
            sd = {k: v for k, v in ck.get("state_dict", ck).items() if not k.startswith("fc.")}
            print("loaded backbone from", weights, m.load_state_dict(sd, strict=False))
        m.fc = nn.Identity()
        m.eval().to(dev)
        pre = [T.Lambda(lambda im: T.functional.center_crop(
            im, [int(min(im.size) * crop)] * 2))] if crop else []
        tf = T.Compose(pre + [T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                              T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        print(f"embedding {len(todo)} images on {dev} ...")
        with torch.no_grad():
            for i in range(0, len(todo), 32):
                batch = todo[i:i + 32]
                x = torch.stack([tf(Image.open(p).convert("RGB")) for p in batch]).to(dev)
                # horizontal-flip TTA
                e = (m(x) + m(torch.flip(x, dims=[3]))) / 2
                for p, v in zip(batch, e.cpu().numpy()):
                    done[str(p)] = v.astype(np.float32)
        np.savez(cache, paths=np.array(list(done)), emb=np.stack(list(done.values())))
    return {str(p): done[str(p)] for p in paths}


def patient_embeddings(links: pd.DataFrame, root: Path, args) -> dict[str, np.ndarray]:
    links = links.assign(full=[root / p for p in links.path])
    missing = [p for p in links.full if not p.exists()]
    if missing:
        print(f"WARNING: {len(missing)} linked images not found, e.g. {missing[0]}")
        links = links[[p.exists() for p in links.full]]
    tag = hashlib.md5(f"{args.weights}|{args.crop}".encode()).hexdigest()[:8]
    emb = embed_images(list(links.full), args.weights, args.crop,
                       Path(args.out) / f"embeddings_{tag}.npz")
    out = {}
    for pid, g in links.groupby("patient_id"):
        E = np.stack([emb[str(p)] for p in g.full])
        E = E / np.linalg.norm(E, axis=1, keepdims=True)
        out[pid] = E.mean(0)
    return out


# ----------------------------------------------------------------- models
def catboost(Xtr, ytr, cat, seed):
    from catboost import CatBoostClassifier
    m = CatBoostClassifier(iterations=400, learning_rate=0.03, depth=4, l2_leaf_reg=5,
                           auto_class_weights="Balanced", random_seed=seed,
                           verbose=0, allow_writing_files=False, thread_count=-1)
    m.fit(Xtr, ytr, cat_features=cat)
    return m


def image_block(Etr, Ete, n_pc, seed):
    sc = StandardScaler().fit(Etr)
    pca = PCA(n_components=min(n_pc, len(Etr) - 1), random_state=seed).fit(sc.transform(Etr))
    return pca.transform(sc.transform(Etr)), pca.transform(sc.transform(Ete))


def fit_image_lr(Ztr, ytr):
    return LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000).fit(Ztr, ytr)


def run_cv(X, E, y, cat, num, args, models=MODELS, seed=SEED):
    rskf = RepeatedStratifiedKFold(n_splits=args.folds, n_repeats=args.repeats, random_state=seed)
    rows, oof = [], {m: np.zeros((args.repeats, len(y))) for m in models}
    for k, (tr, te) in enumerate(rskf.split(X, y)):
        r = k // args.folds
        Xtr, Xte = impute(X.iloc[tr], X.iloc[te], cat, num)
        Ztr, Zte = image_block(E[tr], E[te], args.pcs, seed + k)
        p = {}
        if "clinical" in models or "late" in models:
            p["clinical"] = catboost(Xtr, y[tr], cat, seed + k).predict_proba(Xte)[:, 1]
        if "image" in models or "late" in models:
            p["image"] = fit_image_lr(Ztr, y[tr]).predict_proba(Zte)[:, 1]
        if "late" in models:
            p["late"] = (p["clinical"] + p["image"]) / 2
        if "early" in models:
            pcs = [f"img_pc{i + 1}" for i in range(Ztr.shape[1])]
            Ftr = pd.concat([Xtr.reset_index(drop=True), pd.DataFrame(Ztr, columns=pcs)], axis=1)
            Fte = pd.concat([Xte.reset_index(drop=True), pd.DataFrame(Zte, columns=pcs)], axis=1)
            p["early"] = catboost(Ftr, y[tr], cat, seed + k).predict_proba(Fte)[:, 1]
        for m in models:
            oof[m][r, te] = p[m]
            rows.append(dict(repeat=r, fold=k % args.folds, model=m,
                             auc=roc_auc_score(y[te], p[m]),
                             bal_acc=balanced_accuracy_score(y[te], p[m] >= 0.5),
                             brier=brier_score_loss(y[te], p[m])))
    return pd.DataFrame(rows), oof


def nadeau_bengio(diff: np.ndarray, n_train: int, n_test: int):
    """Corrected resampled t-test (Nadeau & Bengio 2003) for repeated k-fold."""
    J = len(diff)
    var = np.var(diff, ddof=1)
    if var == 0:
        return np.nan, np.nan
    t = diff.mean() / np.sqrt((1 / J + n_test / n_train) * var)
    return t, 2 * stats.t.sf(abs(t), df=J - 1)


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="cat/DATA.xlsx")
    ap.add_argument("--link", default="fusion/link.xlsx")
    ap.add_argument("--images", default="images/Unhealthy")
    ap.add_argument("--weights", default=None, help="optional classifier.py model.pt backbone")
    ap.add_argument("--crop", type=float, default=None, help="center-crop fraction, e.g. 0.8")
    ap.add_argument("--pcs", type=int, default=8, help="image principal components")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--perm", type=int, default=200, help="label permutations for image-only test (0=off)")
    ap.add_argument("--out", default="fusion/results")
    ap.add_argument("--demo-n", type=int, default=90)
    ap.add_argument("--demo", action="store_true",
                    help="random fake links + random embeddings, to check the pipeline runs")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    df = load_clinical(args.data)
    if args.demo:
        print("DEMO MODE: fake image links and random embeddings. Numbers are meaningless.")
        rng = np.random.default_rng(SEED)
        pids = rng.choice(df[ID_COL].unique(), size=min(args.demo_n, df[ID_COL].nunique()), replace=False)
        pemb = {p: rng.normal(size=512).astype(np.float32) for p in pids}
    else:
        if not Path(args.link).exists():
            sys.exit(f"{args.link} not found. Run fusion/make_link_template.py, fill patient_id, "
                     "save as fusion/link.xlsx. (Or try --demo.)")
        links = load_links(args.link)
        all_ids = set(pd.read_excel(args.data, header=1)[ID_COL].astype(str).str.strip())
        unknown = sorted(set(links.patient_id) - all_ids)
        if unknown:
            print(f"WARNING: {len(unknown)} patient_id(s) in link file not in DATA.xlsx: {unknown[:10]}")
        pemb = patient_embeddings(links, Path(args.images), args)

    # cohort = graded patients that have images
    df = df[df[ID_COL].isin(pemb)].drop_duplicates(ID_COL).reset_index(drop=True)
    y = (df[TARGET] >= 3).astype(int).to_numpy()
    E = np.stack([pemb[p] for p in df[ID_COL]])
    X, cat, num = split_features(df)

    cohort = (f"linked patients with images: {len(pemb)}\n"
              f"of those with a known grade: {len(df)}\n"
              f"  mild (grade 1-2):   {(y == 0).sum()}\n"
              f"  severe (grade 3-4): {(y == 1).sum()}\n"
              f"clinical features: {len(cat)} categorical, {len(num)} numeric; image PCs: {args.pcs}\n")
    print(cohort)
    (out / "cohort.txt").write_text(cohort + ("DEMO MODE\n" if args.demo else ""))
    if min((y == 0).sum(), (y == 1).sum()) < args.folds * 2:
        sys.exit("too few patients in the minority class for this many folds")
    if len(df) < 40:
        print("NOTE: fewer than 40 patients. Expect very wide intervals; report as a feasibility study.")

    folds, oof = run_cv(X, E, y, cat, num, args)
    folds.to_csv(out / "folds.csv", index=False)

    n_te = len(y) // args.folds
    n_tr = len(y) - n_te
    summ = []
    for m in MODELS:
        f = folds[folds.model == m]
        row = dict(model=m)
        for met in ["auc", "bal_acc", "brier"]:
            row[f"{met}_mean"] = f[met].mean(); row[f"{met}_sd"] = f[met].std()
        pooled = [roc_auc_score(y, oof[m][r]) for r in range(args.repeats)]
        row["auc_pooled_oof"] = float(np.mean(pooled))
        if m != "clinical":
            d = f.auc.to_numpy() - folds[folds.model == "clinical"].auc.to_numpy()
            t, p = nadeau_bengio(d, n_tr, n_te)
            row.update(delta_auc_vs_clinical=d.mean(), t_nb=t, p_nb=p)
        summ.append(row)
    summ = pd.DataFrame(summ)

    # permutation test: is image-only better than chance?
    if args.perm:
        obs = summ.loc[summ.model == "image", "auc_mean"].item()
        a1 = argparse.Namespace(**{**vars(args), "repeats": 1})
        rng = np.random.default_rng(SEED)
        null = []
        for i in range(args.perm):
            yp = rng.permutation(y)
            fp, _ = run_cv(X, E, yp, cat, num, a1, models=["image"], seed=SEED + i)
            null.append(fp.auc.mean())
        p_perm = (1 + np.sum(np.array(null) >= obs)) / (1 + args.perm)
        summ.loc[summ.model == "image", "p_perm"] = p_perm
        print(f"image-only permutation p = {p_perm:.4f} ({args.perm} permutations)")

    summ.to_csv(out / "summary.csv", index=False)
    pd.DataFrame({"patient_id": df[ID_COL], "grade": df[TARGET], "y_severe": y,
                  **{f"p_{m}": oof[m].mean(0) for m in MODELS}}).to_csv(out / "oof_predictions.csv", index=False)

    # feature importance of the fused model, refit on everyone
    Xa, _ = impute(X, X, cat, num)
    Za, _ = image_block(E, E, args.pcs, SEED)
    pcs = [f"img_pc{i + 1}" for i in range(Za.shape[1])]
    Fa = pd.concat([Xa.reset_index(drop=True), pd.DataFrame(Za, columns=pcs)], axis=1)
    fm = catboost(Fa, y, cat, SEED)
    imp = pd.DataFrame({"feature": Fa.columns, "importance": fm.get_feature_importance()}) \
        .sort_values("importance", ascending=False)
    imp.to_csv(out / "fused_feature_importance.csv", index=False)
    img_share = imp[imp.feature.str.startswith("img_pc")].importance.sum() / imp.importance.sum()

    # ROC plot
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 5))
        for m in MODELS:
            pm = oof[m].mean(0)
            fpr, tpr, _ = roc_curve(y, pm)
            ax.plot(fpr, tpr, label=f"{m} (AUC {roc_auc_score(y, pm):.2f})")
        ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=0.8)
        ax.set(xlabel="1 - specificity", ylabel="sensitivity",
               title="Severe vs mild stenosis (out-of-fold)")
        ax.legend(loc="lower right", frameon=False)
        fig.tight_layout(); fig.savefig(out / "roc.png", dpi=200); plt.close(fig)
    except Exception as e:
        print("plot skipped:", e)

    pd.set_option("display.width", 160)
    print("\n", summ.round(3).to_string(index=False))
    print(f"\nimage PCs share of fused-model importance: {img_share:.1%}")
    print(f"results in {out}/")


if __name__ == "__main__":
    main()
