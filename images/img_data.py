"""Dataset, patient-grouped splitting, augmentation and leakage auditing
for healthy vs unhealthy laryngeal endoscopy classification.

Expected layout (either works):

    images/healthy/<patient>_<frame>.jpg
    images/unhealthy/<patient>_<frame>.jpg

or a CSV with columns: path,label,patient_id

The patient id is what protects you from frame-level leakage. If your file
names do not encode it, build the CSV instead -- do not fall back to
splitting by image.
"""
import hashlib
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Grouping key ("patient"/session) parsed from the file stem. This dataset has
# two naming conventions:
#   * "001(1).JPG", "001S(3).JPG", "111.JPG"  -> case number; the trailing "S"
#     variant is the same case, another view/session, so it is merged in.
#   * "2025-02-17_212626_IMG0003.jpg"         -> acquisition date; each calendar
#     date is treated as one session. "1970-01-01" is the camera's default
#     clock (unknown provenance) and collapses into a single conservative group.
# A leading "Копия"/"Copy of" (duplicate export) is stripped before parsing.
_COPY_RE = re.compile(r"^(?:коп[ияей]+|copy of|копия)\s+", re.IGNORECASE)
_DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})")
_CASE_RE = re.compile(r"^(\d{1,4})s?", re.IGNORECASE)


def patient_from_name(stem: str) -> str:
    stem = _COPY_RE.sub("", stem.strip())
    m = _DATE_RE.match(stem)
    if m:
        return m.group(1)
    m = _CASE_RE.match(stem)
    if m:
        return f"case{int(m.group(1)):03d}"
    return stem


def build_index(root=None, csv=None) -> pd.DataFrame:
    """Return a frame with columns: path, label (0=healthy, 1=unhealthy), patient_id."""
    if csv:
        df = pd.read_csv(csv)
        missing = {"path", "label", "patient_id"} - set(df.columns)
        if missing:
            raise ValueError(f"CSV missing columns: {missing}")
        return df

    root = Path(root)
    # tolerate Healthy/ Unhealthy/ HEALTHY/ etc.
    children = {c.name.lower(): c for c in root.iterdir() if c.is_dir()}
    rows = []
    for label_name, label in [("healthy", 0), ("unhealthy", 1)]:
        folder = children.get(label_name)
        if folder is None or not folder.is_dir():
            raise FileNotFoundError(f"expected a '{label_name}' folder under {root}")
        for p in sorted(folder.rglob("*")):
            if p.suffix.lower() in IMG_EXT:
                rows.append(dict(path=str(p), label=label,
                                 patient_id=f"{label_name}:{patient_from_name(p.stem)}"))
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"no images found under {root}")
    return df


def audit(df: pd.DataFrame) -> dict:
    """Structural checks to run BEFORE training. Every one of these has sunk a paper."""
    report = {}
    report["n_images"] = len(df)
    report["n_patients"] = df["patient_id"].nunique()
    report["class_balance"] = df["label"].value_counts().to_dict()
    report["patients_per_class"] = df.groupby("label")["patient_id"].nunique().to_dict()

    per_pat = df.groupby("patient_id").size()
    report["images_per_patient"] = dict(
        min=int(per_pat.min()), median=float(per_pat.median()),
        max=int(per_pat.max()), mean=round(float(per_pat.mean()), 2))

    # A patient appearing in both classes means the id parse is wrong or the
    # same child was imaged before and after -- either way, splits break.
    both = df.groupby("patient_id")["label"].nunique()
    report["patients_in_both_classes"] = int((both > 1).sum())

    # Exact duplicate files across the whole set.
    digests = {}
    dupes = 0
    for p in df["path"]:
        try:
            h = hashlib.md5(Path(p).read_bytes()).hexdigest()
        except OSError:
            continue
        if h in digests:
            dupes += 1
        digests[h] = p
    report["exact_duplicate_files"] = dupes

    # Image size fingerprinting: if healthy and unhealthy come from different
    # equipment, resolution alone can separate the classes and the CNN will
    # learn that instead of the pathology.
    sizes = []
    sample = pd.concat([g.sample(min(len(g), 200), random_state=0)
                        for _, g in df.groupby("label")])
    for _, r in sample.iterrows():
        try:
            with Image.open(r["path"]) as im:
                sizes.append((r["label"], im.size, im.mode))
        except OSError:
            continue
    by_label = {}
    for lab in (0, 1):
        c = Counter(s for l, s, _ in sizes if l == lab)
        by_label[lab] = c.most_common(4)
    report["common_resolutions_by_class"] = by_label
    report["resolution_disjoint"] = len(
        set(dict(by_label[0])) & set(dict(by_label[1]))) == 0
    return report


def print_audit(rep: dict):
    print("=" * 62)
    print("DATASET AUDIT")
    print("=" * 62)
    print(f"images            : {rep['n_images']}")
    print(f"patients          : {rep['n_patients']}")
    print(f"class balance     : {rep['class_balance']}")
    print(f"patients per class: {rep['patients_per_class']}")
    print(f"images/patient    : {rep['images_per_patient']}")
    print(f"exact duplicates  : {rep['exact_duplicate_files']}")
    for lab, name in [(0, "healthy"), (1, "unhealthy")]:
        print(f"resolutions {name:9s}: {rep['common_resolutions_by_class'][lab]}")
    print("-" * 62)
    if rep["patients_in_both_classes"]:
        print(f"!! {rep['patients_in_both_classes']} patient ids span both classes -- "
              "check the id parse before trusting any split")
    if rep["resolution_disjoint"]:
        print("!! healthy and unhealthy have NO resolution in common. The classes are "
              "separable on acquisition metadata alone; high accuracy here means "
              "little. Crop borders and confirm with Grad-CAM.")
    if rep["images_per_patient"]["max"] > 1:
        print("   multiple frames per patient -> patient-level splitting is mandatory")
    print("=" * 62)


# --------------------------------------------------------------------------
# transforms
# --------------------------------------------------------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(size=224, train=True, center_crop_frac=None):
    """center_crop_frac: crop the central fraction first, to strip scope
    letterboxing, timestamps and device overlays that leak acquisition source."""
    from torchvision import transforms as T
    pre = []
    if center_crop_frac:
        pre.append(T.Lambda(lambda im: T.functional.center_crop(
            im, [int(min(im.size) * center_crop_frac)] * 2)))
    if train:
        return T.Compose(pre + [
            T.RandomResizedCrop(size, scale=(0.7, 1.0), ratio=(0.85, 1.18)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),          # scope orientation is arbitrary
            T.RandomRotation(25),
            T.ColorJitter(0.25, 0.25, 0.20, 0.04),  # light source variation
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            T.RandomErasing(p=0.25, scale=(0.02, 0.12)),
        ])
    return T.Compose(pre + [
        T.Resize(int(size * 1.14)), T.CenterCrop(size),
        T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


class EndoscopyDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.loc[i]
        with Image.open(r["path"]) as im:
            im = im.convert("RGB")
            x = self.transform(im)
        return x, torch.tensor(float(r["label"])), i


def oversample_to_balance(df, seed=0):
    """Duplicate minority-class rows (with replacement) until the classes are
    even. Combined with the random train-time augmentation in
    ``build_transforms``, the duplicated frames are seen under different
    crops/rotations/jitter each epoch, so this acts as augmentation of the
    scarce healthy class rather than plain repetition. Apply to the training
    split only -- never to validation or test."""
    counts = df["label"].value_counts()
    if len(counts) < 2:
        return df.reset_index(drop=True)
    n_max = int(counts.max())
    parts = []
    for lab, g in df.groupby("label"):
        if len(g) < n_max:
            g = pd.concat([g, g.sample(n_max - len(g), replace=True,
                                       random_state=seed)])
        parts.append(g)
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def patient_splits(df, n_splits=5, n_repeats=2, seed=0):
    """Grouped CV that never places a patient's frames on both sides AND
    guarantees every fold gets a share of each class.

    ``StratifiedGroupKFold`` is the usual choice, but with only ~7 healthy
    sessions its greedy assignment routinely leaves a fold with zero healthy
    patients -> ``roc_auc`` is undefined for that fold. Here each class's
    groups are shuffled and dealt round-robin into the folds instead, so a
    class with >= n_splits groups lands in every fold. Folds are still pure
    at the patient level."""
    groups = df.groupby("patient_id")["label"].first()
    idx_by_group = df.groupby("patient_id").indices  # group -> row positions
    out = []
    for rep in range(n_repeats):
        fold_of = {}
        for lab in sorted(groups.unique()):
            grps = list(groups.index[groups == lab])
            rng = np.random.default_rng(seed + rep * 97 + int(lab))
            rng.shuffle(grps)
            start = int(rng.integers(0, n_splits))
            for i, g in enumerate(grps):
                fold_of[g] = (start + i) % n_splits
        assign = np.empty(len(df), dtype=int)
        for g, pos in idx_by_group.items():
            assign[pos] = fold_of[g]
        for k in range(n_splits):
            te = np.where(assign == k)[0]
            tr = np.where(assign != k)[0]
            if len(te) == 0 or len(tr) == 0:
                continue
            assert not (set(df.iloc[tr]["patient_id"]) & set(df.iloc[te]["patient_id"])), \
                "patient leaked across the split"
            out.append((rep, k, tr, te))
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root")
    ap.add_argument("--csv")
    a = ap.parse_args()
    df = build_index(a.root, a.csv)
    print_audit(audit(df))
