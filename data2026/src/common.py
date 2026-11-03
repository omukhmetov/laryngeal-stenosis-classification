"""Shared paths and data loading for the baseline pipeline.

Working directory layout (--work, kept outside the repository):
    frames/00000.jpg ...   field-of-view crops, 288x288 (prepare_frames.py)
    index.csv              frame index -> file_path in the manifest
    fov.csv                field-of-view fraction and diameter per frame
    weights/*.pth          ImageNet weights (downloaded by embed.py)
    features/emb_*.npy     frozen features (embed.py)
    results/               metrics (evaluate.py, finetune.py, summarize_ft.py)
"""
import argparse, os
import numpy as np, pandas as pd

GRADE_TO_CLASS = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2}   # normal / mild (1-2) / severe (3-4)
CLASS_NAMES = ["normal", "mild", "severe"]
MAX_FOV_FRAC = 0.9                                  # above this: no endoscope circle (frame outside the scope)


def parse_args(extra=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="path to manifest.csv from build_manifest.py")
    ap.add_argument("--work", required=True, help="working directory (frames, features, results)")
    if extra:
        extra(ap)
    a = ap.parse_args()
    for sub in ("frames", "weights", "features", "results"):
        os.makedirs(os.path.join(a.work, sub), exist_ok=True)
    return a


def image_table(manifest):
    """All image rows of the manifest, in the order used for frames/<idx>.jpg."""
    man = pd.read_csv(manifest)
    img = man[man.media == "image"].reset_index(drop=True)
    img["idx"] = img.index
    img["y"] = img.grade.map(GRADE_TO_CLASS)
    return img


def selected_frames(manifest, work):
    """Frames used in the experiments: include == True and inside the endoscope circle."""
    img = image_table(manifest)
    fov = pd.read_csv(os.path.join(work, "fov.csv"))
    assert len(fov) == len(img), "fov.csv does not match the manifest; rerun prepare_frames.py"
    img["fov_frac"] = fov.fov_frac.values
    img["fov_diam"] = fov.fov_diam_px.values
    sel = img.include.astype(bool).values & (img.fov_frac.values <= MAX_FOV_FRAC)
    return img, sel


def frame_path(work, idx):
    return os.path.join(work, "frames", f"{idx:05d}.jpg")
