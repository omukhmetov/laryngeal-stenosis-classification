"""Run a checkpoint from ``img_benchmark.py --save-final`` on new images.

    python3 img_predict.py --ckpt model.pt --images path/to/folder
    python3 img_predict.py --ckpt model.pt --images a.jpg b.jpg --csv out.csv

Prints one probability per image (P(unhealthy)); with --group it also
aggregates to one prediction per patient/session using the same leading-token
rule as training.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from img_data import IMG_EXT, build_transforms, patient_from_name


def load_model(ckpt_path, device):
    import timm
    ck = torch.load(ckpt_path, map_location=device)
    model = timm.create_model(ck["backbone"], pretrained=False, num_classes=1)
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    return model, ck


def gather(images):
    paths = []
    for item in images:
        p = Path(item)
        if p.is_dir():
            paths += [q for q in sorted(p.rglob("*")) if q.suffix.lower() in IMG_EXT]
        elif p.suffix.lower() in IMG_EXT:
            paths.append(p)
    return paths


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--images", nargs="+", required=True,
                    help="image files and/or folders")
    ap.add_argument("--csv", help="write per-image predictions here")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--group", action="store_true",
                    help="also aggregate to one prediction per patient/session")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    device = a.device or ("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    model, ck = load_model(a.ckpt, device)
    tf = build_transforms(ck["size"], train=False,
                          center_crop_frac=ck.get("center_crop_frac"))

    paths = gather(a.images)
    if not paths:
        raise SystemExit("no images found")

    rows = []
    for p in paths:
        try:
            with Image.open(p) as im:
                x = tf(im.convert("RGB")).unsqueeze(0).to(device)
            prob = torch.sigmoid(model(x).squeeze()).item()
        except Exception as e:  # noqa: BLE001
            print(f"skip {p}: {e}")
            continue
        rows.append(dict(path=str(p), patient_id=patient_from_name(p.stem),
                         p_unhealthy=round(prob, 4),
                         pred="unhealthy" if prob >= a.threshold else "healthy"))
    df = pd.DataFrame(rows)
    pd.set_option("display.max_rows", None)
    print(df.to_string(index=False))
    if a.csv:
        df.to_csv(a.csv, index=False)
        print(f"\nwrote {a.csv}")

    if a.group and len(df):
        g = df.groupby("patient_id")["p_unhealthy"].mean().reset_index()
        g["pred"] = np.where(g["p_unhealthy"] >= a.threshold, "unhealthy", "healthy")
        print("\n=== per patient/session (mean probability) ===")
        print(g.to_string(index=False))


if __name__ == "__main__":
    main()
