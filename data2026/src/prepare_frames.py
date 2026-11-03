"""Crop every image to the endoscope's field of view and resize to 288x288.

Why: in this data set 45 % of normal frames were taken with a small field of view
(another scope or zoom) versus 2-9 % of stenosis frames, so a model can learn the
circle size instead of the anatomy. Cropping to the circle removes that cue.

Usage:
    python src/prepare_frames.py --data-root "/path/to/Data 2026" \
        --manifest "/path/to/Data 2026/_manifest/manifest.csv" --work ./work
"""
import io, os
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
from PIL import Image
from common import parse_args, image_table, frame_path

SIZE = 288
THRESH = 25          # grey level separating the black border from the image


def crop_to_fov(path):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    small = np.asarray(im.convert("L").resize((w // 8, h // 8))) > THRESH
    frac = float(small.mean())
    ys, xs = np.where(small)
    if 0.02 <= frac <= 0.9 and len(xs):
        x0, x1 = np.percentile(xs, [0.5, 99.5]) * 8
        y0, y1 = np.percentile(ys, [0.5, 99.5]) * 8
        cx, cy, r = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0) / 2
        box = (int(cx - r), int(cy - r), int(cx + r), int(cy + r))
    else:                                   # no circle found: central square
        s = min(w, h)
        box = ((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)
    return im.crop(box).resize((SIZE, SIZE), Image.BICUBIC), frac, box[2] - box[0]


def work_item(job):
    idx, src, dst = job
    im, frac, diam = crop_to_fov(src)
    im.save(dst, "JPEG", quality=92)
    return idx, frac, diam


if __name__ == "__main__":
    a = parse_args(lambda ap: ap.add_argument("--data-root", required=True))
    img = image_table(a.manifest)
    jobs = [(i, os.path.join(a.data_root, p), frame_path(a.work, i)) for i, p in zip(img.idx, img.file_path)]
    with ProcessPoolExecutor() as ex:
        rows = list(ex.map(work_item, jobs, chunksize=16))
    pd.DataFrame(rows, columns=["idx", "fov_frac", "fov_diam_px"]).to_csv(os.path.join(a.work, "fov.csv"), index=False)
    img[["idx", "file_path"]].to_csv(os.path.join(a.work, "index.csv"), index=False)
    print(f"{len(rows)} frames -> {os.path.join(a.work, 'frames')}")
