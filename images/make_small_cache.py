"""Downsize every image into a flat cache so training isn't bottlenecked on
decoding 1920x1080 JPEGs. Filenames are made unique but keep the leading token
that patient_from_name() parses, so grouping is preserved.

    python3 make_small_cache.py --src . --dst images_small --max-side 320
"""
import argparse
import hashlib
from pathlib import Path

from PIL import Image

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=".")
    ap.add_argument("--dst", default="images_small")
    ap.add_argument("--max-side", type=int, default=320)
    ap.add_argument("--quality", type=int, default=90)
    a = ap.parse_args()

    src, dst = Path(a.src), Path(a.dst)
    children = {c.name.lower(): c for c in src.iterdir() if c.is_dir()}
    n_ok = n_bad = 0
    for label in ("healthy", "unhealthy"):
        folder = children[label]
        out = dst / label
        out.mkdir(parents=True, exist_ok=True)
        for p in sorted(folder.rglob("*")):
            if p.suffix.lower() not in IMG_EXT:
                continue
            # keep the parseable leading token; disambiguate with a path hash
            tag = hashlib.md5(str(p.relative_to(folder)).encode()).hexdigest()[:8]
            target = out / f"{p.stem}__{tag}.jpg"
            if target.exists():
                continue
            try:
                with Image.open(p) as im:
                    im = im.convert("RGB")
                    w, h = im.size
                    s = a.max_side / max(w, h)
                    if s < 1:
                        im = im.resize((round(w * s), round(h * s)), Image.LANCZOS)
                    im.save(target, "JPEG", quality=a.quality)
                n_ok += 1
            except Exception as e:  # noqa: BLE001
                print(f"skip {p}: {e}")
                n_bad += 1
    print(f"wrote {n_ok} images to {dst}  ({n_bad} skipped)")


if __name__ == "__main__":
    main()
