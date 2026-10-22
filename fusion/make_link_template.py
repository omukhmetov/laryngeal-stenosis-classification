#!/usr/bin/env python3
"""
Step 1 of the image + clinical fusion study: build the image -> patient linking sheet.

The endoscopy file names carry a capture timestamp but no patient ID, so we
cannot join images to DATA.xlsx automatically. This script groups the
Unhealthy frames into candidate EXAMS (frames from the same day with no gap
longer than --gap minutes) and writes one row per exam to
    fusion/link_template.xlsx
with an empty `patient_id` column. Someone with access to the clinic records
fills in `patient_id` using the DATA.xlsx format (e.g. 2024-2). Leave it blank
if unknown. Several exams can share one patient_id. To split an exam that
actually covers two patients, duplicate the row and fix `first_file`/`last_file`.

Hints written per row: capture date, time range, parent folder name
(some folders already contain a surname and date), number of frames.
Note: 1970-01-01 dates mean the camera clock was reset, so the date is
meaningless there; rely on the folder name / time order.

Run from the Stenosis folder:
    python3 fusion/make_link_template.py
Needs only pandas + openpyxl.
"""
from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})_(\d{6})")
CASE_RE = re.compile(r"^(\d{1,4})s?", re.IGNORECASE)


def parse(p: Path):
    m = TS_RE.match(p.stem)
    if m:
        ts = datetime.strptime(m.group(1) + m.group(2), "%Y-%m-%d%H%M%S")
        return "ts", ts
    m = CASE_RE.match(p.stem)
    if m:
        return "case", int(m.group(1))
    return "other", p.stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="images/Unhealthy")
    ap.add_argument("--gap", type=float, default=20.0,
                    help="minutes of silence that start a new exam (default 20)")
    ap.add_argument("--out", default="fusion/link_template.xlsx")
    a = ap.parse_args()

    root = Path(a.root)
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXT)
    rows = []
    for p in files:
        kind, key = parse(p)
        rows.append(dict(path=str(p.relative_to(root)), folder=str(p.parent.relative_to(root)),
                         kind=kind, key=key))
    df = pd.DataFrame(rows)

    exams = []
    # timestamped frames: split by folder, then by time gaps
    ts = df[df.kind == "ts"].copy()
    for folder, g in ts.groupby("folder"):
        g = g.sort_values("key")
        prev, cur = None, []
        for _, r in g.iterrows():
            if prev is not None and (r.key - prev).total_seconds() / 60 > a.gap:
                exams.append((folder, cur)); cur = []
            cur.append(r); prev = r.key
        if cur:
            exams.append((folder, cur))
    # case-numbered frames (001, 001S, 005S(9) ...): one exam per case number per folder
    cs = df[df.kind == "case"]
    for (folder, key), g in cs.groupby(["folder", "key"]):
        exams.append((folder, [r for _, r in g.sort_values("path").iterrows()]))
    for _, r in df[df.kind == "other"].iterrows():
        exams.append((r.folder, [r]))

    out = []
    for i, (folder, rs) in enumerate(exams, 1):
        k0, k1 = rs[0].key, rs[-1].key
        is_ts = rs[0].kind == "ts"
        out.append({
            "exam_id": f"E{i:03d}",
            "patient_id": "",
            "n_frames": len(rs),
            "capture_date": k0.strftime("%Y-%m-%d") if is_ts else "",
            "time_range": f"{k0:%H:%M}-{k1:%H:%M}" if is_ts else "",
            "clock_reset": bool(is_ts and k0.year < 2000),
            "case_no": "" if is_ts else str(k0),
            "folder": "" if folder == "." else folder,
            "first_file": rs[0].path,
            "last_file": rs[-1].path,
            "notes": "",
        })
    ex = pd.DataFrame(out)

    # long table: every image with its exam_id (the fusion script uses this)
    img_rows = []
    for (folder, rs), e in zip(exams, ex.exam_id):
        img_rows += [{"exam_id": e, "path": r.path} for r in rs]
    imgs = pd.DataFrame(img_rows)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(a.out) as w:
        ex.to_excel(w, sheet_name="exams", index=False)
        imgs.to_excel(w, sheet_name="images", index=False)
    print(f"{len(files)} frames -> {len(ex)} candidate exams "
          f"({int(ex.clock_reset.sum())} with reset camera clock, "
          f"{(ex.case_no != '').sum()} case-numbered)")
    print(f"wrote {a.out}. Fill the 'patient_id' column on the 'exams' sheet, "
          "then save as fusion/link.xlsx")


if __name__ == "__main__":
    main()
