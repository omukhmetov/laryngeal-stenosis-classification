# Data 2026: manifest and honest 3-class image baseline

Code for the data audit and the first image baseline on the 2025 endoscopy set
(normal vs. Cotton-Myer grade 1–4), part of grant AP27510665
(AI for early diagnosis of cicatricial laryngeal stenosis in children).

**The images and the manifest are not in this repository.** They contain patient
data and stay with the clinical team. Only code and aggregate metrics are published.

## Main findings (2026-11-03)

- With patient-level splits, grade cannot be read from a whole frame:
  AUC mild (1–2) vs. severe (3–4) 0.53–0.67, normal vs. stenosis 0.63–0.73.
- A frame-level random split inflates every metric by 0.22–0.39 (AUC up to 0.87–0.89): leakage between frames of the same examination.
- Acquisition confound: 45 % of normal frames have a small field of view vs. 2–9 % of stenosis frames.
  Colour alone or the circle diameter alone reach AUC 0.60–0.66 for normal vs. stenosis.
- Fine-tuning EfficientNet-B0 is no better than frozen features (86 independent groups).

Full tables: [`results/baseline_3class/`](results/baseline_3class/).

| Model (session level, honest split) | Balanced acc. | QWK | AUC normal/stenosis | AUC mild/severe |
|---|---|---|---|---|
| EfficientNet-B0, frozen | 0.51 | 0.30 | 0.63 | 0.67 |
| ResNet-50, frozen | 0.42 | 0.18 | 0.70 | 0.56 |
| ViT-B/16, frozen | 0.44 | 0.29 | 0.73 | 0.53 |
| EfficientNet-B0, fine-tuned (95 % CI) | 0.42 (0.32–0.51) | 0.23 | 0.70 (0.58–0.80) | 0.56 (0.43–0.71) |
| Control: FOV diameter | 0.47 | 0.24 | 0.60 | 0.54 |
| Control: colour/brightness | 0.34 | 0.08 | 0.66 | 0.39 |

Chance level: balanced accuracy 0.33, AUC 0.50.

## Data layout expected

```
Data 2026/
  Здоров 2025г/<patient folder or loose frames>
  Нездоров 2025г/1 степень/ ... 4 степень/
```

## Pipeline

```bash
cd data2026
pip install -r requirements.txt
DATA="/path/to/Data 2026"
M="$DATA/_manifest/manifest.csv"

python src/build_manifest.py "$DATA"                                   # 1. manifest (written next to the data)
python src/prepare_frames.py --data-root "$DATA" --manifest "$M" --work ./work   # 2. crop to field of view
python src/embed.py     --manifest "$M" --work ./work                  # 3. frozen ImageNet features
python src/evaluate.py  --manifest "$M" --work ./work --repeats 5      # 4. grouped CV + controls + leaky split
python src/finetune.py  --manifest "$M" --work ./work --epochs 10      # 5. fine-tuning (GPU if available)
python src/summarize_ft.py --manifest "$M" --work ./work               # 6. metrics + bootstrap CI
```

| Script | What it does |
|---|---|
| `build_manifest.py` | One row per file: patient ID (pseudonymised P001…), session, `split_group`, duplicates (MD5), label conflicts, flags, `include` |
| `prepare_frames.py` | Finds the endoscope circle, crops to it, resizes to 288×288; writes `fov.csv` |
| `embed.py` | EfficientNet-B0, ResNet-50, ViT-B/16 features (timm, weights from timm GitHub releases) |
| `evaluate.py` | PCA + logistic regression; StratifiedGroupKFold on `split_group` × repeats; random-split comparison; colour and FOV controls |
| `finetune.py` | EfficientNet-B0 fine-tuning, 5 grouped folds, fixed epochs |
| `summarize_ft.py` | Frame- and session-level metrics, bootstrap CI over groups |
| `common.py`, `metrics.py` | Shared loading and metrics |

### Key manifest fields

- `split_group` — **the only column to split train/test on.** It joins frames of one patient,
  identical copies of a file, and loose frames with a reset camera clock (union-find).
- `session_id` — one examination; loose frames are clustered by capture time (gap > 10 min or frame counter reset).
- `include` — provisional selection pending clinician review: no duplicates, no conflicting labels, no "unclear" folders.

## Classes

normal = grade 0 · mild = grades 1–2 · severe = grades 3–4
(grades 2 and 4 alone come from one or two children each).

## Next steps

Lumen segmentation (measure % obstruction, as Cotton-Myer defines it) instead of whole-frame
classification; more patients (2023–2024 images, linking loose frames via the endoscopy log);
temporal validation of the normal-vs-abnormal model from the MICAD 2026 abstract.
