# Healthy vs unhealthy laryngeal endoscopy — CNN backbone benchmark

Patient-level cross-validated comparison of ImageNet-pretrained backbones,
with the leakage and provenance safeguards built into the protocol rather
than bolted on afterwards.

## Files
- `img_data.py` — dataset index, audit, augmentation, patient-grouped splits
- `img_benchmark.py` — training, CV, metrics, statistics, Grad-CAM

## Install
```bash
pip install torch torchvision timm scikit-learn pandas matplotlib pillow scipy
```

## Data layout
```
images/
  healthy/    <patient>_<frame>.jpg
  unhealthy/  <patient>_<frame>.jpg
```
The patient ID is parsed from the leading token of the filename
(`PATIENT_RE` in `img_data.py` — adjust to your convention). If filenames
don't encode it, supply a CSV with columns `path,label,patient_id` and pass
`--csv`. Do not fall back to splitting by image.

## Run
```bash
# 1. Audit first — read the warnings before training anything
python3 img_benchmark.py --root images --audit-only

# 2. Full backbone comparison
python3 img_benchmark.py --root images --repeats 2 --folds 5 --mode finetune

# 3. Frozen-backbone variant (often better at small n)
python3 img_benchmark.py --root images --mode linear_probe --out img_probe

# 4. Border-crop sensitivity analysis (see below)
python3 img_benchmark.py --root images --center-crop-frac 0.80 --out img_crop
```

## Backbones
`resnet18`, `resnet50`, `efficientnet_b0`, `densenet121`, `convnext_tiny`,
`vit_small_patch16_224` — any timm model name works via `--backbones`.

## What the protocol enforces
- **StratifiedGroupKFold on patient ID**, with an assertion that no patient
  crosses a fold. The inner early-stopping split is patient-grouped too.
- **Patient-level metrics** (mean probability per patient) reported alongside
  image-level. The patient-level number is the one that means something.
- **Nadeau–Bengio corrected resampled t-test** for backbone comparison;
  uncorrected tests across overlapping CV folds are anticonservative.
- **Dataset audit** flags patients spanning both classes, exact duplicate
  files, and disjoint resolutions between classes.
- **Grad-CAM** for confirming the model looks at the airway.

## Two results to report regardless of outcome
1. **Border-crop sensitivity.** Run with and without `--center-crop-frac 0.80`.
   If AUC drops sharply when overlays and letterboxing are cropped away, the
   model was reading acquisition source, not pathology.
2. **Grad-CAM panels.** Heat on the lumen supports the result; heat on borders,
   timestamps or the scope's circular mask invalidates it.

## Expected outcome
Presence-detection on endoscopic images typically lands at 0.95+ AUC and the
backbones will not separate significantly. Frame the paper around the
protocol and the negative comparison, not the headline accuracy.
