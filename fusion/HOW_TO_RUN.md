# Image + clinical fusion: stenosis severity (grade 3–4 vs 1–2)

Question: do endoscopic images tell us anything about severity that the clinical
variables in DATA.xlsx don't already tell us?

## 1. Link images to patients (manual, one-time)

`link_template.xlsx` is already generated. Its **exams** sheet has one row per
candidate exam: frames from the same day with no gap longer than 20 minutes.
Hints per row are the date, time range, folder name and frame count.

1. Fill `patient_id` using the DATA.xlsx format (e.g. `2024-2`). Leave it blank if unknown.
2. Several exams can share one patient. If one row actually covers two patients,
   split it by adding a row and editing the **images** sheet so each frame points to the right `exam_id`.
3. `clock_reset = TRUE` means the camera date was 1970. Ignore the date and use the folder name or clinic log.
4. Save as `fusion/link.xlsx`.

To regenerate the template (e.g. with a different gap):
`python3 fusion/make_link_template.py --gap 30`

## 2. Run (from the Stenosis folder)

```bash
python3 fusion/fusion_stenosis.py                # main result
python3 fusion/fusion_stenosis.py --crop 0.8 --out fusion/results_crop   # border-crop sensitivity
python3 fusion/fusion_stenosis.py --demo         # pipeline check with fake links
```

It uses the same ImageNet ResNet-18 as `images/classifier.py` (frozen, used only as a
feature extractor). Embeddings are cached, so reruns take about a minute.
`--weights images/model.pt` uses your healthy/unhealthy backbone instead.

## 3. Models (same patients, same folds, 5-fold × 10 repeats)

| | |
|---|---|
| clinical | CatBoost on DATA.xlsx variables |
| image | logistic regression on 8 PCs of the per-patient mean embedding |
| late | average of clinical and image probabilities |
| early | CatBoost on clinical variables + 8 image PCs |

## 4. Outputs (`fusion/results/`)

- `cohort.txt`: how many patients were linked, graded and in each class. Report this first.
- `summary.csv`: AUC / balanced accuracy / Brier ± SD. `delta_auc_vs_clinical` and `p_nb` come from the
  Nadeau–Bengio corrected t-test. `p_perm` is the label-permutation test for image-only.
- `oof_predictions.csv`, `roc.png`, `fused_feature_importance.csv`

## Reading it

- Clinical-only is weak on this outcome: about AUC 0.56 on all 172 graded patients (checked with `--demo --demo-n 500`).
  So there is room for images to help, and a null result is also informative.
- The claim to make is "fusion vs clinical", tested with `p_nb`, not the absolute AUC.
- Fewer than about 60 linked patients makes this a feasibility study. Say so.
- If `--crop 0.8` drops image-only AUC a lot, the model is reading equipment or era
  (e.g. older exams = different camera), not anatomy. Year is not a model input, but it can leak through the image.
