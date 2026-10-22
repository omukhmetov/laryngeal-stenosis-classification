# How to run the healthy / unhealthy image classifier

One file does everything: [`classifier.py`](classifier.py). It trains on the
images you already have, tells you how well it separates healthy from unhealthy,
saves the trained model, and then scores new images on request.

## 1. One-time setup

You already have everything installed on this Mac. If you ever move to a fresh
machine:

```bash
pip install torch torchvision pillow numpy scikit-learn
```

## 2. Where the images live

The script looks for two folders next to it:

```
images/
  Healthy/      any images, any sub-folders
  Unhealthy/    any images, any sub-folders
```

Both `Healthy/Unhealthy` (this folder) and the pre-shrunk `images_small/`
cache are picked up automatically — it uses `images_small/` when present
because training is faster on the smaller files.

Frames are grouped into **imaging sessions** by their file name (a date like
`2025-02-17_...`, or a case number like `001`, `001S`). The same session is
never split between training and testing, so the accuracy you see is honest.

## 3. Train and evaluate

```bash
python3 classifier.py
```

This runs 5-fold cross-validation, prints image-level and session-level AUC,
then retrains on all the data and writes **`model.pt`**. Takes ~15–20 minutes
on this Mac (Apple GPU).

Faster check that everything works (~3 min, weaker numbers):

```bash
python3 classifier.py --quick
```

Useful variants:

```bash
python3 classifier.py --mode finetune   # train the whole network (slower, can overfit here)
python3 classifier.py --crop 0.8        # ignore image borders — see "Is it real?" below
python3 classifier.py --folds 5 --epochs 30 --no-save   # evaluate only, don't save
```

Run `python3 classifier.py --help` for the full list.

## 4. Use the trained model on new images

```bash
python3 classifier.py --predict path/to/new/folder
python3 classifier.py --predict scan1.jpg scan2.jpg
```

It prints `P(unhealthy)` and a healthy/unhealthy call for every image, plus an
averaged call per session when a folder holds several frames from the same one.

## 5. Reading the result — and a warning

- **Session-level AUC** is the meaningful score. Image-level looks higher only
  because many frames from one session are near-duplicates.
- The dataset has **only 7 healthy imaging sessions**. Every number swings a lot
  depending on which of those 7 lands in the test fold — that is why the script
  prints a per-fold list and a `+/- SD`. Read it as "roughly this, give or take".
- **Is it detecting disease, or the camera?** Healthy and unhealthy frames were
  captured on different equipment, so borders, black bars and on-screen text can
  give the class away. Run `python3 classifier.py --crop 0.8`: if the score
  drops a lot, the model was partly reading acquisition artefacts, not the
  larynx. Fixing that needs more healthy sessions recorded on the same
  equipment as the unhealthy ones.

`classifier.py` is standalone. The older `img_benchmark.py` / `img_data.py`
pipeline (multi-backbone comparison, corrected significance tests, Grad-CAM) is
still here and unchanged if you want the deeper analysis.
