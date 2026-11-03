"""Full fine-tuning of EfficientNet-B0 (ImageNet init), 5-fold StratifiedGroupKFold on
`split_group`. The number of epochs is fixed in advance (no tuning on the test fold).
Out-of-fold probabilities -> results/oof_finetune_effb0.npy.

Usage: python src/finetune.py --manifest .../manifest.csv --work ./work [--epochs 10] [--device cuda]
"""
import os, time
import numpy as np, torch
import torchvision.transforms as T
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from common import parse_args, selected_frames, frame_path
from embed import load_backbone

if __name__ == "__main__":
    a = parse_args(lambda ap: (ap.add_argument("--epochs", type=int, default=10),
                               ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")))
    torch.manual_seed(0); np.random.seed(0)
    img, sel = selected_frames(a.manifest, a.work)
    d = img[sel].reset_index(drop=True)
    y, groups = d.y.values, d.split_group.values
    ims = [Image.open(frame_path(a.work, i)).convert("RGB") for i in d.idx]
    norm = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    tr_tf = T.Compose([T.RandomResizedCrop(224, scale=(0.6, 1.0)), T.RandomHorizontalFlip(), T.RandomRotation(15),
                       T.ColorJitter(0.2, 0.2, 0.15, 0.03), T.ToTensor(), norm])
    te_tf = T.Compose([T.Resize(224), T.ToTensor(), norm])
    dev, P, t0 = a.device, np.zeros((len(y), 3)), time.time()
    out = os.path.join(a.work, "results", "oof_finetune_effb0.npy")

    for fold, (tr, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(np.zeros(len(y)), y, groups)):
        m, _, _ = load_backbone("efficientnet_b0", a.work, num_classes=3, drop_rate=0.3)
        m = m.to(dev)
        w = torch.tensor(len(tr) / (3 * np.bincount(y[tr], minlength=3)), dtype=torch.float32, device=dev)
        lossf = torch.nn.CrossEntropyLoss(weight=w, label_smoothing=0.1)
        opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-2)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=a.epochs * int(np.ceil(len(tr) / 32)), pct_start=0.15)
        for ep in range(a.epochs):
            m.train(); perm = np.random.permutation(tr); tl = 0.0
            for i in range(0, len(perm), 32):
                b = perm[i:i + 32]
                x = torch.stack([tr_tf(ims[j]) for j in b]).to(dev)
                t = torch.tensor(y[b], device=dev)
                opt.zero_grad(); l = lossf(m(x), t); l.backward(); opt.step(); sch.step()
                tl += l.item() * len(b)
            print(f"fold {fold} epoch {ep} loss {tl / len(tr):.3f}  {time.time() - t0:.0f} s", flush=True)
        m.eval()
        with torch.no_grad():
            for i in range(0, len(te), 64):
                b = te[i:i + 64]
                x = torch.stack([te_tf(ims[j]) for j in b]).to(dev)
                P[b] = ((m(x).softmax(1) + m(torch.flip(x, [3])).softmax(1)) / 2).cpu().numpy()
        np.save(out, P)
    print("saved", out)
