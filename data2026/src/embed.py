"""Frozen ImageNet features (mean of the frame and its horizontal flip).

Usage: python src/embed.py --manifest .../manifest.csv --work ./work
"""
import os, time
import numpy as np, timm, torch
from PIL import Image
from common import parse_args, image_table, frame_path

RELEASES = "https://github.com/rwightman/pytorch-image-models/releases/download"
MODELS = {  # timm name: (weights file on GitHub releases, mean, std)
    "efficientnet_b0": ("v0.1-weights/efficientnet_b0_ra-3dd342df.pth", (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "resnet50": ("v0.1-rsb-weights/resnet50_a1_0-14fe96d1.pth", (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "vit_base_patch16_224": ("v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth", (0.5,) * 3, (0.5,) * 3),
}


def load_backbone(name, work, num_classes=0, **kw):
    rel, mean, std = MODELS[name]
    wpath = os.path.join(work, "weights", os.path.basename(rel))
    if not os.path.exists(wpath):
        torch.hub.download_url_to_file(f"{RELEASES}/{rel}", wpath)
    m = timm.create_model(name, pretrained=False, num_classes=num_classes, **kw)
    sd = torch.load(wpath, map_location="cpu")
    sd = {k: v for k, v in sd.items() if not k.startswith(("fc.", "classifier.", "head."))}
    m.load_state_dict(sd, strict=False)
    return m, mean, std


if __name__ == "__main__":
    a = parse_args()
    img = image_table(a.manifest)
    for name in MODELS:
        out = os.path.join(a.work, "features", f"emb_{name}.npy")
        if os.path.exists(out):
            continue
        m, mean, std = load_backbone(name, a.work)
        m.eval()
        mean_t, std_t = torch.tensor(mean).view(1, 3, 1, 1), torch.tensor(std).view(1, 3, 1, 1)
        feats, t0 = [], time.time()
        for i in range(0, len(img), 32):
            x = np.stack([np.asarray(Image.open(frame_path(a.work, j)).convert("RGB").resize((224, 224), Image.BICUBIC),
                                     dtype=np.float32) / 255 for j in img.idx[i:i + 32]])
            x = (torch.from_numpy(x).permute(0, 3, 1, 2) - mean_t) / std_t
            with torch.no_grad():
                feats.append(((m(x) + m(torch.flip(x, [3]))) / 2).numpy())
        np.save(out, np.concatenate(feats))
        print(f"{name}: {time.time() - t0:.0f} s", flush=True)
