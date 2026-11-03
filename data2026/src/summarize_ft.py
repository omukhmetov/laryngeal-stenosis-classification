"""Metrics for the fine-tuned model: frame and session level, 95 % CI by bootstrap over split groups.

Usage: python src/summarize_ft.py --manifest .../manifest.csv --work ./work
"""
import os
import numpy as np, pandas as pd
from sklearn.metrics import confusion_matrix
from common import parse_args, selected_frames
from metrics import metrics, session_level

if __name__ == "__main__":
    a = parse_args(lambda ap: ap.add_argument("--boot", type=int, default=1000))
    img, sel = selected_frames(a.manifest, a.work)
    d = img[sel].reset_index(drop=True)
    y = d.y.values
    P = np.load(os.path.join(a.work, "results", "oof_finetune_effb0.npy"))
    ys, Ps, gs = session_level(P, y, d.session_id.values, d.split_group.values)

    rng, ug, boot = np.random.default_rng(0), np.unique(gs), []
    for _ in range(a.boot):
        idx = np.concatenate([np.where(gs == u)[0] for u in rng.choice(ug, len(ug))])
        try:
            boot.append(metrics(ys[idx], Ps[idx]))
        except ValueError:          # a resample without one of the classes
            pass
    B = pd.DataFrame(boot)
    out = pd.DataFrame({"frame": metrics(y, P), "session": metrics(ys, Ps),
                        "session_ci_low": B.quantile(0.025), "session_ci_high": B.quantile(0.975)}).round(3)
    out.to_csv(os.path.join(a.work, "results", "finetune_effb0_summary.csv"))
    print(out)
    print("session-level confusion (rows = true normal/mild/severe):\n", confusion_matrix(ys, Ps.argmax(1)))
