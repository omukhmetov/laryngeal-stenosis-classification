import numpy as np, pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, cohen_kappa_score, roc_auc_score


def metrics(yt, P):
    """3-class metrics; P = class probabilities (normal, mild, severe)."""
    yp = P.argmax(1)
    m = yt > 0
    return dict(
        bal_acc=balanced_accuracy_score(yt, yp),
        macro_f1=f1_score(yt, yp, average="macro"),
        qwk=cohen_kappa_score(yt, yp, weights="quadratic"),
        auc_macro=roc_auc_score(yt, P, multi_class="ovr", average="macro"),
        auc_norm_vs_sten=roc_auc_score(yt > 0, P[:, 1:].sum(1)),
        auc_mild_vs_severe=roc_auc_score(yt[m] == 2, P[m, 2] / P[m, 1:].sum(1)),
    )


def session_level(P, y, sessions, groups=None):
    """Average frame probabilities within each session; label = the session's majority class."""
    df = pd.DataFrame(P, columns=[0, 1, 2])
    df["s"], df["y"] = sessions, y
    if groups is not None:
        df["g"] = groups
    g = df.groupby("s")
    Ps = g[[0, 1, 2]].mean().values
    ys = g.y.agg(lambda v: v.mode()[0]).values
    gs = g.g.first().values if groups is not None else None
    return ys, Ps, gs
