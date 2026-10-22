"""
PICLS severity score: a bedside points score predicting severe (Cotton-Myer 3-4)
vs mild (1-2) post-intubation laryngeal stenosis in children.

Pipeline
 1. Data preparation (only predictors known BEFORE the diagnostic endoscopy)
 2. Univariable and full multivariable logistic regression
 3. Backward selection by AIC (grouped variables) -> parsimonious model
 4. Points by the Sullivan (Framingham) method: beta / smallest beta, rounded
 5. Apparent performance: AUC, calibration (intercept, slope), Brier
 6. Internal validation: 1000 bootstrap replicates repeating the WHOLE procedure
    (selection + points + recalibration) -> optimism-corrected AUC and slope
 7. Temporal validation: develop on 2017-2022, test on 2023-2024
 8. Decision curve analysis, risk-group table, figures

Usage:  python picls_score.py DATA.xlsx out_dir
"""
import sys, os, json, warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(2026)
N_BOOT = 1000

# ---------------------------------------------------------------- 1. data
GROUPS = {  # variable group -> dummy columns (reference category in comment)
    "age_gt12":   ["age_gt12"],                   # ref: intubated at <=12 months
    "reintub":    ["reintub"],                    # ref: single intubation
    "reason":     ["reason_inf_neo", "reason_cns_obs"],  # ref: elective surgery
    "tertiary":   ["tertiary"],                   # ref: primary/secondary clinic
    "premature":  ["premature"],                  # ref: term birth
    "comorb":     ["comorb"],                     # ref: no comorbidity
    "rural":      ["rural"],                      # ref: urban
    "male":       ["male"],                       # ref: female
}
LABELS_RU = {
    "age_gt12": "Возраст первичной интубации > 12 мес",
    "reintub": "Повторная интубация",
    "reason_inf_neo": "Причина интубации: инфекция / неонатальная",
    "reason_cns_obs": "Причина интубации: патология ЦНС / обструкция",
    "tertiary": "Интубация в клинике 3-го уровня",
    "premature": "Недоношенность",
    "comorb": "Наличие сопутствующего синдрома",
    "rural": "Сельское проживание",
    "male": "Мужской пол",
}


def load(path):
    df = pd.read_excel(path, header=1)
    df = df.replace({"no data": np.nan, "not reported": np.nan})
    df["Delivery category"] = df["Delivery category"].astype(str).str.strip().replace({"nan": np.nan})
    return df


def prepare(df):
    g = pd.to_numeric(df["Grade of stenosis"], errors="coerce")
    d = df.loc[g.notna()].copy()
    d["grade"] = g[g.notna()].astype(int)
    d["severe"] = (d["grade"] >= 3).astype(int)
    age = pd.to_numeric(d["Age at primary intubation (months)"].replace({"up to 1": 0.5}), errors="coerce")
    d["age_months"] = age
    reason = d["Reason of intubation"]
    X = pd.DataFrame(index=d.index)
    X["age_gt12"] = (age > 12).astype(float).where(age.notna())
    X["reintub"] = pd.to_numeric(d["re-intubation"], errors="coerce")
    X["reason_inf_neo"] = reason.isin(["infection", "neonatal causes"]).astype(float).where(reason.notna())
    X["reason_cns_obs"] = reason.isin(["CNS pathology", "obstruction"]).astype(float).where(reason.notna())
    lvl = d["intubation clinic's level"]
    X["tertiary"] = (lvl == "tertiary").astype(float).where(lvl.notna())
    dc = d["Delivery category"]
    X["premature"] = (dc == "premature").astype(float).where(dc.notna())
    X["comorb"] = (d["Leading comorbidity syndrome"] != "none").astype(float)
    res = d["patient's residence"]
    X["rural"] = (res == "rural").astype(float).where(res.notna())
    X["male"] = (d["Gender"] == "m").astype(float)
    n_missing = int(X.isna().any(axis=1).sum())
    # <2% missing per variable -> mode imputation (documented in report)
    X = X.fillna(X.mode().iloc[0])
    return d, X, d["severe"].values, n_missing


# ---------------------------------------------------------------- 2-4. modelling
def fit_logit(X, y, cols):
    Xm = sm.add_constant(X[cols].values, has_constant="add")
    return sm.Logit(y, Xm).fit(disp=0, maxiter=200)


class Firth:
    """Firth-penalised logistic regression (bias reduction; handles separation).
    Wald CIs from the penalised information matrix."""
    def __init__(self, X, y, cols, iters=100, tol=1e-8):
        A = np.column_stack([np.ones(len(y)), X[cols].values.astype(float)])
        b = np.zeros(A.shape[1])
        for _ in range(iters):
            p = 1 / (1 + np.exp(-A @ b)); W = p * (1 - p)
            I = A.T @ (A * W[:, None]); Iinv = np.linalg.pinv(I)
            h = np.einsum("ij,jk,ik->i", A * np.sqrt(W)[:, None], Iinv, A * np.sqrt(W)[:, None])
            step = Iinv @ (A.T @ (y - p + h * (0.5 - p)))
            mx = np.abs(step).max()
            if mx > 5: step *= 5 / mx
            b = b + step
            if np.abs(step).max() < tol: break
        p = 1 / (1 + np.exp(-A @ b)); W = p * (1 - p)
        self.cov = np.linalg.pinv(A.T @ (A * W[:, None]))
        self.params, self.se = b, np.sqrt(np.diag(self.cov))
        from scipy.stats import norm
        self.pvalues = 2 * norm.sf(np.abs(b / self.se))
        self.A = A
    def conf_int(self):
        return np.column_stack([self.params - 1.96 * self.se, self.params + 1.96 * self.se])
    def predict(self):
        return 1 / (1 + np.exp(-self.A @ self.params))


def backward_aic(X, y, groups=GROUPS):
    keep = list(groups)
    cols = lambda ks: [c for k in ks for c in groups[k]]
    cur = fit_logit(X, y, cols(keep)).aic
    while len(keep) > 1:
        trials = []
        for k in keep:
            ks = [j for j in keep if j != k]
            try:
                trials.append((fit_logit(X, y, cols(ks)).aic, k))
            except Exception:
                continue
        best_aic, drop = min(trials)
        if best_aic < cur:
            keep.remove(drop); cur = best_aic
        else:
            break
    return cols(keep)


UNIT = np.log(1.5)  # 1 point = log-odds of OR 1.5 (fixed a priori)


def sullivan_points(params, cols):
    """Sullivan/Framingham points = round(beta / UNIT).
    A factor with negative beta is re-expressed as its complement
    (e.g. 'tertiary clinic' -> 'NOT tertiary clinic') so that all points are >= 0."""
    b = pd.Series(params[1:], index=cols)
    return b, UNIT, (b / UNIT).round().astype(int)


def score_of(X, pts):
    Z = X[pts.index].values.astype(float)
    Z = np.where(pts.values < 0, 1 - Z, Z)
    return Z @ np.abs(pts.values)


def recalibrate(score, y):
    """logistic model  logit(p) = a + c * score  -> maps points to risk"""
    m = Firth(pd.DataFrame({"s": score.astype(float)}), y, ["s"])
    return m.params


def predict(score, ab):
    return 1 / (1 + np.exp(-(ab[0] + ab[1] * score)))


def cal_stats(p, y):
    lp = np.log(p / (1 - p))
    slope = sm.Logit(y, sm.add_constant(lp)).fit(disp=0).params[1]
    citl = sm.GLM(y, np.ones_like(lp), family=sm.families.Binomial(), offset=lp).fit().params[0]
    return float(citl), float(slope)


def develop(X, y, cols=None):
    cols = backward_aic(X, y) if cols is None else cols
    m = Firth(X, y, cols)
    b, unit, pts = sullivan_points(m.params, cols)
    s = score_of(X, pts)
    ab = recalibrate(s, y)
    return dict(cols=cols, model=m, beta=b, unit=unit, pts=pts, ab=ab)


def evaluate(dev, X, y):
    s = score_of(X, dev["pts"])
    p = np.clip(predict(s, dev["ab"]), 1e-6, 1 - 1e-6)
    citl, slope = cal_stats(p, y)
    return dict(auc=roc_auc_score(y, s), citl=citl, slope=slope, brier=brier_score_loss(y, p))


def auc_ci(y, s, n=2000):
    a = []
    for _ in range(n):
        i = RNG.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        a.append(roc_auc_score(y[i], s[i]))
    return np.percentile(a, [2.5, 97.5])


def net_benefit(y, p, thr):
    n = len(y); out = []
    for t in thr:
        tp = np.sum((p >= t) & (y == 1)); fp = np.sum((p >= t) & (y == 0))
        out.append(tp / n - fp / n * t / (1 - t))
    return np.array(out)


# ---------------------------------------------------------------- main
def main(path, out):
    os.makedirs(out, exist_ok=True)
    raw = load(path)
    d, X, y, n_missing = prepare(raw)
    R = {"n_total": len(raw), "n_graded": len(d), "n_severe": int(y.sum()), "n_mild": int((1 - y).sum()),
         "n_rows_imputed": n_missing,
         "grade_counts": d["grade"].value_counts().sort_index().to_dict()}

    # graded vs ungraded comparison (selection bias check)
    raw_g = pd.to_numeric(raw["Grade of stenosis"], errors="coerce").notna()
    comp = pd.DataFrame({
        "Годы": [f"{raw.Year[raw_g].min()}–{raw.Year[raw_g].max()}", f"{raw.Year[~raw_g].min()}–{raw.Year[~raw_g].max()}"],
        "n": [int(raw_g.sum()), int((~raw_g).sum())],
        "Трахеостомия, %": [round(100 * raw.loc[raw_g, "history of tracheostomy"].astype(float).mean(), 1),
                            round(100 * raw.loc[~raw_g, "history of tracheostomy"].astype(float).mean(), 1)],
        "Экстренная интубация, %": [round(100 * (raw.loc[raw_g, "Emergency/planned intubation"] == "emergency").mean(), 1),
                                    round(100 * (raw.loc[~raw_g, "Emergency/planned intubation"] == "emergency").mean(), 1)],
    }, index=["Со степенью", "Без степени"])
    comp.to_csv(f"{out}/graded_vs_ungraded.csv")

    # univariable + full multivariable
    all_cols = [c for k in GROUPS for c in GROUPS[k]]
    rows = []
    full = fit_logit(X, y, all_cols)
    ci_full = np.exp(full.conf_int()[1:])
    for i, c in enumerate(all_cols):
        n1 = int(X[c].sum()); sev1 = int(y[X[c] == 1].sum())
        u = fit_logit(X, y, [c]); ci = np.exp(u.conf_int()[1])
        rows.append(dict(var=c, label=LABELS_RU[c], n=n1, severe_pct=round(100 * sev1 / max(n1, 1), 1),
                         uni_or=np.exp(u.params[1]), uni_lo=ci[0], uni_hi=ci[1], uni_p=u.pvalues[1],
                         mv_or=np.exp(full.params[i + 1]), mv_lo=ci_full[i][0], mv_hi=ci_full[i][1], mv_p=full.pvalues[i + 1]))
    tab_or = pd.DataFrame(rows); tab_or.to_csv(f"{out}/table_or.csv", index=False)
    R["full_auc"] = roc_auc_score(y, full.predict())

    # final score on all data
    dev = develop(X, y)
    m = dev["model"]; ci = np.exp(m.conf_int()[1:])
    score_tab = pd.DataFrame({"var": dev["cols"], "label": [LABELS_RU[c] for c in dev["cols"]],
                              "beta": dev["beta"].values, "or": np.exp(dev["beta"].values),
                              "lo": ci[:, 0], "hi": ci[:, 1], "p": m.pvalues[1:], "points": dev["pts"].values})
    score_tab.to_csv(f"{out}/table_score.csv", index=False)
    s = score_of(X, dev["pts"]); p = predict(s, dev["ab"])
    app = evaluate(dev, X, y); app["auc_ci"] = auc_ci(y, s).tolist()
    app["model_auc"] = roc_auc_score(y, m.predict())
    R.update(apparent=app, unit=float(dev["unit"]), recal=list(map(float, dev["ab"])),
             selected=dev["cols"], intercept=float(m.params[0]))

    # risk table by total points
    rt = pd.DataFrame({"score": s, "y": y, "p": p}).groupby("score").agg(n=("y", "size"), severe=("y", "sum"), pred=("p", "mean"))
    rt["obs_pct"] = 100 * rt.severe / rt.n; rt["pred_pct"] = 100 * rt.pred
    rt.to_csv(f"{out}/table_risk_by_points.csv")
    grp = pd.cut(s, [-1, 2, 5, 100], labels=["Низкий (0–1)", "Умеренный (3–4)", "Высокий (6–7)"])
    rg = pd.DataFrame({"g": grp, "y": y, "p": p}).groupby("g", observed=True).agg(n=("y", "size"), severe=("y", "sum"), pred=("p", "mean"))
    rg["obs_pct"] = 100 * rg.severe / rg.n; rg["pred_pct"] = 100 * rg.pred
    ci_g = []
    for _, r in rg.iterrows():
        from statsmodels.stats.proportion import proportion_confint
        lo, hi = proportion_confint(r.severe, r.n, method="wilson"); ci_g.append((100 * lo, 100 * hi))
    rg["lo"], rg["hi"] = [c[0] for c in ci_g], [c[1] for c in ci_g]
    rg.to_csv(f"{out}/table_risk_groups.csv")
    # sensitivity/specificity at cut-offs
    cut = []
    for c in sorted(set(s))[1:]:
        pos = s >= c
        cut.append(dict(cut=c, sens=100 * np.mean(pos[y == 1]), spec=100 * np.mean(~pos[y == 0]),
                        ppv=100 * np.mean(y[pos]), npv=100 * np.mean(1 - y[~pos])))
    pd.DataFrame(cut).to_csv(f"{out}/table_cutoffs.csv", index=False)

    # bootstrap optimism (whole procedure)
    sel_freq = pd.Series(0, index=list(GROUPS)); opt = {"auc": [], "slope": [], "citl": []}; boot_pts_n = []
    for b in range(N_BOOT):
        i = RNG.integers(0, len(y), len(y))
        Xb, yb = X.iloc[i].reset_index(drop=True), y[i]
        try:
            db = develop(Xb, yb)
            eb, eo = evaluate(db, Xb, yb), evaluate(db, X, y)
        except Exception:
            continue
        for k, cs in GROUPS.items():
            if any(c in db["cols"] for c in cs): sel_freq[k] += 1
        opt["auc"].append(eb["auc"] - eo["auc"]); opt["slope"].append(eo["slope"])
        opt["citl"].append(eo["citl"]); boot_pts_n.append(len(db["cols"]))
    nb = len(opt["auc"])
    R["bootstrap"] = dict(n_ok=nb, optimism_auc=float(np.mean(opt["auc"])),
                          corrected_auc=float(app["auc"] - np.mean(opt["auc"])),
                          corrected_slope=float(np.mean(opt["slope"])),
                          corrected_citl=float(np.mean(opt["citl"])),
                          selection_freq={k: round(100 * v / nb, 1) for k, v in sel_freq.items()})

    # temporal validation
    early = (d["Year"].astype(int) <= 2022).values
    dv = develop(X[early].reset_index(drop=True), y[early])
    ev = evaluate(dv, X[~early].reset_index(drop=True), y[~early])
    s_val = score_of(X[~early], dv["pts"])
    ev["auc_ci"] = auc_ci(y[~early], s_val).tolist()
    ev_same = evaluate(dev, X[~early].reset_index(drop=True), y[~early])  # final score applied to late period
    dfix = develop(X[early].reset_index(drop=True), y[early], cols=dev["cols"])  # fixed predictor set
    evf = evaluate(dfix, X[~early].reset_index(drop=True), y[~early])
    s_fix = score_of(X[~early], dfix["pts"]); evf["auc_ci"] = auc_ci(y[~early], s_fix).tolist()
    R["temporal"] = dict(n_dev=int(early.sum()), ev_dev=int(y[early].sum()), n_val=int((~early).sum()),
                         ev_val=int(y[~early].sum()), dev_selected=dv["cols"],
                         dev_points={k: int(v) for k, v in dv["pts"].items()}, val=ev,
                         fixed_points={k: int(v) for k, v in dfix["pts"].items()},
                         fixed_beta={k: float(v) for k, v in dfix["beta"].items()}, fixed_val=evf,
                         final_score_on_late_auc=ev_same["auc"])

    # decision curve
    thr = np.arange(0.10, 0.71, 0.01)
    nb_score = net_benefit(y, p, thr); nb_all = net_benefit(y, np.ones_like(p), thr)
    pd.DataFrame({"threshold": thr, "score": nb_score, "treat_all": nb_all}).to_csv(f"{out}/dca.csv", index=False)

    # ---- figures
    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    f, t, _ = roc_curve(y, s); ax.plot(f, t, color="#1f5fa8", lw=2, label=f"Вся выборка (кажущаяся), AUC = {app['auc']:.2f}")
    f, t, _ = roc_curve(y[~early], s_fix); ax.plot(f, t, color="#d9822b", lw=2, ls="--", label=f"Временная валидация 2023–24, AUC = {evf['auc']:.2f}")
    ax.plot([0, 1], [0, 1], color="#999", lw=1)
    ax.set_xlabel("1 − специфичность"); ax.set_ylabel("Чувствительность"); ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False); fig.tight_layout(); fig.savefig(f"{out}/fig_roc.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.bar(rt.index.astype(int).astype(str), rt.obs_pct, color="#1f5fa8", label="Наблюдаемая доля")
    ax.plot(rt.index.astype(int).astype(str), rt.pred_pct, "o-", color="#d9822b", label="Предсказанный риск")
    for x_, (n_, o_) in enumerate(zip(rt.n, rt.obs_pct)): ax.text(x_, 3, f"n={n_}", color="white" if o_ > 8 else "#333", ha="center", fontsize=8)
    ax.set_xlabel("Сумма баллов"); ax.set_ylabel("Тяжёлый стеноз (3–4), %"); ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=8); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{out}/fig_risk_by_points.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(thr, nb_score, color="#1f5fa8", lw=2, label="Шкала")
    ax.plot(thr, nb_all, color="#999", lw=1.5, label="Всем эндоскопию срочно")
    ax.axhline(0, color="#333", lw=1, label="Никому")
    ax.set_ylim(-0.05, max(nb_score.max(), nb_all.max()) + 0.05)
    ax.set_xlabel("Пороговая вероятность"); ax.set_ylabel("Чистая польза"); ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False); fig.tight_layout(); fig.savefig(f"{out}/fig_dca.png", dpi=200); plt.close(fig)

    # calibration plot: groups by score (apparent) + bootstrap-corrected line
    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    ax.scatter(rt.pred_pct / 100, rt.obs_pct / 100, s=rt.n * 3, color="#1f5fa8", alpha=.8, label="Группы по сумме баллов")
    ax.plot([0, 1], [0, 1], color="#999", lw=1, label="Идеальная калибровка")
    xx = np.linspace(0.05, 0.95, 50); lp = np.log(xx / (1 - xx))
    bc = R["bootstrap"]; yy = 1 / (1 + np.exp(-(bc["corrected_citl"] + bc["corrected_slope"] * lp)))
    ax.plot(xx, yy, color="#d9822b", ls="--", lw=1.5, label=f"С поправкой на оптимизм (наклон {bc['corrected_slope']:.2f})")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xlabel("Предсказанный риск"); ax.set_ylabel("Наблюдаемая доля")
    ax.legend(frameon=False, fontsize=7, loc="upper left"); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{out}/fig_calibration.png", dpi=200); plt.close(fig)

    with open(f"{out}/results.json", "w") as fh:
        json.dump(R, fh, ensure_ascii=False, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(json.dumps(R, ensure_ascii=False, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    print(tab_or.round(3).to_string()); print(score_tab.round(3).to_string()); print(rt.round(1).to_string()); print(rg.round(1).to_string()); print(pd.DataFrame(cut).round(1).to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "DATA.xlsx", sys.argv[2] if len(sys.argv) > 2 else "out")
