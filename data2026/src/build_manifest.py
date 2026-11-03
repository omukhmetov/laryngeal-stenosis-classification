#!/usr/bin/env python3
"""Build the frame-level manifest for the endoscopy image set.

Usage:  python src/build_manifest.py "/path/to/Data 2026"

Expects <root>/Здоров 2025г and <root>/Нездоров 2025г/<1-4> степень.
Never moves or renames images; writes only to <root>/_manifest/
(manifest.csv, manifest.xlsx). The outputs contain patient names and must
stay with the data, never in the repository."""
import os, re, sys, json, hashlib, datetime as dt, collections
import pandas as pd

if len(sys.argv) < 2: sys.exit(__doc__)
ROOT = sys.argv[1]
OUT = os.path.join(ROOT, "_manifest"); os.makedirs(OUT, exist_ok=True)
GAP_MIN = 10                     # разрыв > 10 мин => новый сеанс
CLASS_DIRS = {"Здоров 2025г": ("normal", 0),
              "Нездоров 2025г/1 степень": ("stenosis", 1),
              "Нездоров 2025г/2 степень": ("stenosis", 2),
              "Нездоров 2025г/3 степень": ("stenosis", 3),
              "Нездоров 2025г/4 степень": ("stenosis", 4)}
# Optional fixes for folder names the parser splits wrongly, kept with the data (not in the repo):
# <root>/_manifest/name_overrides.json, e.g. {"<name as in folder>": "<Surname I>"}
_ov = os.path.join(OUT, "name_overrides.json")
NAME_OVERRIDE = json.load(open(_ov, encoding="utf-8")) if os.path.exists(_ov) else {}
OTHER_PATH = ["расщелин", "ангиофибром", "гемангиом", "образовани", "адено", "онко",
              "неврологи", "папиллом", "парез", "сужение носовых", "щит жел"]
UNCERTAIN = ["неясн", "неточно"]

def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def parse_folder(name):
    s = re.sub(r"\s+", " ", re.sub(r"([А-Яа-яЁё])(\d)", r"\1 \2", name)).strip()
    m = re.search(r"\d{2}\.\d{2}\.\d{4}", s)
    date = dt.datetime.strptime(m.group(), "%d.%m.%Y").date() if m else None
    head = s[:m.start()].strip() if m else s
    toks = head.split()
    if not toks or toks[0].lower() == "гортань":
        pname = None
    else:
        pname = toks[0]
        if len(toks) > 1 and re.fullmatch(r"[А-ЯЁ]\.?", toks[1]): pname += " " + toks[1].rstrip(".")
        for k, v in NAME_OVERRIDE.items():
            if head.startswith(k) or pname == k: pname = v
    return pname, date

def parse_ts(fname):
    b = re.sub(r"^Копия\s+", "", fname)
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})_(\d{2})(\d{2})(\d{2})_(IMG|VID)(\d+)", b)
    if m:
        y, mo, d, H, M, S = map(int, m.groups()[:6])
        return dt.datetime(y, mo, d, H, M, S), m.group(7), int(m.group(8)), y == 2025
    m = re.match(r"ENDPIC-(\d+)-(\d{2})(\d{2})(\d{4})-(\d{2})(\d{2})(\d{2})", b)
    if m:
        n, d, mo, y, H, M, S = map(int, m.groups())
        return dt.datetime(y, mo, d, H, M, S), "IMG", n, y == 2025
    return None, None, None, False

def is_copy_name(f): return bool(re.search(r"(^Копия|копия|\(\d\))", f))

# ---------- 1. сбор файлов ----------
rows = []
for cdir, (cls, grade) in CLASS_DIRS.items():
    base = os.path.join(ROOT, cdir)
    for r, ds, fs in os.walk(base):
        ds[:] = [d for d in ds if not d.startswith(".")]
        for f in fs:
            ext = f.lower().rsplit(".", 1)[-1]
            if ext not in ("jpg", "jpeg", "mp4"): continue
            full = os.path.join(r, f); rel = os.path.relpath(full, ROOT)
            sub = os.path.relpath(r, base)
            folder = None if sub == "." else sub.split(os.sep)[0]
            ts, kind, num, reliable = parse_ts(f)
            rows.append(dict(file_path=rel, file_name=f, media="video" if ext == "mp4" else "image",
                             class_dir=cdir, label=cls, grade=grade, patient_folder=folder,
                             timestamp=ts, frame_no=num, clock_reliable=reliable,
                             size_kb=round(os.path.getsize(full) / 1024), md5=md5(full)))
df = pd.DataFrame(rows)
print("файлов:", len(df))

# ---------- 2. пациенты из названий папок ----------
fold = df.dropna(subset=["patient_folder"])[["class_dir", "patient_folder"]].drop_duplicates()
pmap, fdate, unk = {}, {}, 0
for _, r in fold.iterrows():
    pname, d = parse_folder(r.patient_folder)
    if pname is None:
        unk += 1; pname = f"(без имени #{unk})"
    pmap[(r.class_dir, r.patient_folder)] = pname; fdate[(r.class_dir, r.patient_folder)] = d
names = sorted(set(pmap.values()))
pid = {n: f"P{i+1:03d}" for i, n in enumerate(names)}
df["patient_name"] = [pmap.get((c, f)) if isinstance(f, str) else None for c, f in zip(df.class_dir, df.patient_folder)]
df["patient_id"] = df.patient_name.map(pid)
df["patient_source"] = df.patient_id.notna().map({True: "папка", False: None})

# ---------- 3. сеансы ----------
df["session_id"] = None; df["session_date"] = None; df["session_confidence"] = None
# 3a. папки пациентов: сеанс = пациент + дата из названия папки
for (c, f), idx in df[df.patient_folder.notna()].groupby(["class_dir", "patient_folder"]).groups.items():
    d = fdate[(c, f)]
    df.loc[idx, "session_date"] = d.isoformat() if d else None
    df.loc[idx, "session_key"] = f"{pmap[(c, f)]}|{d}"
    df.loc[idx, "session_confidence"] = "высокая (папка)"
# 3b. файлы без папки: кластеризация по времени
loose = df[df.patient_folder.isna()].copy()
loose["clock_block"] = [f"{c}|{t.date() if t is not None else 'нет даты'}" + ("" if ok else "|сбитые часы")
                        for c, t, ok in zip(loose.class_dir, loose.timestamp, loose.clock_reliable)]
for blk, g in loose.sort_values("timestamp").groupby("clock_block"):
    k, prev_t, prev_n = 0, None, None
    for i, r in g.iterrows():
        t, n = r.timestamp, r.frame_no
        if prev_t is not None and t is not None:
            gap = (t - prev_t).total_seconds()
            if gap > GAP_MIN * 60 or (r.media == "image" and prev_n is not None and n is not None and n < prev_n and gap > 60):
                k += 1
        df.loc[i, "session_key"] = f"{blk}|s{k}"
        df.loc[i, "session_date"] = t.date().isoformat() if (t is not None and r.clock_reliable) else None
        df.loc[i, "session_confidence"] = "средняя (по времени)" if r.clock_reliable else "низкая (сбитые часы камеры)"
        if r.media == "image": prev_n = n
        prev_t = t
    df.loc[g.index, "clock_block"] = blk
keys = list(dict.fromkeys(df.sort_values(["class_dir", "session_key"]).session_key))
df["session_id"] = df.session_key.map({k: f"S{i+1:03d}" for i, k in enumerate(keys)})

# ---------- 4. дубли ----------
dupe_groups = {h: f"D{i+1:03d}" for i, h in enumerate(sorted(h for h, n in df.md5.value_counts().items() if n > 1))}
df["dup_group"] = df.md5.map(dupe_groups)
df["is_primary"] = True
for h, g in df[df.dup_group.notna()].groupby("md5"):
    order = sorted(g.index, key=lambda i: (not isinstance(df.at[i, "patient_folder"], str), is_copy_name(df.at[i, "file_name"]), len(df.at[i, "file_path"])))
    df.loc[order[1:], "is_primary"] = False
conflict_md5 = {h for h, g in df[df.dup_group.notna()].groupby("md5") if g.grade.nunique() > 1}
df["label_conflict"] = df.md5.isin(conflict_md5)

# ---------- 5. привязка файлов без папки к пациентам ----------
folder_by_md5 = df[df.patient_id.notna()].groupby("md5").patient_id.first()
for sid, g in df[df.patient_id.isna()].groupby("session_id"):
    hit = g.md5.map(folder_by_md5).dropna()
    if len(hit):
        df.loc[g.index, "patient_id"] = hit.iloc[0]; df.loc[g.index, "patient_source"] = "совпадение файла (точно)"
        continue
    tt = g.timestamp.dropna()
    if not g.clock_reliable.all() or tt.empty: continue
    lo, hi = tt.min() - dt.timedelta(minutes=GAP_MIN), tt.max() + dt.timedelta(minutes=GAP_MIN)
    cand = df[df.patient_id.notna() & df.clock_reliable & df.timestamp.between(lo, hi) & (df.patient_source == "папка")]
    if cand.patient_id.nunique() == 1:
        df.loc[g.index, "patient_id"] = cand.patient_id.iloc[0]; df.loc[g.index, "patient_source"] = "по времени съёмки (вероятно)"
inv = {v: k for k, v in pid.items()}
df["patient_name"] = df.patient_id.map(inv)

# ---------- 6. группа для разбиения (union-find) ----------
par = {}
def find(x):
    par.setdefault(x, x)
    while par[x] != x: par[x] = par[par[x]]; x = par[x]
    return x
def union(a, b): par[find(a)] = find(b)
for _, r in df.iterrows():
    node = f"sess:{r.session_id}"; find(node)
    if isinstance(r.patient_id, str): union(node, f"pat:{r.patient_id}")
    if isinstance(r.dup_group, str): union(node, f"dup:{r.dup_group}")
    if not r.clock_reliable and not isinstance(r.patient_folder, str): union(node, f"blk:{r.clock_block}")
roots = {}
df["split_group"] = [roots.setdefault(find(f"sess:{s}"), f"G{len(roots)+1:03d}") for s in df.session_id]

# ---------- 7. флаги ----------
pat_grades = df[df.patient_id.notna()].groupby("patient_id").grade.unique()
folder_grades = df[df.patient_source == "папка"].groupby("patient_id").grade.unique()
S = lambda x: x if isinstance(x, str) else ""
def flags(r):
    fl, pf = [], S(r.patient_folder).lower()
    if r.label_conflict: fl.append("конфликт меток: тот же файл в другом классе")
    if isinstance(r.dup_group, str) and not r.is_primary: fl.append("дубль (не основная копия)")
    if not isinstance(r.patient_id, str): fl.append("пациент неизвестен")
    if not r.clock_reliable: fl.append("сбитые часы камеры")
    if any(w in pf for w in UNCERTAIN): fl.append("сомнительный диагноз в названии папки")
    if r.grade == 0:
        o = [w for w in OTHER_PATH if w in pf]
        if o: fl.append("другая патология: " + ", ".join(o))
        if "стеноз" in pf and "норма" not in pf:
            fl.append("в «Здоров», но в названии «стеноз»" + (" (по ИБ — 1 степень)" if "из иб" in pf else ""))
    if isinstance(r.patient_id, str) and len(pat_grades.get(r.patient_id, [])) > 1:
        fl.append("у пациента файлы в разных классах: " + "/".join(map(str, sorted(pat_grades[r.patient_id]))))
        if r.patient_source != "папка" and r.grade not in folder_grades.get(r.patient_id, []):
            fl.append("класс не совпадает с папкой пациента")
    if r.grade > 0 and S(r.patient_folder) and "стеноз" not in pf and \
       not any("стеноз" in S(x).lower() for x in df[df.patient_id == r.patient_id].patient_folder.unique()):
        fl.append("в названии папки нет «стеноз»")
    if S(r.patient_name).startswith("(без имени") and S(r.session_date):
        same = df[(df.session_date == r.session_date) & df.patient_name.fillna("").str.startswith("(без имени") & (df.patient_id != r.patient_id)]
        if len(same): fl.append("возможно тот же пациент, что и другая папка без имени за ту же дату")
    return "; ".join(fl)
df["flags"] = df.apply(flags, axis=1)
bad = lambda s: any(k in s for k in ("конфликт меток", "дубль", "сомнительный", "по ИБ — 1 степень", "класс не совпадает"))
df["include"] = (df.media == "image") & ~df["flags"].map(bad)

# ---------- 8. вывод ----------
cols = ["file_path", "media", "label", "grade", "patient_id", "patient_source", "session_id", "session_date",
        "session_confidence", "split_group", "timestamp", "clock_reliable", "frame_no", "dup_group", "is_primary",
        "label_conflict", "include", "flags", "size_kb", "md5"]
man = df.sort_values(["grade", "patient_id", "session_id", "timestamp"], na_position="last")[cols]
man.to_csv(os.path.join(OUT, "manifest.csv"), index=False, encoding="utf-8-sig")

img = df[df.media == "image"]
summ = img.groupby("grade").agg(кадров=("file_path", "size"), к_использованию=("include", "sum"),
        пациентов_известно=("patient_id", "nunique"), сеансов=("session_id", "nunique"),
        групп_для_разбиения=("split_group", "nunique"), без_пациента=("patient_id", lambda s: s.isna().sum()),
        конфликт_меток=("label_conflict", "sum"), дублей=("is_primary", lambda s: (~s).sum()))
summ.loc["всего"] = [len(img), img.include.sum(), img.patient_id.nunique(), img.session_id.nunique(),
                     img.split_group.nunique(), img.patient_id.isna().sum(), img.label_conflict.sum(), (~img.is_primary).sum()]
summ = summ.reset_index().rename(columns={"grade": "степень (0 = здоров)"})
summ.insert(1, "видео", [int((df[(df.media == "video")].grade == g).sum()) if g != "всего" else int((df.media == "video").sum())
                         for g in summ["степень (0 = здоров)"]])

pats = df[df.patient_id.notna()].groupby("patient_id").agg(
    имя=("patient_name", "first"), степени=("grade", lambda s: "/".join(map(str, sorted(s.unique())))),
    сеансов=("session_id", "nunique"), даты=("session_date", lambda s: ", ".join(sorted({x for x in s if x}))),
    кадров=("media", lambda s: (s == "image").sum()), видео=("media", lambda s: (s == "video").sum()),
    папки=("patient_folder", lambda s: " | ".join(sorted({x for x in s if x})))).reset_index()

sess = df.groupby("session_id").agg(степень=("grade", "first"), пациент=("patient_id", "first"),
    дата=("session_date", "first"), начало=("timestamp", "min"), конец=("timestamp", "max"),
    кадров=("media", lambda s: (s == "image").sum()), видео=("media", lambda s: (s == "video").sum()),
    надёжность=("session_confidence", "first"), группа=("split_group", "first"),
    папка=("patient_folder", "first"), блок=("clock_block", "first")).reset_index()

rev = df[df["flags"].str.contains("конфликт|сомнительный|в «Здоров», но|другая патология|разных классах|нет «стеноз»|возможно тот же")]
rev = rev.groupby(["class_dir", "patient_folder", "patient_id"], dropna=False).agg(
    файлов=("file_path", "size"),
    вопросы=("flags", lambda s: "; ".join(sorted({x for f in s for x in f.split("; ") if x and x not in ("пациент неизвестен", "сбитые часы камеры", "дубль (не основная копия)")}))),
    примеры=("file_path", lambda s: " | ".join(list(s)[:3]))).reset_index()
rev = rev.rename(columns={"patient_id": "пациент"}); rev.insert(0, "решение врача", "")

readme = pd.DataFrame({"поле": [
 "file_path", "label / grade", "patient_id", "patient_source", "session_id", "session_confidence", "split_group",
 "clock_reliable", "dup_group / is_primary", "label_conflict", "include", "flags"], "что значит": [
 "путь относительно папки Data 2026",
 "класс из папки: normal/stenosis; степень Cotton-Myer (0 = здоров)",
 "обезличенный ID пациента (имена — только на листе «Пациенты»)",
 "откуда известен пациент: папка / совпадение файла (точно) / по времени съёмки (вероятно)",
 f"осмотр: папка пациента = один сеанс; файлы без папки разбиты по разрыву > {GAP_MIN} мин или сбросу номера кадра",
 "насколько надёжно определён сеанс",
 "ГЛАВНОЕ ПОЛЕ для train/test: объединяет кадры одного пациента, дубли и блоки со сбитыми часами. Делить только по нему (GroupKFold / StratifiedGroupKFold)",
 "False — дата камеры сбита (1970, 2014, 1998, 2000, 2017, 2023 и т. п.)",
 "одинаковые по содержимому файлы; основная копия — из папки пациента и без «Копия/(1)»",
 "тот же файл лежит в классах с разной степенью",
 "предлагаемый отбор: только изображения; исключены дубли, конфликты меток, «неясный/неточно», «стеноз 1 из ИБ» в группе «Здоров» и кадры, чей класс расходится с папкой того же пациента (до решения врача)",
 "все замечания по строке"]})

with pd.ExcelWriter(os.path.join(OUT, "manifest.xlsx"), engine="openpyxl") as w:
    readme.to_excel(w, sheet_name="Пояснения", index=False)
    summ.to_excel(w, sheet_name="Сводка", index=False)
    rev.to_excel(w, sheet_name="На проверку врачу", index=False)
    pats.to_excel(w, sheet_name="Пациенты", index=False)
    sess.to_excel(w, sheet_name="Сеансы", index=False)
    man.to_excel(w, sheet_name="Манифест", index=False)
    for ws in w.book.worksheets:
        ws.freeze_panes = "A2"; ws.auto_filter.ref = ws.dimensions
        for col in ws.columns:
            width = min(60, max(len(str(c.value or "")) for c in col[:200]) + 2)
            ws.column_dimensions[col[0].column_letter].width = width
print(summ.to_string(index=False)); print(); print(pats[["patient_id", "имя", "степени", "сеансов", "кадров"]].to_string(index=False))
print("\nна проверку:", len(rev))
