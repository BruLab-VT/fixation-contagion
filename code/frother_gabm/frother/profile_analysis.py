"""Attractive-feature profiles: per-model matrix and cross-model comparison.

Per model (results_dir/profile/concepts.csv, graded):
  default      base rate in unexposed solo concepts
  drift_rise   mean rate in the last 3 memory iterations minus iteration 1
  social_rise  round-8 minus round-1 rate in interacting (see-all) teams
  nominal_rise same in revise-alone teams (elaboration control)
  social_index social_rise - nominal_rise
  social_r0    pooled SIS R0 in interacting teams (NaN if too few carriers)
  copy_index   exposed rate when shown as exemplar minus default (neutral instruction)
  copy_ceiling same under the directive instruction (4 fixed + own default only)
  disposition  copy_ceiling - copy_index: how much the model withholds unless told
  base_avoid   for the base concept's own features: rate under exposure minus default
Each property is computed per half; the profile reports the mean and the
Spearman correlation between halves (split-half reliability).

Cross-model (a directory of per-model result dirs): heatmaps, Spearman
matrices, consensus/idiosyncratic sets, and hypotheses P1-P4.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .schema import SCHEMA, FACETS

FEATS = [f.fid for f in SCHEMA.features]
PROPS = ["default", "drift_rise", "social_rise", "nominal_rise", "social_index", "social_r0", "copy_index", "copy_ceiling", "disposition"]
MIN_EXPRESSING = 15


def _sets(col):
    return col.apply(lambda s: set(json.loads(s)) if isinstance(s, str) else set())


def _rates(sets) -> pd.Series:
    n = max(1, len(sets))
    return pd.Series({f: sum(f in s for s in sets) / n for f in FEATS})


def _spearman(a: pd.Series, b: pd.Series) -> float:
    d = pd.concat([a, b], axis=1).dropna()
    if len(d) < 4 or d.iloc[:, 0].std() == 0 or d.iloc[:, 1].std() == 0:
        return np.nan
    return float(d.iloc[:, 0].rank().corr(d.iloc[:, 1].rank()))


def _sis_r0(c: pd.DataFrame, ex: pd.DataFrame, T: int) -> pd.Series:
    """Pooled R0 per feature over the given interacting teams."""
    st = {(r["half"], r["rep"], int(r["round"]), int(r["agent"])): r for _, r in c.iterrows()}
    out = {}
    for f in FEATS:
        a_attr = a_sp = inf_sus = inf = rec = risk = expr = 0
        src = {}
        if ex is not None and len(ex):
            for h, rep_, rr, vw, sv in zip(ex["half"], ex["rep"], ex["round"], ex["viewer"], ex["source"]):
                if rr >= 2:
                    s = st.get((h, rep_, int(rr) - 1, int(sv)))
                    src[(h, rep_, int(rr), int(vw))] = src.get((h, rep_, int(rr), int(vw)), 0) + (int(f in s["fs"]) if s is not None else 0)
        for (h, rep_, rr, ag), r in st.items():
            has = f in r["fs"]; prev = st.get((h, rep_, rr - 1, ag)); k = src.get((h, rep_, rr, ag), 0)
            inf += k; expr += int(has)
            if prev is None or f not in prev["fs"]:
                inf_sus += k
                if has:
                    if k > 0: a_attr += 1
                    else: a_sp += 1
            else:
                risk += 1
                if not has: rec += 1
        if expr < MIN_EXPRESSING or inf_sus == 0 or risk == 0:
            out[f] = np.nan; continue
        p = (a_attr + 0.5) / (inf_sus + 1); cc = inf / expr; g = (rec + 0.5) / (risk + 1)
        out[f] = p * cc / g if g > 0 else np.nan
    return pd.Series(out)


def profile_model(results_dir: str, figures_dir: str | None = None, name: str | None = None) -> dict:
    d = os.path.join(results_dir, "profile")
    c = pd.read_csv(os.path.join(d, "concepts.csv"))
    c["fs"] = _sets(c["features"])
    if c["fs"].apply(len).sum() == 0:
        raise RuntimeError(f"{d}: concepts are ungraded; run rescore with the grader model first")
    exp = os.path.join(d, "exposures.csv")
    ex = pd.read_csv(exp) if os.path.exists(exp) else None
    name = name or (json.load(open(os.path.join(d, "model.json")))["model"] if os.path.exists(os.path.join(d, "model.json")) else os.path.basename(results_dir))
    cand = json.load(open(os.path.join(d, "scan_candidates.json"))) if os.path.exists(os.path.join(d, "scan_candidates.json")) else {}
    halves = sorted(c["half"].unique())
    per_half = {}
    for h in halves:
        ch = c[c["half"] == h]
        pr = ch[ch.component == "prior"]; dr = ch[ch.component == "drift"]
        so = ch[(ch.component == "social") & (ch["mode"] == "interacting")]
        no = ch[(ch.component == "social") & (ch["mode"] == "nominal")]
        sc_all = ch[ch.component == "scan"]
        sc = sc_all[sc_all["mode"].isin(["exposed", "neutral"])]
        scd = sc_all[sc_all["mode"] == "directive"]
        T = int(so["round"].max()) if len(so) else 0
        row = {"default": _rates(pr["fs"])}
        if len(dr):
            last = dr[dr["round"] >= max(1, dr["round"].max() - 2)]
            row["drift_rise"] = _rates(last["fs"]) - _rates(dr[dr["round"] == 1]["fs"])
        if len(so):
            row["social_rise"] = _rates(so[so["round"] == T]["fs"]) - _rates(so[so["round"] == 1]["fs"])
            exh = ex[(ex["half"] == h) & (ex["mode"] == "interacting")] if ex is not None else None
            row["social_r0"] = _sis_r0(so, exh, T)
        if len(no):
            Tn = int(no["round"].max())
            row["nominal_rise"] = _rates(no[no["round"] == Tn]["fs"]) - _rates(no[no["round"] == 1]["fs"])
        if "social_rise" in row and "nominal_rise" in row:
            row["social_index"] = row["social_rise"] - row["nominal_rise"]
        if len(sc):
            ci = pd.Series(np.nan, index=FEATS)
            for feat, g in sc.groupby("exemplar"):
                ci[feat] = np.mean([feat in s for s in g["fs"]]) - row["default"][feat]
            row["copy_index"] = ci
            if len(scd):
                cc = pd.Series(np.nan, index=FEATS)
                for feat, g in scd.groupby("exemplar"):
                    cc[feat] = np.mean([feat in s for s in g["fs"]]) - row["default"][feat]
                row["copy_ceiling"] = cc
                row["disposition"] = cc - ci
            # avoidance of the base exemplar's own features (pooled over all exemplars)
            base_feats = ["whisk_coil", "battery_motor", "stainless_steel", "one_touch_auto"]
            row["base_avoid"] = pd.Series({f: (np.mean([f in s for s in sc["fs"]]) - row["default"][f]) if f in base_feats else np.nan for f in FEATS})
        per_half[h] = row
    props = [p for p in PROPS + ["base_avoid"] if all(p in per_half[h] for h in halves)]
    mat = pd.DataFrame(index=FEATS)
    mat["facet"] = [SCHEMA[f].facet for f in FEATS]
    rel = {}
    for p in props:
        cols = pd.concat([per_half[h][p].rename(f"{p}_h{h}") for h in halves], axis=1)
        mat[p] = cols.mean(axis=1)
        for cc in cols:
            mat[cc] = cols[cc]
        rel[p] = _spearman(cols.iloc[:, 0], cols.iloc[:, 1]) if len(halves) >= 2 else np.nan
    # classification
    def cls(r):
        tags = []
        if r["default"] >= 0.30: tags.append("default")
        if r.get("social_index", 0) >= 0.10 or (np.isfinite(r.get("social_r0", np.nan)) and r["social_r0"] >= 1 and r["default"] < 0.15): tags.append("social_attractor")
        if r.get("drift_rise", 0) >= 0.10: tags.append("drift_attractor")
        if np.isfinite(r.get("copy_index", np.nan)) and r["copy_index"] >= 0.10: tags.append("copyable")
        if np.isfinite(r.get("base_avoid", np.nan)) and r["base_avoid"] <= -0.15: tags.append("repellent_when_shown")
        return "|".join(tags)
    mat["class"] = mat.apply(cls, axis=1)
    mat.insert(0, "model", name)
    mat.to_csv(os.path.join(d, "profile_matrix.csv"))
    pd.DataFrame([{"model": name, **{f"rel_{k}": v for k, v in rel.items()}}]).to_csv(os.path.join(d, "profile_reliability.csv"), index=False)
    summary = {"model": name, "n_concepts": int(len(c)), "reliability": rel,
               "defaults": mat[mat["class"].str.contains("default")].index.tolist(),
               "social_attractors": mat[mat["class"].str.contains("social_attractor")].index.tolist(),
               "drift_attractors": mat[mat["class"].str.contains("drift_attractor")].index.tolist(),
               "copyable": mat[mat["class"].str.contains("copyable")].index.tolist(),
               "repellent": mat[mat["class"].str.contains("repellent")].index.tolist(),
               "scan": cand}
    with open(os.path.join(d, "profile_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    if figures_dir:
        os.makedirs(figures_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(7, 8))
        show = [p for p in ("default", "drift_rise", "social_index", "social_r0", "copy_index", "copy_ceiling") if p in mat]
        m = mat[show].copy()
        m["social_r0"] = m.get("social_r0", pd.Series(dtype=float)).clip(upper=3) / 3 if "social_r0" in m else np.nan
        im = ax.imshow(m.values.astype(float), aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_yticks(range(len(FEATS))); ax.set_yticklabels(FEATS, fontsize=6)
        ax.set_xticks(range(len(show))); ax.set_xticklabels(show, rotation=30, ha="right", fontsize=8)
        ax.set_title(f"Attractiveness profile: {name}\n(R0 shown as R0/3, capped)", fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.03)
        fig.tight_layout(); fig.savefig(os.path.join(figures_dir, f"profile_{_safe(name)}.png"), dpi=200); plt.close(fig)
    return summary


def _safe(s):
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in s)


# ==========================================================================
def cross_model(profile_root: str, figures_dir: str, meta_path: str | None = None, report_path: str | None = None):
    """profile_root contains one results dir per model, each with profile/profile_matrix.csv."""
    mats, rels = [], []
    for name in sorted(os.listdir(profile_root)):
        p = os.path.join(profile_root, name, "profile", "profile_matrix.csv")
        if os.path.exists(p):
            m = pd.read_csv(p, index_col=0); m["model"] = name; mats.append(m)
            r = os.path.join(profile_root, name, "profile", "profile_reliability.csv")
            if os.path.exists(r):
                rr = pd.read_csv(r); rr["model"] = name; rels.append(rr)
    if not mats:
        print("[cross] no profiles found"); return
    os.makedirs(figures_dir, exist_ok=True)
    models = [m["model"].iloc[0] for m in mats]
    meta = pd.read_csv(meta_path, sep="\t").set_index("name") if meta_path and os.path.exists(meta_path) else None
    lines = ["# Cross-model attractiveness report", f"\nmodels: {', '.join(models)}\n"]
    rel = pd.concat(rels) if rels else None
    if rel is not None:
        rel.to_csv(os.path.join(profile_root, "reliability_all.csv"), index=False)
        lines.append("## Split-half reliability (Spearman between halves)\n")
        try: lines.append(rel.set_index("model").round(2).to_markdown())
        except ImportError: lines.append("```\n" + rel.set_index("model").round(2).to_string() + "\n```")
    props = [p for p in ("default", "drift_rise", "social_index", "social_r0", "copy_index", "copy_ceiling", "disposition") if all(p in m for m in mats)]
    for p in props:
        wide = pd.DataFrame({m["model"].iloc[0]: m[p] for m in mats})
        wide.to_csv(os.path.join(profile_root, f"cross_{p}.csv"))
        # heatmap
        fig, ax = plt.subplots(figsize=(1.2 + 0.55 * len(models), 8))
        v = wide.values.astype(float)
        if p == "social_r0":
            v = np.clip(v, 0, 3); im = ax.imshow(v, aspect="auto", cmap="Reds", vmin=0, vmax=3)
        elif p == "default":
            im = ax.imshow(v, aspect="auto", cmap="Blues", vmin=0, vmax=1)
        else:
            im = ax.imshow(v, aspect="auto", cmap="RdBu_r", vmin=-0.6, vmax=0.6)
        ax.set_yticks(range(len(wide))); ax.set_yticklabels(wide.index, fontsize=6)
        ax.set_xticks(range(len(models))); ax.set_xticklabels(models, rotation=45, ha="right", fontsize=7)
        ax.set_title(f"{p} by model and feature", fontsize=9); plt.colorbar(im, ax=ax, fraction=0.04)
        fig.tight_layout(); fig.savefig(os.path.join(figures_dir, f"cross_{p}.png"), dpi=200); plt.close(fig)
        # model x model spearman
        S = pd.DataFrame(index=models, columns=models, dtype=float)
        for a in models:
            for b in models:
                S.loc[a, b] = _spearman(wide[a], wide[b])
        S.to_csv(os.path.join(profile_root, f"cross_{p}_spearman.csv"))
        off = S.values[~np.eye(len(models), dtype=bool)]
        ceiling = rel[f"rel_{p}"].mean() if rel is not None and f"rel_{p}" in rel else np.nan
        lines.append(f"\n## {p}\n\n- mean between-model Spearman = {np.nanmean(off):.2f} (within-model split-half ceiling = {ceiling:.2f})")
        # consensus / idiosyncratic
        thr = {"default": 0.30, "drift_rise": 0.10, "social_index": 0.10, "social_r0": 1.0, "copy_index": 0.10, "copy_ceiling": 0.50, "disposition": 0.50}[p]
        hits = (wide >= thr)
        k = hits.sum(axis=1)
        cons = k[k >= max(2, int(np.ceil(0.75 * len(models))))].index.tolist()
        idio = k[k == 1].index.tolist()
        lines.append(f"- consensus (>=75% of models above {thr}): {cons if cons else 'none'}")
        lines.append(f"- idiosyncratic (exactly one model): {idio if idio else 'none'}")
        if meta is not None and "family" in meta and len(models) >= 4:
            same_fam, diff_fam = [], []
            for i, a in enumerate(models):
                for b in models[i + 1:]:
                    if a in meta.index and b in meta.index:
                        (same_fam if meta.loc[a, "family"] == meta.loc[b, "family"] else diff_fam).append(S.loc[a, b])
            if same_fam and diff_fam:
                lines.append(f"- P3 family effect: mean Spearman same-family {np.nanmean(same_fam):.2f} vs different-family {np.nanmean(diff_fam):.2f} (n pairs {len(same_fam)}/{len(diff_fam)})")
        if meta is not None and "reasoning" in meta and len(models) >= 2:
            pass
    # clustering dendrogram on the concatenated profile
    try:
        from scipy.cluster.hierarchy import linkage, dendrogram
        allp = pd.concat([pd.DataFrame({m["model"].iloc[0]: m[p] for m in mats}).add_prefix(p + ":") for p in props], axis=1)
        X = allp.T.fillna(0).values
        X = (X - X.mean(0)) / (X.std(0) + 1e-9)
        Z = linkage(X, "average", metric="correlation")
        fig, ax = plt.subplots(figsize=(7, 3.5))
        dendrogram(Z, labels=allp.columns.tolist(), ax=ax, leaf_rotation=45, leaf_font_size=7)
        ax.set_title("Models clustered by attractiveness profile (correlation distance, average linkage)", fontsize=9)
        fig.tight_layout(); fig.savefig(os.path.join(figures_dir, "cross_dendrogram.png"), dpi=200); plt.close(fig)
    except Exception as e:      # noqa: BLE001
        lines.append(f"\n(dendrogram skipped: {e!r})")
    lines.append("\n## Hypotheses\n")
    lines.append("- P1 defaults shared across models: see consensus list under `default`.")
    lines.append("- P2 social attractors partly shared, partly model-specific: see consensus and idiosyncratic lists under `social_index` / `social_r0`.")
    lines.append("- P3 family > size: see the family-effect line under each property (requires models.tsv).")
    lines.append("- P4 reasoning toggle changes copy_index more than default: compare the Qwen3-32B thinking-off/on rows in cross_copy_index.csv vs cross_default.csv.")
    out = report_path or os.path.join(profile_root, "CROSS_MODEL_REPORT.md")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[cross] wrote {out} and figures to {figures_dir}")
