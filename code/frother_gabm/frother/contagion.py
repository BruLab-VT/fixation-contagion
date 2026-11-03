"""Endogenous contagion: treat EVERY feature as a candidate epidemic.

For the team job D2 (and the profile social teams, via profile_analysis) and each cell
(instruction / mode / dose), and for every feature in the schema:
  * prevalence by round, and the round-1 -> last-round change
  * SIS parameters pooled over teams: p (per-contact transmission), c
    (infectious contacts per expressing agent-round), gamma (abandonment),
    R0 = p*c/gamma, plus attributed vs spontaneous adoptions
  * dose-response: P(adopt | susceptible, k visible teammates expressing)
Outputs table_endogenous_<job>.csv and three figures per job. This is where
features the agents themselves favour show up as contagious, whether or not
anything was seeded.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .schema import SCHEMA

MIN_EXPRESSING = 20      # agent-rounds expressing a feature before its R0 is reported


def _cells(df, job):
    if job in ("D", "D2"):
        cols = [c for c in ("instruction", "mode", "dose", "seeded") if c in df.columns]
    else:
        cols = [c for c in ("mode", "seeded") if c in df.columns]
    return cols


def _label(cellvals, cols):
    return " / ".join(f"{c}={v}" for c, v in zip(cols, cellvals))


def endogenous_for_job(results_dir, figures_dir, job, report: list):
    d = os.path.join(results_dir, job)
    cp, ep = os.path.join(d, "concepts.csv"), os.path.join(d, "exposures.csv")
    if not os.path.exists(cp):
        return
    c = pd.read_csv(cp)
    c["features"] = c["features"].apply(lambda s: set(json.loads(s)) if isinstance(s, str) else set())
    ex = pd.read_csv(ep) if os.path.exists(ep) else None
    cols = _cells(c, job)
    T = int(c["round"].max())
    feats = [f.fid for f in SCHEMA.features]
    for f in feats:
        c[f] = c["features"].apply(lambda s, f=f: int(f in s))
    rows, dose_rows, cell_frames = [], [], {}
    for cellvals, g in c.groupby(cols):
        cellvals = cellvals if isinstance(cellvals, tuple) else (cellvals,)
        lab = _label(cellvals, cols)
        cell_frames[lab] = g
        # state lookup
        st = {(r["rep"], int(r["round"]), int(r["agent"])): r for _, r in g.iterrows()}
        e = None
        if ex is not None:
            m = np.ones(len(ex), dtype=bool)
            for col, v in zip(cols, cellvals):
                if col in ex.columns:
                    m &= (ex[col] == v).values
            e = ex[m]
        for f in feats:
            prev = g.groupby("round")[f].mean()
            n_expr = int(g[f].sum())
            row = {"cell": lab, **dict(zip(cols, cellvals)), "feature": f, "facet": SCHEMA[f].facet,
                   "rare_at_start": bool(prev.get(1, 1.0) < 0.15),
                   "prev_round1": prev.get(1, np.nan), "prev_last": prev.get(T, np.nan),
                   "rise": prev.get(T, np.nan) - prev.get(1, np.nan), "peak": prev.max(),
                   "n_expressing": n_expr}
            if e is not None and len(e):
                n_adopt_attr = n_adopt_spont = n_inf_sus = n_inf = n_rec = n_risk = 0
                # per viewer-round exposure counts
                src_expr = {}
                for rr, rep_, vw, sv in zip(e["round"], e["rep"], e["viewer"], e["source"]):
                    if rr >= 2:
                        s = st.get((rep_, int(rr) - 1, int(sv)))
                        src_expr[(rep_, int(rr), int(vw))] = src_expr.get((rep_, int(rr), int(vw)), 0) + (int(s[f]) if s is not None else 0)
                for (rep_, rr, ag), r in st.items():
                    prevs = st.get((rep_, rr - 1, ag))
                    inf = src_expr.get((rep_, rr, ag), 0)
                    n_inf += inf
                    if prevs is None or not prevs[f]:
                        n_inf_sus += inf
                        if r[f]:
                            if inf > 0:
                                n_adopt_attr += 1
                            else:
                                n_adopt_spont += 1
                        if rr >= 2:
                            dose_rows.append({"cell": lab, "feature": f, "k": inf, "adopt": int(r[f])})
                    else:
                        n_risk += 1
                        if not r[f]:
                            n_rec += 1
                p = (n_adopt_attr + 0.5) / (n_inf_sus + 1) if n_inf_sus > 0 else np.nan
                cc = n_inf / n_expr if n_expr > 0 else np.nan
                gm = (n_rec + 0.5) / (n_risk + 1) if n_risk > 0 else np.nan
                r0 = p * cc / gm if np.isfinite(p) and np.isfinite(cc) and np.isfinite(gm) and gm > 0 else np.nan
                row.update({"p": p, "c": cc, "gamma": gm, "r0": r0 if n_expr >= MIN_EXPRESSING else np.nan,
                            "adopt_attributed": n_adopt_attr, "adopt_spontaneous": n_adopt_spont,
                            "share_attributed": n_adopt_attr / max(1, n_adopt_attr + n_adopt_spont)})
            rows.append(row)
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(d, f"table_endogenous_{job}.csv"), index=False)
    dose = pd.DataFrame(dose_rows)
    if len(dose):
        dose["k_bin"] = pd.cut(dose["k"], [-1, 0, 1, 2, 4, 99], labels=["0", "1", "2", "3-4", "5+"])
        dt = dose.groupby(["cell", "feature", "k_bin"], observed=True)["adopt"].agg(["mean", "size"]).reset_index()
        dt.to_csv(os.path.join(d, f"table_dose_response_{job}.csv"), index=False)

    # ---- report ----------------------------------------------------------
    report.append(f"\n### Endogenous contagion - job {job} (all features)\n")
    for lab, g in tab.groupby("cell"):
        top = g.sort_values("rise", ascending=False).head(4)
        bot = g.sort_values("rise").head(2)
        line = f"- **{lab}**: risers " + ", ".join(f"{r.feature} ({r.prev_round1:.2f}→{r.prev_last:.2f})" for r in top.itertuples())
        line += "; fallers " + ", ".join(f"{r.feature} ({r.prev_round1:.2f}→{r.prev_last:.2f})" for r in bot.itertuples())
        if "r0" in g and g["r0"].notna().any():
            crit = g[(g["r0"] >= 1) & (g["prev_round1"] < 0.15)].sort_values("r0", ascending=False)
            held = g[(g["r0"] >= 1) & (g["prev_round1"] >= 0.15)].sort_values("r0", ascending=False)
            line += (f"; **emergent features with R0≥1** (rare at round 1, spread through the team): " +
                     ", ".join(f"{r.feature} (R0={r.r0:.2f}, p={r.p:.2f}, γ={r.gamma:.2f}, {r.share_attributed:.0%} of adoptions right after exposure)"
                               for r in crit.itertuples())) if len(crit) else "; no rare feature reaches R0≥1"
            if len(held):
                line += "; default features merely maintained (common at round 1, low abandonment): " + ", ".join(f"{r.feature} (R0={r.r0:.1f})" for r in held.itertuples())
        report.append(line)

    # ---- figures ------------------------------------------------------------
    cells = list(tab["cell"].unique())
    # (1) trajectories of top risers/fallers per cell
    fig, axes = plt.subplots(1, len(cells), figsize=(3.6 * len(cells), 3.2), sharey=True, squeeze=False)
    for ax, lab in zip(axes[0], cells):
        g = tab[tab.cell == lab]
        pick = list(g.sort_values("rise", ascending=False).head(4)["feature"]) + list(g.sort_values("rise").head(2)["feature"])
        sub = cell_frames[lab]
        for f in pick:
            s = sub.groupby("round")[f].mean()
            ax.plot(s.index, s.values, marker="o", ms=3, label=f, ls="-" if g[g.feature == f]["rise"].iloc[0] >= 0 else "--")
        ax.set_title(lab, fontsize=8); ax.set_xlabel("round"); ax.legend(fontsize=6)
    axes[0][0].set_ylabel("share of concepts with feature")
    fig.suptitle(f"{job}: features that rise and fall over rounds (no seed involved)", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(figures_dir, f"{job}_endogenous_trajectories.png")); fig.savefig(os.path.join(figures_dir, f"{job}_endogenous_trajectories.pdf")); plt.close(fig)
    # (2) R0 per feature per cell
    if "r0" in tab and tab["r0"].notna().any():
        fig, axes = plt.subplots(1, len(cells), figsize=(3.6 * len(cells), 3.6), squeeze=False)
        for ax, lab in zip(axes[0], cells):
            g = tab[(tab.cell == lab) & tab["r0"].notna()].sort_values("r0", ascending=False).head(10)
            ax.barh(g["feature"], g["r0"], color=np.where(g["r0"] >= 1, "#C44E52", "#8C8C8C"))
            ax.axvline(1, color="k", ls="--", lw=0.8); ax.invert_yaxis(); ax.tick_params(axis="y", labelsize=7)
            ax.set_title(lab, fontsize=8); ax.set_xlabel("pooled R0")
        fig.suptitle(f"{job}: reproduction number per feature (features with ≥{MIN_EXPRESSING} expressing agent-rounds)", fontsize=10)
        fig.tight_layout(); fig.savefig(os.path.join(figures_dir, f"{job}_endogenous_r0.png")); fig.savefig(os.path.join(figures_dir, f"{job}_endogenous_r0.pdf")); plt.close(fig)
    # (3) dose-response for the top-3 risers per cell
    if len(dose):
        fig, axes = plt.subplots(1, len(cells), figsize=(3.6 * len(cells), 3.2), sharey=True, squeeze=False)
        for ax, lab in zip(axes[0], cells):
            g = tab[tab.cell == lab].sort_values("rise", ascending=False).head(3)
            for f in g["feature"]:
                s = dt[(dt.cell == lab) & (dt.feature == f)]
                ax.plot(range(len(s)), s["mean"], marker="o", ms=3, label=f)
                ax.set_xticks(range(len(s))); ax.set_xticklabels(s["k_bin"].astype(str))
            ax.set_title(lab, fontsize=8); ax.set_xlabel("visible teammates expressing the feature"); ax.legend(fontsize=6)
        axes[0][0].set_ylabel("P(adopt | did not have it)")
        fig.suptitle(f"{job}: dose-response of adoption", fontsize=10)
        fig.tight_layout(); fig.savefig(os.path.join(figures_dir, f"{job}_endogenous_dose_response.png")); fig.savefig(os.path.join(figures_dir, f"{job}_endogenous_dose_response.pdf")); plt.close(fig)


def endogenous_all(cfg, report: list):
    for job in ("D2",):
        try:
            endogenous_for_job(cfg.results_dir, cfg.figures_dir, job, report)
        except Exception as e:      # noqa: BLE001
            report.append(f"\n[endogenous {job}] analysis error: {e!r}")
            print(f"[analyze] endogenous {job}: {e!r}")
