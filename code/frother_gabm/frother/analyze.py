"""Figures, tables and a hypothesis report from the live CSVs.

Run any time, even while jobs are still running: it uses whatever rows exist.
Outputs:
  figures/<job>_*.png|.pdf
  results/<job>/table_*.csv
  results/REPORT.md  (one line per hypothesis with numbers and a verdict)
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .metrics import bootstrap_mean, contrast
from .schema import S1_BUNDLE, S1_FLAW, SCHEMA

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 120, "savefig.dpi": 300, "axes.titlesize": 10,
                     "legend.frameon": False, "font.family": "DejaVu Sans"})
PAL = {"neutral": "#4C72B0", "verbatim": "#C44E52", "seeded": "#DD8452", "control": "#8C8C8C",
       "nominal": "#55A868", "interacting": "#8172B2", "propose_next": "#937860", "revise": "#4C72B0"}
REPORT = []


def _save(fig, figdir, name):
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, name + ".png"))
    fig.savefig(os.path.join(figdir, name + ".pdf"))
    plt.close(fig)


def _load(results_dir, job, name="units.csv"):
    p = os.path.join(results_dir, job, name)
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        return None
    df = pd.read_csv(p)
    if "features" in df.columns:
        df["features"] = df["features"].apply(lambda s: set(json.loads(s)) if isinstance(s, str) else set())
    return df


def _verdict(c, direction="+"):
    if not np.isfinite(c.get("g", np.nan)):
        if c.get("n_a", 0) >= 2 and c.get("n_b", 0) >= 2 and c.get("diff") == 0:
            return "no difference (cells identical)"
        return "insufficient data"
    if c["excludes_zero"] and ((direction == "+" and c["diff"] > 0) or (direction == "-" and c["diff"] < 0)):
        return "SUPPORTED"
    if c["excludes_zero"]:
        return "REVERSED"
    return "not supported (CI spans 0)"


def _rep(h, label, c, direction="+"):
    line = (f"| {h} | {label} | {c['diff']:+.3f} | [{c['lo']:+.3f}, {c['hi']:+.3f}] | "
            f"{c['g']:.2f} | {c['n_a']}/{c['n_b']} | {_verdict(c, direction)} |")
    REPORT.append(line)
    return c


def _bar_ci(ax, labels, means, los, his, colors, ylabel, title=""):
    x = np.arange(len(labels))
    ax.bar(x, means, color=colors, width=0.7)
    ax.errorbar(x, means, yerr=[np.array(means) - np.array(los), np.array(his) - np.array(means)],
                fmt="none", ecolor="k", capsize=3, lw=1)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel(ylabel); ax.set_title(title)
    ax.axhline(0, color="k", lw=0.6)


# ==========================================================================
# ==========================================================================
def analyze_study1(rd, fd, nb):
    u = _load(rd, "study1")
    if u is None:
        return
    REPORT.append("\n### Study 1 - individual fixation\n")
    REPORT.append("| H | contrast | diff | 95% CI | g | n | verdict |\n|---|---|---|---|---|---|---|")
    rows = []
    for brief in ("neutral", "verbatim"):
        d = u[u["brief"] == brief]
        base = d[d["exposure"] == "none"]
        if base.empty:
            continue
        b_hits, b_flaw = base["marker_hits"].mean(), base["flaw"].mean()
        for ex in ("none", "single", "triple", "memory"):
            s = d[d["exposure"] == ex]
            if s.empty:
                continue
            m, lo, hi = bootstrap_mean(s["marker_hits"] - b_hits, nb)
            rows.append({"brief": brief, "exposure": ex, "n": len(s),
                         "markers_per_concept": s["marker_hits"].mean(),
                         "excess": m, "excess_lo": lo, "excess_hi": hi,
                         "full_bundle": s["full_bundle"].mean(), "flaw": s["flaw"].mean(),
                         "self_first_sim": s["self_first_sim"].mean(), "dpp": s["dpp"].mean(),
                         **{f"adopt_{k}": s[f"adopt_{k}"].mean() for k in S1_BUNDLE}})
        for ex, h in (("single", "H1a"), ("triple", "H1a")):
            s = d[d["exposure"] == ex]
            _rep(h, f"{brief}: {ex} vs none (excess adoption)",
                 contrast(s["marker_hits"], base["marker_hits"], nb))
        _rep("H1a'", f"{brief}: triple vs single (dose)",
             contrast(d[d["exposure"] == "triple"]["marker_hits"], d[d["exposure"] == "single"]["marker_hits"], nb))
        for ex in ("single", "triple"):
            s = d[d["exposure"] == ex]
            _rep("H1c", f"{brief}: flaw reuse {ex} vs none (expect NO effect)",
                 contrast(s["flaw"], base["flaw"], nb))
        mem = d[d["exposure"] == "memory"]
        _rep("H1d", f"{brief}: memory vs none, marker adoption (feature space)",
             contrast(mem["marker_hits"], base["marker_hits"], nb))
        _rep("H1d", f"{brief}: memory vs none, self-similarity (embedding space)",
             contrast(mem["self_first_sim"], base["self_first_sim"], nb))
    # brief moderation: excess(neutral) - excess(verbatim) at each dose
    for ex in ("single", "triple"):
        a = u[(u.brief == "neutral") & (u.exposure == ex)]["marker_hits"] - u[(u.brief == "neutral") & (u.exposure == "none")]["marker_hits"].mean()
        b = u[(u.brief == "verbatim") & (u.exposure == ex)]["marker_hits"] - u[(u.brief == "verbatim") & (u.exposure == "none")]["marker_hits"].mean()
        _rep("H1b", f"brief moderation at {ex}: excess(neutral) - excess(verbatim)", contrast(a, b, nb))
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(rd, "study1", "table_cells.csv"), index=False)

    # Fig 1: excess adoption
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3))
    ax = axes[0]
    w = 0.38
    for i, brief in enumerate(("neutral", "verbatim")):
        t = tab[tab.brief == brief].set_index("exposure").reindex(["none", "single", "triple", "memory"])
        x = np.arange(4) + (i - 0.5) * w
        ax.bar(x, t["excess"], w, color=PAL[brief], label=brief)
        ax.errorbar(x, t["excess"], yerr=[t["excess"] - t["excess_lo"], t["excess_hi"] - t["excess"]],
                    fmt="none", ecolor="k", capsize=2, lw=0.8)
    ax.set_xticks(range(4)); ax.set_xticklabels(["none", "1\nexample", "3\nexamples", "memory\n(no example)"])
    ax.axhline(0, color="k", lw=0.6); ax.set_ylabel("excess marker adoption (markers/concept)")
    ax.set_title("(a) exposure raises bundle adoption"); ax.legend()
    ax = axes[1]
    for i, k in enumerate(list(S1_BUNDLE) + [S1_FLAW]):
        col = f"adopt_{k}" if k != S1_FLAW else "flaw"
        t = tab[tab.brief == "neutral"].set_index("exposure").reindex(["none", "single", "triple"])
        ax.plot(range(3), t[col], marker="o", label=k, color="#C44E52" if k == S1_FLAW else None,
                ls="--" if k == S1_FLAW else "-")
    ax.set_xticks(range(3)); ax.set_xticklabels(["none", "1 ex.", "3 ex."]); ax.set_ylim(0, 1)
    ax.set_ylabel("adoption rate (neutral brief)"); ax.set_title("(b) functional features vs the flaw"); ax.legend(fontsize=7)
    ax = axes[2]
    t = tab.set_index(["brief", "exposure"])
    labs, ms, los, his, cols = [], [], [], [], []
    for brief in ("neutral", "verbatim"):
        for ex in ("none", "memory"):
            if (brief, ex) in t.index:
                s = u[(u.brief == brief) & (u.exposure == ex)]["self_first_sim"]
                m, lo, hi = bootstrap_mean(s, nb)
                labs.append(f"{brief}\n{ex}"); ms.append(m); los.append(lo); his.append(hi); cols.append(PAL[brief])
    _bar_ci(ax, labs, ms, los, his, cols, "cosine to own first concept", "(c) memory: embedding self-similarity")
    if ms:
        ax.set_ylim(min(los) - 0.05, max(his) + 0.05)
    _save(fig, fd, "study1_fixation")

    # Fig 2: iteration-resolved adoption
    c = _load(rd, "study1", "concepts.csv")
    if c is not None:
        fig, ax = plt.subplots(figsize=(5, 3))
        for brief in ("neutral", "verbatim"):
            for ex, ls in (("none", ":"), ("single", "--"), ("triple", "-"), ("memory", "-.")):
                s = c[(c.brief == brief) & (c.exposure == ex)].groupby("iteration")["marker_hits"].mean()
                if len(s):
                    ax.plot(s.index, s.values, ls=ls, color=PAL[brief], label=f"{brief}/{ex}")
        ax.set_xlabel("iteration within run"); ax.set_ylabel("markers per concept")
        ax.set_title("Study 1: adoption across iterations"); ax.legend(fontsize=6, ncol=2)
        _save(fig, fd, "study1_iterations")
        # full-bundle table
        fb = c.groupby(["brief", "exposure"]).agg(full_bundle=("marker_hits", lambda x: (x == 3).mean()),
                                                  any_marker=("marker_hits", lambda x: (x >= 1).mean()),
                                                  n=("marker_hits", "size")).reset_index()
        fb.to_csv(os.path.join(rd, "study1", "table_bundle_vs_attribute.csv"), index=False)



# ==========================================================================
# ==========================================================================
def analyze_P(rd, fd, nb):
    u = _load(rd, "P")
    if u is None:
        return
    REPORT.append("\n### P - permission test (is avoidance an instruction effect?)\n")
    REPORT.append("| H | contrast | diff | 95% CI | g | n | verdict |\n|---|---|---|---|---|---|---|")
    base = u[u.mechanism_shown == "none"]
    cell = lambda m, i: u[(u.mechanism_shown == m) & (u.instruction == i)]
    for instr in ("neutral", "permissive", "directive"):
        _rep("H7a", f"whisk shown, {instr}: whisk use vs baseline", contrast(cell("common", instr)["has_whisk_coil"], base["has_whisk_coil"], nb), "-")
    _rep("H7a", "whisk shown: permissive vs neutral, whisk use (expect higher if instruction-driven)",
         contrast(cell("common", "permissive")["has_whisk_coil"], cell("common", "neutral")["has_whisk_coil"], nb))
    _rep("H7c", "whisk shown: directive vs permissive, whisk use", contrast(cell("common", "directive")["has_whisk_coil"], cell("common", "permissive")["has_whisk_coil"], nb))
    for instr in ("neutral", "permissive", "directive"):
        _rep("H7b", f"venturi shown, {instr}: venturi use vs baseline", contrast(cell("rare", instr)["has_aerator_nozzle"], base["has_aerator_nozzle"], nb))
    _rep("H7b", "venturi shown: permissive vs neutral, venturi use", contrast(cell("rare", "permissive")["has_aerator_nozzle"], cell("rare", "neutral")["has_aerator_nozzle"], nb))
    _rep("H7c", "venturi shown: directive vs permissive, venturi use", contrast(cell("rare", "directive")["has_aerator_nozzle"], cell("rare", "permissive")["has_aerator_nozzle"], nb))
    for instr in ("neutral", "permissive", "directive"):
        _rep("H7b", f"heater (in both exemplars), {instr}: heater use vs baseline",
             contrast(u[(u.mechanism_shown != "none") & (u.instruction == instr)]["has_integrated_heating"], base["has_integrated_heating"], nb))
    tab = u.groupby(["mechanism_shown", "instruction"]).agg(n=("rep", "size"), whisk=("has_whisk_coil", "mean"),
        venturi=("has_aerator_nozzle", "mean"), impeller=("has_impeller_turbine", "mean"), heater=("has_integrated_heating", "mean"),
        steel=("has_stainless_steel", "mean"), battery=("has_battery_motor", "mean"), one_button=("has_one_touch_auto", "mean"),
        features=("n_features", "mean")).reset_index()
    tab.to_csv(os.path.join(rd, "P", "table_cells.csv"), index=False)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    order = [("none", "none"), ("common", "neutral"), ("common", "permissive"), ("common", "directive"),
             ("rare", "neutral"), ("rare", "permissive"), ("rare", "directive")]
    labs = ["baseline", "whisk\nneutral", "whisk\npermissive", "whisk\ndirective", "venturi\nneutral", "venturi\npermissive", "venturi\ndirective"]
    cols = ["#8C8C8C", "#C44E52", "#E07B7B", "#F2B5B5", "#4C72B0", "#7FA1D3", "#B9CDE8"]
    for ax, col, ttl in zip(axes, ("has_whisk_coil", "has_aerator_nozzle", "has_integrated_heating"),
                            ("(a) whisk use", "(b) venturi use", "(c) heater use (in both exemplars)")):
        ms, los, his = [], [], []
        for m, i in order:
            g = base if m == "none" else cell(m, i)
            mm, lo, hi = bootstrap_mean(g[col], nb); ms.append(mm); los.append(lo); his.append(hi)
        _bar_ci(ax, labs, ms, los, his, cols, "share of concepts", ttl); ax.tick_params(axis="x", labelsize=6); ax.set_ylim(0, 1)
    _save(fig, fd, "P_permission")
    u["copy_language"] = u["rationale"].fillna("").str.lower().str.contains(r"build on|building on|inspired by|borrow|adapt|similar to|the example|reuse|keep").astype(int)
    u.groupby(["mechanism_shown", "instruction"])["copy_language"].mean().reset_index().to_csv(os.path.join(rd, "P", "table_rationale_language.csv"), index=False)

# ==========================================================================
# ==========================================================================
# ==========================================================================
# ==========================================================================
def analyze_D2(rd, fd, nb):
    u = _load(rd, "D2")
    if u is None:
        return
    REPORT.append("\n### D2 - convergence vs contagion (revise instruction, no seed)\n")
    REPORT.append("| H | contrast | diff | 95% CI | g | n | verdict |\n|---|---|---|---|---|---|---|")
    cell = lambda m: u[u["mode"] == m]
    for m in ("int3", "int7"):
        if cell(m).empty or cell("nominal").empty:
            continue
        _rep("H6a", f"{m} vs nominal: feature entropy at last round (expect lower)", contrast(cell(m)["entropy_last"], cell("nominal")["entropy_last"], nb), "-")
        _rep("H6a", f"{m} vs nominal: DPP diversity at last round (expect lower)", contrast(cell(m)["dpp_last"], cell("nominal")["dpp_last"], nb), "-")
        _rep("H6a", f"{m} vs nominal: unique features at last round (expect fewer)", contrast(cell(m)["unique_last"], cell("nominal")["unique_last"], nb), "-")
    if not cell("int7").empty and not cell("int3").empty:
        _rep("H6c", "int7 vs int3: feature entropy at last round (expect lower)", contrast(cell("int7")["entropy_last"], cell("int3")["entropy_last"], nb), "-")
    for m in ("nominal", "int3", "int7"):
        if not cell(m).empty:
            _rep("H6a", f"{m}: entropy last round vs round 1 (within-team change)", contrast(cell(m)["entropy_last"], cell(m)["entropy_round1"], nb), "-")
    tab = u.groupby("mode").agg(n=("rep", "size"), entropy_round1=("entropy_round1", "mean"), entropy_last=("entropy_last", "mean"),
                                dpp_round1=("dpp_round1", "mean"), dpp_last=("dpp_last", "mean"),
                                unique_last=("unique_last", "mean"), mean_features_last=("mean_features_last", "mean")).reset_index()
    tab.to_csv(os.path.join(rd, "D2", "table_cells.csv"), index=False)
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3))
    for ax, col, ttl in zip(axes, ("entropy_last", "dpp_last", "unique_last"),
                            ("(a) feature entropy, last round", "(b) DPP diversity, last round", "(c) unique features, last round")):
        labs, ms, los, his = [], [], [], []
        for m in ("nominal", "int3", "int7"):
            if cell(m).empty:
                continue
            mm, lo, hi = bootstrap_mean(cell(m)[col], nb); labs.append(m); ms.append(mm); los.append(lo); his.append(hi)
        _bar_ci(ax, labs, ms, los, his, [PAL["nominal"], "#B39DDB", PAL["interacting"]][:len(labs)], "", ttl)
        if ms:
            ax.set_ylim(min(los) - 0.1 * abs(min(los)), max(his) + 0.1 * abs(max(his)))
    _save(fig, fd, "D2_convergence")


def analyze_all(cfg, jobs=None):
    global REPORT
    REPORT = ["# Hypothesis report", f"\nmodel: `{cfg.model}`  extractor: `{cfg.extract_model}`  "
              f"embeddings: `{cfg.embed_model}`\n"]
    rd, fd, nb = cfg.results_dir, cfg.figures_dir, cfg.n_boot
    for name, fn in (("study1", analyze_study1), ("P", analyze_P), ("D2", analyze_D2)):
        if jobs and name not in jobs:
            continue
        try:
            fn(rd, fd, nb)
        except Exception as e:      # noqa: BLE001
            REPORT.append(f"\n[{name}] analysis error: {e!r}")
            print(f"[analyze] {name}: {e!r}")
    from .contagion import endogenous_all
    endogenous_all(cfg, REPORT)
    REPORT.append("\n---\nVerdicts use percentile-bootstrap 95% CIs of the mean difference; g is Hedges' g. "
                  "'SUPPORTED' = CI excludes zero in the predicted direction.")
    with open(os.path.join(rd, "REPORT.md"), "w") as f:
        f.write("\n".join(REPORT) + "\n")
    print(f"[analyze] wrote {os.path.join(rd, 'REPORT.md')} and figures to {fd}/")
