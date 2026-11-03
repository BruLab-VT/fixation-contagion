"""Feature extraction: one JSON call per concept against the fixed checklist.

Simpler than the earlier annotate -> embed -> map pipeline. The cost is that a
schema change means re-extracting (calls are cached, so only new features
cost). The gain is that there is no similarity threshold to calibrate, no
anchor geometry, and no absence-phrase trap.

Two gates protect the measurement:
  * fixture_check()  - extraction recall/precision on hand-written concepts
  * validate_seed()  - every seeded feature is recoverable from its stimulus
"""
from __future__ import annotations

import json
import re

import numpy as np
from concurrent.futures import ThreadPoolExecutor

from .schema import SCHEMA, FACETS, CANDIDATES, S1_BUNDLE, S1_FLAW, stimulus, S1_EXAMPLES, S1_FLAW_SENTENCE

EXTRACT_SYSTEM = (
    "You are a careful annotator of product design concepts. You do not "
    "deliberate; you answer immediately with JSON only."
)

EXTRACT_USER = """Read the milk-frother concept below and decide which of the listed features it explicitly describes. Include a feature only if the concept clearly states it (do not infer). Go facet by facet; a concept may have several features in one facet and none in another.

FEATURE LIST (id: definition)
{checklist}

CONCEPT
{concept}

Answer with JSON only, one key per facet, listing the matching feature ids (empty list if none), exactly like:
{{"mechanism": [], "drive": [], "construction": [], "thermal": [], "interface": []}}"""


def _parse(text: str) -> set[str]:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return set()
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return set()
    if not isinstance(obj, dict):
        return set()
    feats = []
    if "features" in obj:                       # legacy flat form
        v = obj["features"]
        feats += [k for k, b in v.items() if b] if isinstance(v, dict) else list(v or [])
    for facet in FACETS:                        # facet-structured form
        v = obj.get(facet, [])
        if isinstance(v, dict):
            v = [k for k, b in v.items() if b]
        if isinstance(v, str):
            v = [v]
        feats += list(v or [])
    return {f for f in feats if isinstance(f, str) and f in SCHEMA.by_id}


def verify(features: set[str], concept_text: str) -> set[str]:
    """Keep a flagged feature only if the concept text also matches that
    feature's vocabulary pattern. This is the precision layer: the LLM grader
    decides recall, the text decides whether the flag is about words that are
    actually there (e.g. 'transparent plastic' is not a glass vessel)."""
    t = concept_text.lower()
    return {f for f in features if re.search(SCHEMA[f].pattern, t)}


def pattern_features(concept_text: str) -> set[str]:
    """Vocabulary-only detection (no LLM). Used for candidate selection inside
    a run before batched grading; not a substitute for the graded features."""
    t = concept_text.lower()
    return {f.fid for f in SCHEMA.features if re.search(f.pattern, t)}


class Extractor:
    def __init__(self, client, cfg):
        self.client, self.cfg = client, cfg

    def extract_raw(self, concept_text: str, seed: int = 7) -> set[str]:
        """LLM flags only, no text verification (for diagnostics)."""
        msgs = [{"role": "system", "content": EXTRACT_SYSTEM},
                {"role": "user", "content": EXTRACT_USER.format(
                    checklist=SCHEMA.checklist(), concept=concept_text)}]
        raw = self.client.chat(msgs, seed, temperature=0.0, max_tokens=300,
                               json_mode=True, model=self.cfg.extract_model)
        return _parse(raw)

    def extract_one(self, concept_text: str, seed: int = 7) -> set[str]:
        return verify(self.extract_raw(concept_text, seed), concept_text)

    def extract_many(self, texts: list[str], seed: int = 7) -> list[set[str]]:
        with ThreadPoolExecutor(max_workers=max(1, self.cfg.workers)) as ex:
            return list(ex.map(lambda t: self.extract_one(t, seed), texts))


# ==========================================================================
# gates
# ==========================================================================

FIXTURES = [
    ("Concept Title: Steam Jet\nConcept Description: A pressurised steam wand "
     "injects steam into the milk to heat and froth it at once. It runs from a "
     "mains-powered corded base. The jug is stainless steel. A single button "
     "runs the whole automatic cycle.",
     {"steam_injection", "mains_electric", "stainless_steel", "one_touch_auto"}),
    ("Concept Title: Crank Whisk\nConcept Description: A geared hand crank spins "
     "a rotating wire whisk head that beats air into the milk. The vessel is "
     "borosilicate glass with a transparent sight window showing the milk level. "
     "The user turns the crank for thirty seconds.",
     {"hand_crank", "whisk_coil", "glass_vessel", "sight_window"}),
    ("Concept Title: Sonic Pod\nConcept Description: An ultrasonic piezo "
     "transducer in the base cavitates the milk into foam. A USB-rechargeable "
     "lithium battery pack powers it. The body is recycled plastic. A froth "
     "sensor stops the cycle automatically.",
     {"ultrasonic", "usb_rechargeable", "recycled_plastic", "auto_stop_sensing"}),
    ("Concept Title: Magno Jug\nConcept Description: An enclosed bladed impeller "
     "drives a vortex in the milk. A sealed magnetic coupling in the base spins "
     "the impeller with no shaft through the wall. An integrated electric "
     "heating element warms the milk during frothing. The jug is a double-walled "
     "insulated vessel. The user selects a target milk temperature and presses start.",
     {"impeller_turbine", "magnetic_drive", "integrated_heating",
      "double_wall_insulation", "temperature_control"}),
    ("Concept Title: Shake Bottle\nConcept Description: The whole vessel is "
     "shaken back and forth to aerate the milk. The user's own arm motion "
     "provides all the energy. The vessel is a flexible silicone body. The parts "
     "are dishwasher safe.",
     {"shaking_agitation", "silicone_body", "dishwasher_safe"}),
    ("Concept Title: Gas Whip\nConcept Description: A pressurised gas cartridge "
     "supplies the aerating energy, pushing milk through a static shear chamber "
     "that entrains air. It is a slim handheld wand dipped into any cup. It is "
     "controlled from a companion smartphone app.",
     {"pressurised_gas", "shear_chamber", "handheld_wand", "app_connected"}),
]


def fixture_check(extractor: Extractor, verbose=True) -> dict:
    """Recall and precision of the extractor on hand-written concepts."""
    got = extractor.extract_many([t for t, _ in FIXTURES], seed=3)
    tp = fp = fn = 0
    rows = []
    for (text, truth), pred in zip(FIXTURES, got):
        tp += len(truth & pred); fp += len(pred - truth); fn += len(truth - pred)
        rows.append({"missed": sorted(truth - pred), "extra": sorted(pred - truth)})
    recall = tp / max(1, tp + fn)
    precision = tp / max(1, tp + fp)
    ok = recall >= 0.75 and precision >= 0.70
    if verbose:
        print(f"[gate] extractor fixture recall={recall:.2f} precision={precision:.2f} "
              f"-> {'PASS' if ok else 'FAIL'}")
        for i, r in enumerate(rows):
            if r["missed"] or r["extra"]:
                print(f"        fixture {i}: missed={r['missed']} extra={r['extra']}")
    return {"recall": recall, "precision": precision, "pass": ok, "rows": rows}


def validate_seed(extractor: Extractor, attribute: str, verbose=True) -> dict:
    """Stimulus 0 is the one B, C and D show, so the attribute MUST be
    recovered from it (checked under three extraction seeds) and must be absent
    from its matched control. Stimuli 1-2 are only used as extra exemplars in
    the calibration pilot, so misses there are reported, not fatal."""
    base_hits = [attribute in extractor.extract_one(stimulus(0, attribute), seed=s) for s in (11, 12, 13)]
    base_false = [attribute in extractor.extract_one(stimulus(0, None), seed=s) for s in (11, 12, 13)]
    other_hits = [attribute in extractor.extract_one(stimulus(i, attribute), seed=11) for i in (1, 2)]
    ok = all(base_hits) and not any(base_false)
    if verbose:
        print(f"[gate] seed '{attribute}': main stimulus {sum(base_hits)}/3 recovered, "
              f"{sum(base_false)}/3 false hits; extra stimuli {sum(other_hits)}/2 -> "
              f"{'PASS' if ok else 'FAIL'}")
    return {"attribute": attribute, "base_hits": base_hits, "base_false_hits": base_false,
            "other_hits": other_hits, "pass": ok}


def wild_precision_check(extractor: Extractor, texts: list[str], out_csv: str | None = None,
                         verbose=True) -> "pd.DataFrame":
    """Flag-vs-text agreement on the model's OWN concepts (not hand-written
    fixtures). For each feature: how often the LLM flags it, how often the text
    contains its vocabulary, how many flags the verification layer removes.
    Features where the raw flag rate is more than double the text rate are the
    grader's confusions; they are reported so nobody seeds one of them."""
    import pandas as pd
    raws = [extractor.extract_raw(t, seed=5) for t in texts]
    rows = []
    for f in SCHEMA.features:
        flag = np.mean([f.fid in r for r in raws])
        text = np.mean([bool(re.search(f.pattern, t.lower())) for t in texts])
        kept = np.mean([f.fid in verify(r, t) for r, t in zip(raws, texts)])
        rows.append({"feature": f.fid, "facet": f.facet, "raw_flag_rate": flag,
                     "text_pattern_rate": text, "verified_rate": kept,
                     "flags_removed": flag - kept,
                     "suspect": bool(flag > 0.05 and flag > 2 * text)})
    df = pd.DataFrame(rows)
    if out_csv:
        df.to_csv(out_csv, index=False)
    if verbose:
        bad = df[df.suspect]
        print(f"[gate] wild precision on {len(texts)} model concepts: "
              f"{len(bad)} suspect feature(s) -> {'PASS' if bad.empty else 'WARN'}")
        for _, r in bad.iterrows():
            print(f"        {r.feature}: LLM flags {r.raw_flag_rate:.2f}, text supports "
                  f"{r.text_pattern_rate:.2f}, verified {r.verified_rate:.2f}")
    return df


def validate_s1_examples(extractor: Extractor, verbose=True) -> dict:
    """Study 1 bundle + flaw recoverable from each v6 exemplar."""
    res = []
    for ex in S1_EXAMPLES:
        f = extractor.extract_one(ex + S1_FLAW_SENTENCE, seed=13)
        res.append({m: (m in f) for m in S1_BUNDLE + (S1_FLAW,)})
    ok = all(all(r.values()) for r in res)
    if verbose:
        print(f"[gate] Study 1 exemplars: {'PASS' if ok else 'PARTIAL'} " +
              str([{k: int(v) for k, v in r.items()} for r in res]))
    return {"rows": res, "pass": ok}
