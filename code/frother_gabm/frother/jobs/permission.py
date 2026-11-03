"""P — permission test: is the avoidance of a shown mechanism a reading of the
instruction ("propose a concept" = "propose a different one") or a disposition?

Solo, no memory, one exemplar framed as an earlier designer's work.
exemplar {common whisk, rare venturi}, both with the rare heater
 x instruction {neutral, permissive, directive}
 + unexposed baseline.

  neutral    : "Propose your concept now."                      (the default instruction)
  permissive : "... You may reuse, adapt or build on any part of the example if you
                judge it good; similarity to the example is acceptable."
  directive  : "Propose a concept that builds directly on the example: keep what
                works in it and improve the rest."

H7a  permission raises retention of the shown common mechanism (whisk) above the
     neutral cell; if it stays near zero, avoidance is dispositional.
H7b  permission raises adoption of the shown rare mechanism (venturi) above zero.
H7c  directive >= permissive >= neutral on both.
"""
from __future__ import annotations

from ..schema import EXAMPLE_HEADERS, stimulus
from ..session import seed_of
from ..store import JobStore

# The exemplar is the base concept ("Meadow Froth") plus an integrated heater
# (rare, peripheral) and differs only in its MECHANISM sentence:
#     common = rotating whisk (the model's own default)
#     rare   = venturi air-injection
MECHS = {"common": None, "rare": "aerator_nozzle"}
PERIPHERAL = "integrated_heating"
# every feature the exemplars carry, for per-feature tracking
TRACKED = ("whisk_coil", "aerator_nozzle", "battery_motor", "stainless_steel",
           "integrated_heating", "one_touch_auto")


def exemplar(mech: str) -> str:
    attrs = [PERIPHERAL] + ([MECHS[mech]] if MECHS[mech] else [])
    return stimulus(0, attrs)

JOB = "P"
ASKS = {
    "neutral": "Propose your concept now.",
    "permissive": ("Propose your concept now. You may reuse, adapt or build on any part of the "
                   "example if you judge it good; similarity to the example is acceptable."),
    "directive": ("Propose a concept that builds directly on the example: keep what works in it "
                  "and improve the rest."),
}


def _unit(sess, store, key, mech, instr, rep):
    blocks = [] if mech == "none" else [EXAMPLE_HEADERS["prior"] + "\n" + exemplar(mech)]
    ask = ASKS["neutral"] if mech == "none" else ASKS[instr]
    c = sess.generate(brief="neutral", blocks=blocks, ask=ask,
                      seed=seed_of(sess.cfg.master_seed, JOB, mech, instr, rep))
    row = {"job": JOB, "mechanism_shown": mech, "instruction": instr, "rep": rep,
           **{f"has_{f}": int(f in c["features"]) for f in TRACKED},
           "has_impeller_turbine": int("impeller_turbine" in c["features"]),
           "n_features": len(c["features"]), "features": c["features"],
           "title": c["title"], "description": c["description"], "rationale": c["rationale"]}
    store.commit(key, row, [dict(row, unit=key)])


def run(sess, cfg):
    store = JobStore(cfg.results_dir, JOB)
    keys, spec = [], {}
    for rep in range(cfg.e_baseline_reps):
        k = f"none|none|{rep}"; keys.append(k); spec[k] = ("none", "none", rep)
    for mech in ("common", "rare"):
        for instr in ASKS:
            for rep in range(cfg.e_reps):
                k = f"{mech}|{instr}|{rep}"; keys.append(k); spec[k] = (mech, instr, rep)
    store.run_units(keys, lambda k: _unit(sess, store, k, *spec[k]), cfg.workers)
