"""One team session (revise instruction), reusable by D2 and the profile
protocol. Also holds the team prompts. Returns concept rows and exposure records; the caller commits."""
from __future__ import annotations

import numpy as np

from .schema import EXAMPLE_HEADERS, PERSONAS
from .session import seed_of

# Instruction and memory headers for team sessions (moved here from the
# retired job D so that D2 and profile share one definition).
ASK = {"propose_next": "Propose your next concept now.",
       "revise": ("Revise your current concept or replace it. You may keep it "
                  "unchanged, refine it, or propose a different one - whichever "
                  "you judge best for the brief. Respond in the same format.")}
MEM_HEAD = {"propose_next": "Your previous concepts in this session:",
            "revise": "Your current concept (most recent last):"}


def run_team(sess, *, n: int, rounds: int, dose: int, seed_parts: tuple, instruction: str = "revise",
             tag: dict | None = None) -> tuple[list[dict], list[dict]]:
    cfg = sess.cfg
    rng = np.random.default_rng(seed_of(cfg.master_seed, *seed_parts))
    personas = list(PERSONAS[:n]) + [f"{PERSONAS[i % len(PERSONAS)]} (2)" for i in range(len(PERSONAS), n)]
    rng.shuffle(personas)
    hist = {a: [] for a in range(n)}
    crows, exposures = [], []
    for t in range(rounds):
        latest = {a: hist[a][-1] for a in range(n) if hist[a]}
        for a in range(n):
            others = [b for b in latest if b != a]
            rng.shuffle(others)
            vis = others[:dose]
            own = hist[a][-cfg.d_history:] if cfg.d_history > 0 else hist[a]
            mem = ""
            if own:
                mem = MEM_HEAD[instruction] + "\n" + "\n\n".join(f"{i+1}. {h['text']}" for i, h in enumerate(own))
            obs = ""
            if vis:
                obs = EXAMPLE_HEADERS["teammates"] + "\n" + "\n\n".join(latest[b]["text"] for b in vis)
            ask = "Propose your concept now." if not hist[a] else ASK[instruction]
            c = sess.generate(brief="neutral", blocks=[mem, obs], persona=personas[a],
                              setting=("participating in a team design session" if dose else
                                       "working alone on a design session"),
                              seed=seed_of(cfg.master_seed, *seed_parts, t, a), ask=ask)
            hist[a].append(c)
            for b in vis:
                exposures.append({**(tag or {}), "round": t + 1, "viewer": a, "source": b})
            crows.append({**(tag or {}), "round": t + 1, "agent": a, "visible": sorted(vis),
                          "features": c["features"], "features_pattern": c["features_pattern"],
                          "title": c["title"], "description": c["description"], "rationale": c["rationale"]})
    return crows, exposures
