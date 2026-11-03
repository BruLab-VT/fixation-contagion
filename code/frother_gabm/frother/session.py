"""Shared runtime for all jobs."""
from __future__ import annotations

import hashlib
import os

from .extract import Extractor, pattern_features
from .llm import make_client, check_endpoint
from .schema import SCHEMA, BRIEFS, OUTPUT_FORMAT, SYSTEM_PROMPTS
from .store import parse_concept, concept_text


def seed_of(*parts) -> int:
    h = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(h[:8], "big") & ((1 << 31) - 1)


class Session:
    def __init__(self, cfg):
        self.cfg = cfg
        self.client = make_client(cfg, SCHEMA)
        self.extractor = Extractor(self.client, cfg)
        cfg.dump(os.path.join(cfg.results_dir, "config_last_run.json"))

    def check(self) -> bool:
        msg = check_endpoint(self.cfg)
        print("[endpoint]", msg)
        return not msg.startswith("ERROR")

    # ---- generation -----------------------------------------------------------
    def generate(self, *, brief: str, blocks: list[str], seed: int,
                 persona: str | None = None, setting: str = "working alone on an ideation session",
                 ask: str = "Propose your concept now.", extract_seed: int | None = None) -> dict:
        """One concept. blocks are placed between the format and the ask, in
        order, so the LAST block is what the model reads just before generating."""
        system = SYSTEM_PROMPTS[brief].format(setting=setting)
        if persona:
            system = system.replace("a product design engineer", persona) \
                           .replace("an inventive generalist who enjoys unusual mechanisms", persona)
        parts = [f"DESIGN BRIEF\n{BRIEFS[brief]}", OUTPUT_FORMAT] + [b for b in blocks if b] + [ask]
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": "\n\n".join(parts)}]
        raw = self.client.chat(msgs, seed)
        c = parse_concept(raw)
        c["text"] = concept_text(c["title"], c["description"])
        c["features_pattern"] = pattern_features(c["text"])
        c["features"] = self.extractor.extract_one(c["text"], seed=extract_seed or 7) if self.cfg.extract else set()
        c["raw"] = raw
        c["prompt_chars"] = len(msgs[1]["content"])
        return c

    def embed(self, texts):
        if not self.cfg.embed:
            import numpy as np
            return np.zeros((len(texts), 2))
        return self.client.embed(texts)
