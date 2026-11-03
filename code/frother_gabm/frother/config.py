"""Configuration. Edit defaults here or override on the command line.

Sizes are those used in the paper: study1 (1,920 concepts), P (380), D2 (2,880)
on a laptop with Ollama; profile (~3,000 per model) on ARC with vLLM.
`--quick` shrinks everything to a few minutes for a first end-to-end check.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, asdict


@dataclass
class Config:
    # ---- LLM endpoint -----------------------------------------------------
    backend: str = "ollama"                 # "ollama" (local) or "openai" (vLLM on ARC)
    base_url: str = "http://localhost:11434"
    embed_base_url: str = ""                 # "" -> same as base_url
    api_key: str = ""                        # only needed for vLLM
    model: str = "llama3.1:8b-instruct-q8_0"
    extract_model: str = ""                  # "" -> same as model
    embed_model: str = "qwen3-embedding:4b"
    temperature: float = 0.7
    max_tokens: int = 400
    timeout_s: float = 240.0
    max_retries: int = 4
    workers: int = 4                          # concurrent LLM calls (match OLLAMA_NUM_PARALLEL)
    extract: bool = True                      # False -> store concepts ungraded; grade later with `rescore`
    chat_kwargs: str = ""                     # JSON passed as chat_template_kwargs (vLLM), e.g. '{"enable_thinking": false}'
    embed: bool = True                        # False -> skip embedding-based metrics

    # ---- paths ---------------------------------------------------------------
    results_dir: str = "results"
    figures_dir: str = "figures"
    cache_dir: str = "cache"
    master_seed: int = 20260927

    # ---- job sizes -------------------------------------------------------------
    study1_reps: int = 20                     # study1: runs per (brief, exposure) cell
    study1_iterations: int = 12               # study1: concepts per run

    e_reps: int = 50                          # P: generations per (mechanism, instruction) cell
    e_baseline_reps: int = 80                 # P: unexposed generations

    team_size: int = 8                        # D2 and profile: agents per team
    d_history: int = 2                        # D2 and profile: own previous concepts shown
    d2_reps: int = 15                         # D2: teams per mode
    d2_rounds: int = 8

    # ---- profile protocol (per model) --------------------------------------
    prof_prior_n: int = 200                   # solo unexposed concepts per half
    prof_drift_runs: int = 40                 # memory runs per half
    prof_drift_iters: int = 8
    prof_social_teams: int = 12               # interacting (see-all) teams per half
    prof_nominal_teams: int = 6               # revise-alone control teams per half
    prof_rounds: int = 8
    prof_scan_reps: int = 30                  # concepts per exemplar
    prof_scan_rarest: int = 10                # model's own rarest features to scan
    prof_halves: int = 2                      # split-half replicates

    n_boot: int = 3000

    def resolve(self):
        if not self.extract_model:
            self.extract_model = self.model
        if not self.embed_base_url:
            self.embed_base_url = self.base_url
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        return self

    def quick(self):
        """Tiny sizes for a smoke run."""
        self.study1_reps, self.study1_iterations = 2, 3
        self.e_reps, self.e_baseline_reps = 3, 4
        self.d2_reps, self.d2_rounds = 2, 3
        self.prof_prior_n, self.prof_drift_runs, self.prof_drift_iters = 6, 2, 3
        self.prof_social_teams, self.prof_nominal_teams, self.prof_rounds = 2, 1, 3
        self.prof_scan_reps, self.prof_scan_rarest, self.team_size = 2, 3, 4
        self.n_boot = 200
        return self

    def dump(self, path: str):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, default=list)


def add_config_args(p: argparse.ArgumentParser):
    p.add_argument("--quick", action="store_true", help="tiny sizes, minutes not hours")
    p.add_argument("--backend", choices=["ollama", "openai"])
    p.add_argument("--base-url")
    p.add_argument("--embed-base-url")
    p.add_argument("--api-key")
    p.add_argument("--model")
    p.add_argument("--extract-model")
    p.add_argument("--embed-model")
    p.add_argument("--workers", type=int)
    p.add_argument("--results-dir")
    p.add_argument("--figures-dir")
    p.add_argument("--cache-dir")
    p.add_argument("--temperature", type=float)
    p.add_argument("--max-tokens", type=int, dest="max_tokens")
    p.add_argument("--no-extract", action="store_true", help="store concepts ungraded (grade later with rescore)")
    p.add_argument("--no-embed", action="store_true", help="skip embedding-based metrics")
    p.add_argument("--chat-kwargs", help="JSON for vLLM chat_template_kwargs, e.g. '{\"enable_thinking\": false}'")
    for k in ("study1_reps", "study1_iterations", "e_reps", "e_baseline_reps",
              "team_size", "d_history", "d2_reps", "d2_rounds",
              "prof_prior_n", "prof_drift_runs", "prof_drift_iters", "prof_social_teams",
              "prof_nominal_teams", "prof_rounds", "prof_scan_reps", "prof_scan_rarest", "prof_halves"):
        p.add_argument("--" + k.replace("_", "-"), type=int, dest=k)


def config_from_args(a: argparse.Namespace) -> Config:
    cfg = Config()
    if getattr(a, "quick", False):
        cfg.quick()
    if getattr(a, "no_extract", False):
        cfg.extract = False
    if getattr(a, "no_embed", False):
        cfg.embed = False
    for k, v in vars(a).items():
        if v is None or k in ("quick", "jobs", "command", "no_extract", "no_embed"):
            continue
        key = k.replace("-", "_")
        if hasattr(cfg, key):
            setattr(cfg, key, v)
    if cfg.api_key == "" and os.environ.get("LLM_API_KEY"):
        cfg.api_key = os.environ["LLM_API_KEY"]
    return cfg.resolve()
