"""Incremental, resumable output.

Each job owns a directory results/<job>/ with:
  units.csv     one row per completed unit (run / chain / team), appended live
  concepts.csv  one row per generated concept, appended live
  done.txt      unit keys already completed (resume index)
  progress.log  human-readable progress, tail -f friendly
Rows are written under a lock as soon as a unit finishes, never batched.
"""
from __future__ import annotations

import csv
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


class JobStore:
    def __init__(self, results_dir: str, job: str):
        self.dir = os.path.join(results_dir, job)
        os.makedirs(self.dir, exist_ok=True)
        self.job = job
        self.lock = threading.Lock()
        self._done_path = os.path.join(self.dir, "done.txt")
        self.done = set()
        if os.path.exists(self._done_path):
            with open(self._done_path) as f:
                self.done = {ln.strip() for ln in f if ln.strip()}
        self._files = {}
        self.t0 = time.time()

    def path(self, name: str) -> str:
        return os.path.join(self.dir, name)

    def is_done(self, key: str) -> bool:
        return key in self.done

    def _append(self, name: str, rows: list[dict]):
        if not rows:
            return
        p = self.path(name)
        new = not os.path.exists(p) or os.path.getsize(p) == 0
        cols = list(rows[0].keys())
        if not new:
            with open(p, newline="") as f:
                cols = next(csv.reader(f))
            extra = [c for c in rows[0] if c not in cols]
            if extra:                        # widen the header if a later row has new columns
                import pandas as pd
                df = pd.read_csv(p)
                for c in extra:
                    df[c] = None
                df.to_csv(p, index=False)
                cols = list(df.columns)
        with open(p, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            if new:
                w.writeheader()
            for r in rows:
                w.writerow({c: _cell(r.get(c)) for c in cols})

    def commit(self, key: str, unit_row: dict | None, concept_rows: list[dict] | None = None,
               extra: dict[str, list[dict]] | None = None):
        """Write everything for one unit atomically-enough (single lock), then mark done."""
        with self.lock:
            if unit_row:
                self._append("units.csv", [unit_row])
            if concept_rows:
                self._append("concepts.csv", concept_rows)
            for name, rows in (extra or {}).items():
                self._append(name, rows)
            with open(self._done_path, "a") as f:
                f.write(key + "\n")
            self.done.add(key)

    def log(self, msg: str):
        line = f"[{time.strftime('%H:%M:%S')}] [{self.job}] {msg}"
        print(line, flush=True)
        with self.lock:
            with open(self.path("progress.log"), "a") as f:
                f.write(line + "\n")

    def run_units(self, keys: list[str], fn, workers: int):
        """Run fn(key) for every key not already done, in a thread pool, logging
        progress. fn must call store.commit itself."""
        todo = [k for k in keys if not self.is_done(k)]
        self.log(f"{len(keys)} units total, {len(keys) - len(todo)} already done, "
                 f"{len(todo)} to run with {workers} workers")
        if not todo:
            return
        n_done, t0 = 0, time.time()
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            futs = {ex.submit(fn, k): k for k in todo}
            for fut in as_completed(futs):
                k = futs[fut]
                try:
                    fut.result()
                except Exception as e:      # noqa: BLE001
                    self.log(f"ERROR in unit {k}: {e!r}")
                    continue
                n_done += 1
                el = time.time() - t0
                eta = el / n_done * (len(todo) - n_done)
                self.log(f"{n_done}/{len(todo)} done  elapsed {el/60:.1f} min  "
                         f"ETA {eta/60:.1f} min  ({k})")
        self.log("finished")


def _cell(v):
    if isinstance(v, (list, tuple, set, frozenset, dict)):
        return json.dumps(sorted(v) if isinstance(v, (set, frozenset)) else v)
    return v


# ==========================================================================
# concept parsing
# ==========================================================================

_TITLE = re.compile(r"Concept Title:\s*(.*?)(?=\n\s*Concept Description:|\Z)", re.S | re.I)
_DESC = re.compile(r"Concept Description:\s*(.*?)(?=\n\s*Rationale:|\Z)", re.S | re.I)
_RAT = re.compile(r"Rationale:\s*(.*)", re.S | re.I)


_THINK = re.compile(r"<think>.*?</think>\s*", re.S | re.I)


def parse_concept(raw: str) -> dict:
    t = _THINK.sub("", raw or "").strip()
    tm, dm, rm = _TITLE.search(t), _DESC.search(t), _RAT.search(t)
    title = tm.group(1).strip() if tm else ""
    desc = dm.group(1).strip() if dm else ""
    rat = rm.group(1).strip() if rm else ""
    if not desc:
        desc = _RAT.sub("", t).strip()
    return {"title": title[:200], "description": desc, "rationale": rat[:1000]}


def concept_text(title: str, description: str) -> str:
    """What other agents see, and what the extractor scores: no rationale."""
    return f"Concept Title: {title}\nConcept Description: {description}"
