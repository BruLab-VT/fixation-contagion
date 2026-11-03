"""frother — a set of generative-agent experiments for design fixation.

Jobs (independently runnable and resumable):
  study1  solo sessions: brief x exposure {none, single, triple, memory}   (paper Study 1)
  P       one example x instruction {neutral, permissive, directive}       (paper Study 1)
  D2      eight-agent teams, 0 / 3 / 7 visible teammates                   (paper Study 2)
  profile unexposed, own-history, teams, exposure scan, per model          (paper Study 3)

Every job appends one CSV row per completed unit as it goes, and skips units
already recorded, so a job can be stopped and restarted at any time.
"""
__version__ = "0.7.0"
