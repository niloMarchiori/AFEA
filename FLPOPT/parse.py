import json
from pathlib import Path

import numpy as np


def load_config(path):
    path = Path(path)
    if not path.is_absolute():
        path = (Path(__file__).resolve().parents[1] / path).resolve()
    with open(path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    with open(path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)

    def arr(key, default=None):
        value = cfg.get(key, default)
        if value is None:
            return None
        return np.array(value)

    N = int(cfg.get("N"))
    alpha = arr("alpha")
    c = arr("c")
    S = arr("S")
    f_min = arr("f_min")
    f_max = arr("f_max")
    epsilon_0 = float(cfg.get("epsilon_0", 0.98))
    theta_prev = arr("theta_prev", np.ones(N) * 0.1)

    return dict(
        N=N,
        alpha=alpha,
        c=c,
        S=S,
        f_min=f_min,
        f_max=f_max,
        epsilon_0=epsilon_0,
        theta_prev=theta_prev,
    )
