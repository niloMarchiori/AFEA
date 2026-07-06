
import json
import numpy as np

def load_config(path):
    with open(path, 'r') as f:
        cfg = json.load(f)

    # helpers to coerce to numpy arrays
    def arr(key, default=None):
        v = cfg.get(key, default)
        if v is None:
            return None
        return np.array(v)

    N = int(cfg.get('N'))
    alpha = arr('alpha')
    c = arr('c')
    S = arr('S')
    f_min = arr('f_min')
    f_max = arr('f_max')
    epsilon_0 = float(cfg.get('epsilon_0', 0.98))
    theta_prev = arr('theta_prev', np.ones(N) * 0.1)

    return dict(N=N, alpha=alpha, c=c, S=S, f_min=f_min, f_max=f_max,
                epsilon_0=epsilon_0, theta_prev=theta_prev)