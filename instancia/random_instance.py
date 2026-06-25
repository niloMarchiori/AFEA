import json
from typing import Dict, Tuple, Optional

import numpy as np


def _validate_interval(interval: Tuple[float, float], name: str) -> Tuple[float, float]:
    if len(interval) != 2:
        raise ValueError(f"Intervalo '{name}' deve ter dois valores: (min, max)")
    low, high = float(interval[0]), float(interval[1])
    if low > high:
        raise ValueError(f"Intervalo '{name}' inválido: min deve ser menor ou igual a max")
    return low, high


def random_instance_config(
                            N: int,
                            alpha_range: Tuple[float, float] = (2e-18, 2e-18),
                            
                            c_range: Tuple[float, float] = (5.0, 30.0),
                            S_range: Tuple[float, float] = (1e7, 1e9),
                            f_min_range: Tuple[float, float] = (0.8e9, 1.3e9),
                            f_max_range: Tuple[float, float] = (1.5e9, 3.8e9),

                            epsilon_range: Tuple[float, float] = (0.98, 0.99),
                            theta_prev_range: Tuple[float, float] = (0.01, 0.5),
                            seed: Optional[int] = None,
                        ) -> Dict[str, object]:
    """Gera uma configuração aleatória para FLPOPT.

    Retorna um dicionário compatível com load_config().
    """
    rng = np.random.default_rng(seed)

    alpha_low, alpha_high = _validate_interval(alpha_range, 'alpha_range')
    c_low, c_high = _validate_interval(c_range, 'c_range')
    S_low, S_high = _validate_interval(S_range, 'S_range')
    f_min_low, f_min_high = _validate_interval(f_min_range, 'f_min_range')
    f_max_low, f_max_high = _validate_interval(f_max_range, 'f_max_range')
    epsilon_low, epsilon_high = _validate_interval(epsilon_range, 'epsilon_range')
    theta_low, theta_high = _validate_interval(theta_prev_range, 'theta_prev_range')

    if f_max_high <= f_min_low:
        raise ValueError('O intervalo de f_max deve ser maior que o de f_min para permitir valores válidos.')

    alpha = rng.uniform(alpha_low, alpha_high, size=N).tolist()
    c = rng.uniform(c_low, c_high, size=N).round().astype(int).tolist()
    S = rng.uniform(S_low, S_high, size=N).tolist()
    f_min = rng.uniform(f_min_low, f_min_high, size=N)
    f_max = rng.uniform(f_max_low, f_max_high, size=N)

    f_min, f_max = np.minimum(f_min, f_max), np.maximum(f_min, f_max)
    f_min = f_min.tolist()
    f_max = f_max.tolist()

    epsilon_0 = float(rng.uniform(epsilon_low, epsilon_high))
    theta_prev = rng.uniform(theta_low, theta_high, size=N).tolist()

    return {
        'N': N,
        'alpha': alpha,
        'c': c,
        'S': S,
        'f_min': f_min,
        'f_max': f_max,
        'epsilon_0': epsilon_0,
        'theta_prev': theta_prev
    }


def save_config(config: Dict[str, object], path: str = 'config_random.json') -> None:
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)


if __name__ == '__main__':
    import argparse
    import random

    parser = argparse.ArgumentParser(description='Gera um arquivo de configuração aleatória para FLPOPT')
    parser.add_argument('--N', type=int, required=True, help='Número de clientes')
    parser.add_argument('--seed', type=int, default=None, help='Semente para reprodução')
    parser.add_argument('--output', '-o', default='config_random.json', help='Arquivo JSON de saída')
    parser.add_argument('--alpha', nargs=2, type=float, default=(2e-18, 2e-18), help='Intervalo para alpha')
    parser.add_argument('--c', nargs=2, type=float, default=(5.0, 30.0), help='Intervalo para c')
    parser.add_argument('--S', nargs=2, type=float, default=(1e6, 1e8), help='Intervalo para S')
    parser.add_argument('--f_min', nargs=2, type=float, default=(0.9, 1.3), help='Intervalo para f_min')
    parser.add_argument('--f_max', nargs=2, type=float, default=(1.5, 3.9), help='Intervalo para f_max')
    parser.add_argument('--epsilon', nargs=2, type=float, default=(0.98, 0.99), help='Intervalo para epsilon_0')
    parser.add_argument('--theta_prev', nargs=2, type=float, default=(0.01, 0.5), help='Intervalo para theta_prev')

    args = parser.parse_args()

    cfg = random_instance_config(
        N=args.N,
        alpha_range=tuple(args.alpha),
        c_range=tuple(args.c),
        S_range=tuple(args.S),
        f_min_range=tuple(args.f_min),
        f_max_range=tuple(args.f_max),
        epsilon_range=tuple(args.epsilon),
        theta_prev_range=tuple(args.theta_prev),
        seed=args.seed,
    )
    save_config(cfg, args.output)
    print(f'Configuração aleatória salva em {args.output}')
