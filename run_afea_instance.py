#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from FLPOPT.flopt import FLPOPT
from instancia.parse import load_config


DEFAULT_CONFIG_PATH = Path("configs/config_afea_n11.json")
DEFAULT_OUTPUT_PATH = Path("evolucao_teorica/afea_n11.csv")


def rescale_frequency_bounds(cfg: dict) -> dict:
    f_min = np.array(cfg["f_min"], dtype=float)
    f_max = np.array(cfg["f_max"], dtype=float)
    c = np.array(cfg["c"], dtype=float)
    S = np.array(cfg["S"], dtype=float)
    scale = max(1.0, np.max(c * S) / max(1.0, np.max(f_max)))
    cfg["f_min"] = (f_min * scale).tolist()
    cfg["f_max"] = (f_max * scale).tolist()
    return cfg


def build_config(path: Path) -> dict:
    config = {
        "N": 11,
        "alpha": [
            5.128e-09, 5.060e-09, 4.912e-09, 5.080e-09, 5.185e-09, 5.133e-09,
            5.119e-09, 5.095e-09, 5.056e-09, 4.879e-09, 4.795e-09,
        ],
        "c": [
            25.6363, 25.5332, 39.2031, 30.2821, 27.3132, 31.4829, 29.074,
            30.326, 27.9354, 36.0697, 36.3212,
        ],
        "S": [
            747088320, 684321792, 748350720, 1128383616, 1047438528, 436285440,
            1135301568, 995528640, 1026684672, 352361088, 1102428672,
        ],
        "f_min": [1.4, 1.5, 2.6, 1.5, 1.4, 1.4, 1.5, 1.4, 1.6, 2.6, 2.6],
        "f_max": [2.0, 1.8, 3.9, 3.6, 2.3, 3.7, 2.4, 2.7, 2.0, 2.9, 3.6],
        "epsilon_0": 0.999,
        "theta_prev": [0.1] * 11,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)
    return config


def save_results(res, cfg: dict, output_path: Path) -> pd.DataFrame:
    rows = []
    for i in range(len(res.X)):
        x = res.X[i]
        f = res.F[i]
        row = {
            "obj_0": float(f[0]),
            "obj_1": float(f[1]),
            "obj_2": float(f[2]),
            "T": float(x["T"]),
        }
        for n in range(cfg["N"]):
            row[f"f_{n}"] = float(x[f"f_{n}"])
            row[f"beta_{n}"] = int(x[f"beta_{n}"])
            row[f"psi_{n}"] = int(x[f"psi_{n}"])
            row[f"theta_{n}"] = float(x[f"theta_{n}"])
        rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Executa a otimização AFEA para a instância fornecida")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Caminho do JSON de configuração")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Caminho do CSV com as soluções")
    parser.add_argument("--output-plot", type=Path, default=Path("Figuras/afea_n11_front.png"), help="Caminho da imagem da frente de Pareto")
    parser.add_argument("--n-gen", type=int, default=120, help="Número de gerações do algoritmo")
    parser.add_argument("--pop-size", type=int, default=100, help="Tamanho da população")
    parser.add_argument("--seed", type=int, default=1, help="Semente da execução")
    args = parser.parse_args()

    build_config(args.config)
    cfg = load_config(str(args.config))
    cfg = rescale_frequency_bounds(cfg)

    print(f"Executando otimização para N={cfg['N']} ...")
    instancia = FLPOPT(
        cfg["N"],
        cfg["alpha"],
        cfg["c"],
        cfg["S"],
        cfg["f_min"],
        cfg["f_max"],
        cfg["epsilon_0"],
        cfg["theta_prev"],
    )

    res = instancia.solve(n_gen=args.n_gen, pop_size=args.pop_size, seed=args.seed)

    if res.F is None:
        print("Nenhuma solução viável foi encontrada.")
        return

    df = save_results(res, cfg, args.output)
    print(f"{len(df)} soluções foram salvas em {args.output}")

    args.output_plot.parent.mkdir(parents=True, exist_ok=True)
    instancia.scatterplot(file_name=str(args.output_plot))
    print(f"Frente de Pareto salva em {args.output_plot}")

    pesos = [0.1, 0.9, 0.1]
    idx = instancia.mcdm_pseudo_weights(pesos)
    if idx is not None:
        sol = res.X[idx]
        print("Solução selecionada pelo critério pseudo-weight:")
        print({
            "T": float(sol["T"]),
            "obj_0": float(res.F[idx][0]),
            "obj_1": float(res.F[idx][1]),
            "obj_2": float(res.F[idx][2]),
        })


if __name__ == "__main__":
    main()
