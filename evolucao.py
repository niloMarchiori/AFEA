import json
from pathlib import Path

import numpy as np
import pandas as pd
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from instancia.parse import load_config


CONFIG_PATH = Path("configs/config_afea_n11.json")
OUTPUT_DIR = Path("evolucao_teorica")
LOG_PATH = OUTPUT_DIR / "round_archive_log.csv"
RESULTS_PATH = OUTPUT_DIR / "evolucao_teorica.csv"
ROUND_INPUTS_PATH = OUTPUT_DIR / "round_inputs.csv"

MAX_ROUNDS = 5
SOLUTIONS_PER_ROUND = 3


def keep_nondominated_solutions(df, obj_cols):
    if df.empty:
        return df

    F = df[obj_cols].to_numpy()
    nd_indices = NonDominatedSorting().do(F, only_non_dominated_front=True)
    return df.iloc[nd_indices].reset_index(drop=True)


def serialize_array(value):
    if isinstance(value, np.ndarray):
        return json.dumps(value.tolist())
    return json.dumps(value)


def parse_array_field(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, list):
        return np.array(value)
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            decoded = value
        if isinstance(decoded, list):
            return np.array(decoded)
        return decoded
    return np.array(value)


def simulate_round_archive(cfg, state, round_idx, columns):
    progress = (round_idx + 1) / max(1, MAX_ROUNDS)
    f_min = np.array(cfg["f_min"], dtype=float)
    f_max = np.array(cfg["f_max"], dtype=float)
    c = np.array(cfg["c"], dtype=float)
    S = np.array(cfg["S"], dtype=float)
    base_freq = f_min + (f_max - f_min) * (0.2 + 0.5 * progress)

    rows = []
    for solution_idx in range(SOLUTIONS_PER_ROUND):
        freq = base_freq + 0.01 * solution_idx
        beta = np.array([1.0 if (state["beta_h"][n] + progress) >= 0.45 else 0.0 for n in range(cfg["N"])])
        theta = np.maximum(0.0, state["theta_prev"] - 0.02 * (round_idx + 1))
        psi = beta.astype(int)

        row = {
            "obj_0": float(np.sum((c / 1000.0) * (1.0 - beta))),
            "obj_1": float(np.sum((S / np.max(S)) * (1.0 - beta)) / cfg["N"]),
            "obj_2": float(np.sum(np.maximum(0.0, freq - f_min)) / cfg["N"] + 0.05 * progress),
            "T": float(0.4 + 0.1 * round_idx + 0.02 * solution_idx),
        }
        for n in range(cfg["N"]):
            row[f"f_{n}"] = float(freq[n])
            row[f"beta_{n}"] = float(beta[n])
            row[f"theta_{n}"] = float(theta[n])
            row[f"psi_{n}"] = int(psi[n])
        rows.append(row)

    round_df = pd.DataFrame(rows, columns=columns)
    round_df["round"] = round_idx
    return round_df


def simple_round_advance(state, round_idx):
    progress = (round_idx + 1) / max(1, MAX_ROUNDS)
    state["beta_h"] = np.clip(state["beta_h"] + 0.08 + 0.01 * progress, 0.0, 1.0)
    state["theta_prev"] = np.maximum(0.0, state["theta_prev"] - 0.01 * (round_idx + 1))


if __name__ == "__main__":
    cfg = load_config(str(CONFIG_PATH))
    N = cfg["N"]
    objective_cols = [f"obj_{j}" for j in range(3)]
    columns = objective_cols + ["round", "T"]
    columns += [f"f_{i}" for i in range(N)]
    columns += [f"beta_{i}" for i in range(N)]
    columns += [f"theta_{i}" for i in range(N)]
    columns += [f"psi_{i}" for i in range(N)]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if LOG_PATH.exists():
        archive = pd.read_csv(LOG_PATH)
        missing = [col for col in columns if col not in archive.columns]
        if missing:
            raise ValueError(f"O histórico encontrado está faltando colunas: {missing}")
        archive = archive[columns]
    else:
        archive = pd.DataFrame(columns=columns)

    saved_round_inputs = []
    state = {
        "beta_h": np.zeros(N, dtype=float),
        "theta_prev": np.array(cfg["theta_prev"], dtype=float),
    }

    for t in range(MAX_ROUNDS):
        print(f"\n-------- RODADA {t} --------")
        round_meta = {
            "round": t,
            "n_runs_per_round": 1,
            "n_gen": 100,
            "pop_size":150,
            "alpha": serialize_array(cfg["alpha"]),
            "c": serialize_array(cfg["c"]),
            "S": serialize_array(cfg["S"]),
            "f_min": serialize_array(cfg["f_min"]),
            "f_max": serialize_array(cfg["f_max"]),
            "epsilon_0": cfg["epsilon_0"],
            "beta_h": serialize_array(state["beta_h"]),
            "theta_prev": serialize_array(state["theta_prev"]),
        }
        saved_round_inputs.append(round_meta)

        round_archive = simulate_round_archive(cfg, state, t, columns)
        round_archive = round_archive.drop_duplicates().reset_index(drop=True)
        round_archive = keep_nondominated_solutions(round_archive, objective_cols)
        print(f"  Frente de Pareto da rodada {t} gerada com {len(round_archive)} soluções não dominadas.")

        archive = pd.concat([archive, round_archive], ignore_index=True)
        archive = archive.drop_duplicates().reset_index(drop=True)
        archive.to_csv(LOG_PATH, index=False)
        print(f"Log compartilhado atualizado em {LOG_PATH} ({len(archive)} soluções acumuladas).")

        simple_round_advance(state, t)

    archive.to_csv(RESULTS_PATH, index=False)
    pd.DataFrame(saved_round_inputs).to_csv(ROUND_INPUTS_PATH, index=False)
