import argparse
import json
import os
import numpy as np
import pandas as pd
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from FLPOPT.flopt import FLPOPT
from instancia.parse import load_config


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


def build_solution_rows(X, F, N):
    rows = []
    for i in range(len(F)):
        row = {f'obj_{j}': float(F[i, j]) for j in range(F.shape[1])}
        row['T'] = X[i]['T']
        for n in range(N):
            row[f'f_{n}'] = X[i][f'f_{n}']
            row[f'beta_{n}'] = X[i][f'beta_{n}']
            row[f'theta_{n}'] = X[i][f'theta_{n}']
            row[f'psi_{n}'] = X[i][f'psi_{n}']
        rows.append(row)
    return rows


def build_round_instance(base_cfg, row_inputs):
    alpha = parse_array_field(row_inputs.get('alpha', base_cfg['alpha']))
    c = parse_array_field(row_inputs.get('c', base_cfg['c']))
    S = parse_array_field(row_inputs.get('S', base_cfg['S']))
    f_min = parse_array_field(row_inputs.get('f_min', base_cfg['f_min']))
    f_max = parse_array_field(row_inputs.get('f_max', base_cfg['f_max']))
    theta_prev = parse_array_field(row_inputs.get('theta_prev', base_cfg['theta_prev']))
    epsilon_0 = row_inputs.get('epsilon_0', base_cfg['epsilon_0'])
    beta_h = parse_array_field(row_inputs.get('beta_h', np.zeros(base_cfg['N'])))

    instance = FLPOPT(base_cfg['N'], alpha, c, S, f_min, f_max, epsilon_0, theta_prev)
    if beta_h is not None:
        instance.beta_h = beta_h
    return instance


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evolução iterativa lendo config JSON')
    parser.add_argument('--config', '-c', default='config.json', help='Caminho para o arquivo de configuração (JSON)')
    parser.add_argument('--round-inputs', '-r', default=None, help='CSV com inputs específicos por rodada')
    parser.add_argument('--history', '-H', default=None, help='CSV com histórico acumulado de soluções para carregar')
    args = parser.parse_args()

    cfg = load_config(args.config)
    N = cfg['N']
    objective_cols = [f'obj_{j}' for j in range(3)]
    columns = objective_cols + ['round', 'T']
    columns += [f'f_{i}' for i in range(N)]
    columns += [f'beta_{i}' for i in range(N)]
    columns += [f'theta_{i}' for i in range(N)]
    columns += [f'psi_{i}' for i in range(N)]

    if args.history is not None:
        if os.path.exists(args.history):
            archive = pd.read_csv(args.history)
            missing = [col for col in columns if col not in archive.columns]
            if missing:
                raise ValueError(f'O histórico fornecido está faltando colunas: {missing}')
            archive = archive[columns]
        else:
            raise FileNotFoundError(f'Histórico não encontrado: {args.history}')
    else:
        archive = pd.DataFrame(columns=columns)

    if args.round_inputs is not None:
        if os.path.exists(args.round_inputs):
            round_inputs_df = pd.read_csv(args.round_inputs)
        else:
            raise FileNotFoundError(f'CSV de round inputs não encontrado: {args.round_inputs}')
    else:
        round_inputs_df = None

    os.makedirs('evolucao_teorica', exist_ok=True)
    log_path = os.path.join('evolucao_teorica', 'round_archive_log.csv')
    if os.path.exists(log_path):
        os.remove(log_path)
    saved_round_inputs = []

    t = 0
    while t < 40:
        print(f"\n -------- RODADA {t} --------")
        row_inputs = {}
        if round_inputs_df is not None:
            if t >= len(round_inputs_df):
                raise IndexError(f'Não há linha de round-inputs para a rodada {t}.')
            row_inputs = round_inputs_df.iloc[t].to_dict()

        instancia = build_round_instance(cfg, row_inputs)

        n_runs_per_round = 5
        round_seeds = [t * n_runs_per_round + r + 1 for r in range(n_runs_per_round)]
        round_meta = {
            'round': t,
            'n_runs_per_round': n_runs_per_round,
            'n_gen': 200,
            'pop_size': 150,
            'seeds': json.dumps(round_seeds),
            'alpha': serialize_array(parse_array_field(row_inputs.get('alpha', cfg['alpha']))),
            'c': serialize_array(parse_array_field(row_inputs.get('c', cfg['c']))),
            'S': serialize_array(parse_array_field(row_inputs.get('S', cfg['S']))),
            'f_min': serialize_array(parse_array_field(row_inputs.get('f_min', cfg['f_min']))),
            'f_max': serialize_array(parse_array_field(row_inputs.get('f_max', cfg['f_max']))),
            'epsilon_0': row_inputs.get('epsilon_0', cfg['epsilon_0']),
            'beta_h': serialize_array(parse_array_field(row_inputs.get('beta_h', np.zeros(N)))),
            'theta_prev': serialize_array(parse_array_field(row_inputs.get('theta_prev', cfg['theta_prev'])))
        }
        for key in ['alpha', 'c', 'S', 'f_min', 'f_max', 'beta_h', 'theta_prev', 'epsilon_0']:
            if key in row_inputs and pd.notna(row_inputs[key]):
                if key == 'epsilon_0':
                    round_meta[key] = row_inputs[key]
                else:
                    round_meta[key] = serialize_array(parse_array_field(row_inputs[key]))

        saved_round_inputs.append(round_meta)

        if args.history is not None:
            round_archive = archive[archive['round'] == t].copy()
        else:
            round_archive = pd.DataFrame(columns=columns)

        for run_idx, seed in enumerate(round_seeds):
            print(f'  Execução {run_idx + 1}/{n_runs_per_round} (seed={seed})')
            res = instancia.solve(n_gen=200, pop_size=150, seed=seed)
            if res.F is None:
                print('    Nenhuma solução encontrada nesta execução. Pulando.')
                continue

            new_rows = build_solution_rows(res.X, res.F, N)
            new_df = pd.DataFrame(new_rows, columns=columns)
            new_df['round'] = t
            round_archive = pd.concat([round_archive, new_df], ignore_index=True)

        if round_archive.empty:
            print('Nenhuma solução obtida em nenhuma execução desta rodada. Interrompendo a evolução.')
            break

        round_archive = round_archive.drop_duplicates().reset_index(drop=True)
        round_archive = keep_nondominated_solutions(round_archive, objective_cols)
        print(f'  Frente de Pareto da rodada {t} gerada com {len(round_archive)} soluções não dominadas.')

        archive = pd.concat([archive, round_archive], ignore_index=True)
        archive = archive.drop_duplicates().reset_index(drop=True)
        archive.to_csv(log_path, index=False)
        print(f'Log compartilhado atualizado em {log_path} ({len(archive)} soluções acumuladas).')

        pesos = [0.1, 0.9, 0.1]
        idx = instancia.mcdm_pseudo_weights(pesos)
        instancia.scatterplot(file_name=f'Figuras/saida{t}.png')
        solucao_vars = None
        if idx is not None:
            solucao_vars = res.X[idx].copy()

        if solucao_vars is not None:
            beta_t = np.array([solucao_vars[f'beta_{n}'] for n in range(N)])
            instancia.beta_h += 1 - beta_t
            theta_t = np.array([solucao_vars[f'theta_{n}'] for n in range(N)])
            instancia.theta_prev = np.where(beta_t == 1, theta_t, instancia.theta_prev)
            for i in range(N):
                if solucao_vars[f'beta_{i}'] != 1:
                    solucao_vars[f'theta_{i}'] = instancia.theta_prev[i]

        t += 1

    archive.to_csv('evolucao_teorica/evolucao_teorica.csv', index=False)
    pd.DataFrame(saved_round_inputs).to_csv('evolucao_teorica/round_inputs.csv', index=False)
