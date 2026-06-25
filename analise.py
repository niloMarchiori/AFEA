import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from instancia.parse import load_config

plt.rcParams.update({
    'font.size': 16,
    'legend.fontsize': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'lines.linewidth': 2,
    'figure.figsize': (12, 7),
})

# Valores padrão
DEFAULT_CSV_PATH = 'evolucao_teorica.csv'
DEFAULT_CONFIG_PATH = 'config.json'


def compute_consumption(df: pd.DataFrame, N: int, alpha, c, S) -> pd.DataFrame:
    device_consumption = pd.DataFrame(index=df.index)

    for i in range(N):
        f_i = df[f'f_{i}']
        beta_i = df[f'beta_{i}']
        psi_i = df[f'psi_{i}']

        device_consumption[f'consumo_sta{i}'] = (
            beta_i
            * psi_i
            * alpha[i]
            * c[i]
            * S[i]
            * (f_i ** 2)
            / 2
        )
        device_consumption[f'consumo_sta{i}'] *= 0.3

    device_consumption['consumo_total'] = device_consumption.sum(axis=1)
    df = df.copy()
    df['consumo_por_rodada'] = device_consumption['consumo_total']
    df['consumo_acumulado'] = df['consumo_por_rodada'].cumsum()
    return df, device_consumption


def add_metrics(df: pd.DataFrame, N: int) -> pd.DataFrame:
    df = df.copy()
    df['mean_acc'] = df[[f'theta_{i}' for i in range(N)]].mean(axis=1)
    df['selected_devices'] = df[[f'beta_{i}' for i in range(N)]].sum(axis=1)
    df['cum_time'] = df['T'].cumsum()
    return df


def plot_series(x, y, title, xlabel, ylabel, ax=None, style='-o'):
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(x, y, style)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    return ax


def main(csv_path: str, config_path: str):
    # Carregar configuração
    cfg = load_config(config_path)
    N = cfg['N']
    alpha = cfg['alpha']
    c = cfg['c']
    S = cfg['S']
    
    df = pd.read_csv(csv_path)
    df, df_consumo = compute_consumption(df, N, alpha, c, S)
    df = add_metrics(df, N)
    rounds = np.arange(1, len(df) + 1)

    print(df[['consumo_por_rodada', 'consumo_acumulado', 'mean_acc', 'selected_devices', 'cum_time']])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    plot_series(rounds, df['consumo_acumulado'],
                'Consumo Acumulado por Rodada',
                'Rodada',
                'Consumo Acumulado (unidades)',
                ax=axes[0, 0])

    plot_series(rounds, df['mean_acc'],
                'Média de Acurácia por Rodada',
                'Rodada',
                'Acurácia Média',
                ax=axes[0, 1])

    plot_series(rounds, df['selected_devices'],
                'Número de Dispositivos Selecionados por Rodada',
                'Rodada',
                'Dispositivos Selecionados',
                ax=axes[1, 0],
                style='-s')

    plot_series(df['cum_time'], df['mean_acc'],
                'Acurácia Média vs Tempo Acumulado',
                'Tempo Acumulado (T)',
                'Acurácia Média',
                ax=axes[1, 1])

    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Análise de evolução com configuração')
    parser.add_argument('--csv', '-d', default=DEFAULT_CSV_PATH, 
                        help=f'Caminho para arquivo de evolução CSV (padrão: {DEFAULT_CSV_PATH})')
    parser.add_argument('--config', '-c', default=DEFAULT_CONFIG_PATH,
                        help=f'Caminho para arquivo de configuração JSON (padrão: {DEFAULT_CONFIG_PATH})')
    args = parser.parse_args()
    
    main(args.csv, args.config)
