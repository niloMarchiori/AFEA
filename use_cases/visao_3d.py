### Plot 3D Interativo
# Para que a interatividade funcione, certifique-se de que a célula seja executada.
# Dependendo da sua versão do Jupyter, pode ser necessário reiniciar o Kernel ao alternar backends do matplotlib.

import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

with open('/home/nilo/Documents/AFEA/use_cases/historico_20_rodadas.json', 'r') as f:
    history = json.load(f)

print(f"Número de rodadas na simulação: {len(history['chosen_solutions'])}")

import os
os.makedirs('./Figuras', exist_ok=True)
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14, 'font.family': 'serif', 'axes.labelsize': 16, 'xtick.labelsize': 12, 'ytick.labelsize': 12})


df_inputs = pd.DataFrame(history['inputs'])

chosen_records = []
for r, sol in enumerate(history['chosen_solutions']):
    record = {'Rodada': r, 'idx_fronteira': sol.get('idx', np.nan)}
    
    # Objetivos
    if sol['F'] is not None:
        record['f1_energia'] = sol['F'][0]
        record['f2_unselected'] = sol['F'][1]
        record['f3_tempo'] = sol['F'][2]
        
    # Variáveis de decisão X
    for k, v in sol['X'].items():
        record[k] = v
        
    chosen_records.append(record)

df_chosen = pd.DataFrame(chosen_records)

N_devices = len(df_inputs['c'][0])
c_vals = np.array(df_inputs['c'][0])
S_vals = np.array(df_inputs['S'][0])

df_chosen.head()

total_rounds = len(df_chosen)

rounds_to_plot = [0, (total_rounds - 1) // 2, total_rounds - 1]
df_filtered = df_chosen[df_chosen['Rodada'].isin(rounds_to_plot)]

for r in rounds_to_plot:
    if r >= len(history['inputs']): continue
    solutions = history['found_solutions'][r]
    f1, f2, f3 = [], [], []
    for sol in solutions['F']:
        f1.append(sol[0])
        f2.append(sol[1])
        f3.append(sol[2])
    
    chosen_sol = df_filtered[df_filtered['Rodada'] == r].iloc[0]
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(f1, f2, f3, color='gray', alpha=0.5)
    ax.scatter(chosen_sol['f1_energia'], chosen_sol['f2_unselected'], chosen_sol['f3_tempo'], color='red', s=150, label='Escolhida', zorder=5)
    
    ax.set_xlabel('F1 (Energia)')
    ax.set_ylabel('F2 (Unselected)')
    ax.set_zlabel('F3 (Tempo)')
    ax.set_title(f'Visualização 3D Interativa - Rodada {r}')
    ax.legend()
    
    plt.show()
