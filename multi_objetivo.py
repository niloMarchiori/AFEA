import argparse
import json
import numpy as np
from FLPOPT.flopt import FLPOPT
from FLPOPT.flopt_util import print_solution_details
from instancia.parse import load_config

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Executa otimização multi-objetivo lendo config JSON')
    p.add_argument('--config', '-c', default='config.json', help='Caminho para o arquivo de configuração (JSON)')
    args = p.parse_args()

    cfg = load_config(args.config)
    instancia = FLPOPT(cfg['N'], cfg['alpha'], cfg['c'], cfg['S'], cfg['f_min'], cfg['f_max'], cfg['epsilon_0'], cfg['theta_prev'])

    print('Iniciando a otimização com 3 objetivos...')
    res = instancia.solve(n_gen=200, pop_size=100, seed=1)

# ======================================================================
# 4. EXIBIÇÃO DOS RESULTADOS E PLOTAGEM
# ======================================================================
print("\n--- DETALHAMENTO DAS SOLUÇÕES DA FRONTEIRA DE PARETO ---")

if res.F is not None:
    # for i in range(len(res.X)):
    #     solucao_vars = res.X[i]
    #     objs = res.F[i]
    #     print_solution_details(N, objs, solucao_vars, c, S)

    #--- SOLUÇÃO SELECIONADA PELO MÉTODO DE MCDM (PSEUDO PESOS) ---#
    pesos = [0.8, 0.3, 0.1]  # Exemplo de pesos para os objetivos
    idx= instancia.mcdm_pseudo_weights(pesos,verbose=True)

    #--- SOLUÇÃO SELECIONADA PELO MÉTODO DE MCDM (High Tradeoff Points) ---#
    # idx_knee = instancia.mcdm_knee_point(verbose=True)
    
    
    
    # instancia.scatterplot()
    
            
else:
    print("Nenhuma solução viável foi encontrada. Ajuste os demais parâmetros ou as restrições de evolução.")