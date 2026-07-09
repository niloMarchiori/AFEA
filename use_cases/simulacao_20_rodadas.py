import argparse
import sys
from pathlib import Path
import numpy as np

# Ajuste do path para importar FLPOPT corretamente
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from FLPOPT.flopt import FLPOPT
from FLPOPT import load_config

def main():
    p = argparse.ArgumentParser(description='Executa N rodadas de otimização e salva histórico em JSON')
    p.add_argument('--config', '-c', default=str(REPO_ROOT / 'configs' / 'config_afea_n11.json'), help='Caminho para o arquivo de configuração (JSON)')
    p.add_argument('--rounds', '-r', type=int, default=20, help='Número de rodadas para avançar')
    p.add_argument('--output', '-o', default=str(REPO_ROOT / 'use_cases' / 'historico_20_rodadas.json'), help='Caminho do arquivo de saída do histórico JSON')
    p.add_argument('--ngen', type=int, default=200, help='Número de gerações do NSGA-II por rodada')
    p.add_argument('--pop', type=int, default=120, help='Tamanho da população por rodada')
    args = p.parse_args()

    # Carrega configurações
    cfg = load_config(args.config)
    
    # Inicializa a instância FLPOPT
    flopt = FLPOPT(
        cfg['N'], 
        cfg['alpha'], 
        cfg['c'], 
        cfg['S'], 
        cfg['f_min'], 
        cfg['f_max'], 
        cfg['epsilon_0'], 
        cfg['theta_prev']
    )

    print(f"Iniciando simulação de {args.rounds} rodadas contínuas...")
    
    # Pesos utilizados na tomada de decisão (MCDM) a cada rodada
    # [Energia (f1), Unselected Count Penalty (f2), Tempo (f3)]
    pesos = [0.3, 0.4, 0.3] 

    for r in range(args.rounds):
        print(f"\n[{r+1}/{args.rounds}] --- EXECUTANDO RODADA {r} ---")
        # Roda a otimização
        flopt.solve(n_gen=args.ngen, pop_size=args.pop)
        
        # Seleciona uma solução usando MCDM com pseudo pesos
        idx = flopt.mcdm_pseudo_weights(pesos, verbose=False)
        
        if idx is None:
            print(f"Nenhuma solução encontrada na rodada {r}, interrompendo.")
            break
            
        print(f"Solução escolhida na fronteira de Pareto (índice {idx})")
        print(f"Estado de unselected_count antes da atualização: {flopt.unselected_count}")
        
        # Avança a rodada, atualizando unselected_count e theta_prev internamente
        flopt.advance_round(idx)

    # Ao final, salva todas as entradas e saídas no arquivo escolhido
    print(f"\nSalvando o histórico das rodadas no arquivo {args.output}...")
    flopt.save_history(args.output)
    print("Simulação concluída com sucesso!")

if __name__ == '__main__':
    main()
