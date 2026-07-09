import numpy as np
from pymoo.core.mixed import MixedVariableSampling, MixedVariableMating

class CustomFLSampling(MixedVariableSampling):
    def _do(self, problem, n_samples, **kwargs):
        # 1. Chama a amostragem aleatoria original (para beta, f, theta, etc)
        X = super()._do(problem, n_samples, **kwargs)
        
        # Extrai c_np e S_np de forma segura (numpy array)
        c_np = problem.c if isinstance(problem.c, np.ndarray) else problem.c.get()
        S_np = problem.S if isinstance(problem.S, np.ndarray) else problem.S.get()
        
        # 2. Modifica psi e T para guiar o algoritmo
        for i in range(n_samples):
            theta_vals = np.array([X[i][f"theta_{n}"] for n in range(problem.N)])
            beta_vals = np.array([X[i][f"beta_{n}"] for n in range(problem.N)])
            f_vals = np.array([X[i][f"f_{n}"] for n in range(problem.N)])
            
            # Psi(theta) = -log2(1 - theta)
            Psi_theta = -np.log2(1 - theta_vals)
            
            psi_vals = np.zeros(problem.N)
            for n in range(problem.N):
                if beta_vals[n] == 1:
                    # Perturbacao positiva de 0 ate 5*Psi
                    pert = np.random.uniform(0, 5 * Psi_theta[n])
                    psi = int(np.ceil(Psi_theta[n] + pert))
                else:
                    # Perturbacao negativa para respeitar psi <= Psi
                    pert = np.random.uniform(0, Psi_theta[n] - 0.001) if Psi_theta[n] > 1 else 0
                    psi = max(1, int(np.floor(Psi_theta[n] - pert)))
                
                # Respeita os limites maximos de psi da variavel original
                upper_bound = problem.vars[f"psi_{n}"].bounds[1]
                psi = min(psi, upper_bound)
                
                X[i][f"psi_{n}"] = psi
                psi_vals[n] = psi
            
            # Determina o valor base para T
            min_T = np.max(beta_vals * c_np * S_np * psi_vals / f_vals)
            # Perturbação aleatória positiva
            pert_T = np.random.uniform(0, 0.5 * min_T) if min_T > 0 else np.random.uniform(0, 1.0)
            X[i]["T"] = min_T + pert_T
            
        return X

class CustomFLMating(MixedVariableMating):
    def _do(self, problem, pop, n_offsprings, parents=None, **kwargs):
        # 1. Chama a mating original primeiro (cruzamento e mutação)
        off = super()._do(problem, pop, n_offsprings, parents=parents, **kwargs)
        
        # Obter a matriz de dicionarios
        X = off.get("X")
        
        c_np = problem.c if isinstance(problem.c, np.ndarray) else problem.c.get()
        S_np = problem.S if isinstance(problem.S, np.ndarray) else problem.S.get()
        
        # 2. Aplica o reparo customizado com 30% de probabilidade
        for i in range(len(X)):
            if np.random.rand() < 0.3:
                theta_vals = np.array([X[i][f"theta_{n}"] for n in range(problem.N)])
                beta_vals = np.array([X[i][f"beta_{n}"] for n in range(problem.N)])
                f_vals = np.array([X[i][f"f_{n}"] for n in range(problem.N)])
                
                Psi_theta = -np.log2(1 - theta_vals)
                
                psi_vals = np.zeros(problem.N)
                for n in range(problem.N):
                    if beta_vals[n] == 1:
                        psi = int(np.ceil(Psi_theta[n]))
                    else:
                        psi = max(1, int(np.floor(Psi_theta[n])))
                        
                    upper_bound = problem.vars[f"psi_{n}"].bounds[1]
                    psi = min(psi, upper_bound)
                    
                    X[i][f"psi_{n}"] = psi
                    psi_vals[n] = psi
                    
                min_T = np.max(beta_vals * c_np * S_np * psi_vals / f_vals)
                X[i]["T"] = min_T
                
        off.set("X", X)
        return off
