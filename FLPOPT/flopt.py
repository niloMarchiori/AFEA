import numpy as np
from .solver import FLSolver
from .problem import FederatedLearningProblem
from .flopt_util import print_solution_details
from .parse import load_config
from pymoo.visualization.scatter import Scatter
from pymoo.mcdm.pseudo_weights import PseudoWeights
from pymoo.mcdm.high_tradeoff import HighTradeoffPoints

class FLPOPT:
    def __init__(self, N, alpha, c, S, f_min, f_max, epsilon_0, theta_prev, T_min=0.01, T_max=2**16,**kwargs):
        self.N = N
        self.c=c
        self.S=S
        self.problem = FederatedLearningProblem(N, alpha, c, S, f_min, f_max, epsilon_0, theta_prev, T_min, T_max)
        self.res = None
        self._unselected_count=np.zeros(N)
        self._theta_prev=theta_prev
        self.history = {
            "inputs": [],
            "found_solutions": [],
            "chosen_solutions": []
        }
        self.current_round = 0

    @property
    def unselected_count(self):
        return self._unselected_count

    @unselected_count.setter
    def unselected_count(self,unselected_count:np.array):
        self._unselected_count=unselected_count
        self.problem.unselected_count=unselected_count

    @property
    def theta_prev(self):
        return self._theta_prev

    @theta_prev.setter
    def theta_prev(self,theta:np.array):
        self._theta_prev=theta
        self.problem.theta_prev=theta


    def solve(self, n_gen=500, pop_size=150, theta_prev=None, unselected_count=None, **kwargs):
        if theta_prev is not None:
            self.theta_prev = np.array(theta_prev)
        if unselected_count is not None:
            self.unselected_count = np.array(unselected_count)
            
        run_input = {
            "round": self.current_round,
            "n_gen": n_gen,
            "pop_size": pop_size,
            "alpha": self.problem.alpha.tolist() if isinstance(self.problem.alpha, np.ndarray) else self.problem.alpha,
            "c": self.problem.c.tolist() if isinstance(self.problem.c, np.ndarray) else self.problem.c,
            "S": self.problem.S.tolist() if isinstance(self.problem.S, np.ndarray) else self.problem.S,
            "f_min": self.problem.f_min.tolist() if isinstance(self.problem.f_min, np.ndarray) else self.problem.f_min,
            "f_max": self.problem.f_max.tolist() if isinstance(self.problem.f_max, np.ndarray) else self.problem.f_max,
            "epsilon_0": self.problem.epsilon_0,
            "theta_prev": self._theta_prev.tolist() if isinstance(self._theta_prev, np.ndarray) else self._theta_prev,
            "unselected_count": self._unselected_count.tolist() if isinstance(self._unselected_count, np.ndarray) else self._unselected_count,
            "kwargs": kwargs
        }

        self.solver = FLSolver(self.problem,pop_size=pop_size)
        self.res=self.solver.solve(n_gen=n_gen, **kwargs)

        self.history["inputs"].append(run_input)
        self.history["found_solutions"].append({
            "F": self.res.F.tolist() if self.res.F is not None else None,
            "X": [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in x.items()} if isinstance(x, dict) else (x.tolist() if isinstance(x, np.ndarray) else x) for x in self.res.X] if self.res.X is not None else None
        })

        return self.res

    def scatterplot(self,file_name=None):
        if self.res is not None and self.res.F is not None:
            plot = Scatter(title="Fronteira de Pareto (3 Objetivos)", angle=(45, 45))
            plot.add(self.res.F)
            if not file_name:
                plot.show()
            else:
                plot.save(file_name)
        else:
            print("Nenhuma solução encontrada para plotar.")

    def mcdm_pseudo_weights(self, pesos, verbose=False):
        if not(self.res is not None and self.res.F is not None):
            print("Nenhuma solução encontrada para aplicar MCDM.")
            return None, None, None
        idx_escolhido = PseudoWeights(pesos).do(self.res.F)
        if verbose:
            objs=self.res.F[idx_escolhido]
            solucao_vars=self.res.X[idx_escolhido]
            print("\n--- SOLUÇÃO SELECIONADA PELO MÉTODO DE PSEUDO PESOS ---")
            print_solution_details(self.N,objs, solucao_vars,self.c,self.S)

        return idx_escolhido

    def mcdm_knee_point(self,verbose=False):
        if not(self.res is not None and self.res.F is not None):
            print("Nenhuma solução encontrada para identificar pontos de trade-off.")
            return None, None, None

        idx_knee = HighTradeoffPoints().do(self.res.F)
        
        if verbose:
            for idx in idx_knee:
                objs=self.res.F[idx]
                solucao_vars=self.res.X[idx]
                print(f"\n--- SOLUÇÃO {idx} SELECIONADA PELO MÉTODO DE PONTOS DE TRADE-OFF ---")
                print_solution_details(self.N,objs, solucao_vars,self.c,self.S)

        return idx_knee

    def advance_round(self, selected_idx):
        if not(self.res is not None and self.res.X is not None):
            print("Nenhuma solução para avançar rodada.")
            return

        sol = self.res.X[selected_idx]
        sol_f = self.res.F[selected_idx] if self.res.F is not None else None
        
        self.history["chosen_solutions"].append({
            "idx": int(selected_idx),
            "F": sol_f.tolist() if isinstance(sol_f, np.ndarray) else sol_f,
            "X": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in sol.items()} if isinstance(sol, dict) else (sol.tolist() if isinstance(sol, np.ndarray) else sol)
        })
        
        # Extrair beta_n e theta_n
        beta_vals = np.array([sol[f"beta_{n}"] for n in range(self.N)])
        theta_vals = np.array([sol[f"theta_{n}"] for n in range(self.N)])

        # Atualiza theta_prev apenas para dispositivos selecionados
        new_theta_prev = np.where(beta_vals == 1.0, theta_vals, self.theta_prev)
        self.theta_prev = new_theta_prev

        # Atualiza unselected_count: dispositivos não selecionados somam 1
        self.unselected_count = self.unselected_count + (1 - beta_vals)

        self.current_round += 1

    def save_history(self, filepath="history.json"):
        import json
        import numpy as np

        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                return super(NumpyEncoder, self).default(obj)

        with open(filepath, "w") as f:
            json.dump(self.history, f, indent=4, cls=NumpyEncoder)
