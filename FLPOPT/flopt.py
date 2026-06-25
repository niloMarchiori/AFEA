import numpy as np
from .solver import FLSolver
from .problem import FederatedLearningProblem
from .flopt_util import print_solution_details
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
        self._beta_h=np.zeros(N)
        self._theta_prev=theta_prev

    @property
    def beta_h(self):
        return self._beta_h

    @beta_h.setter
    def beta_h(self,beta_h:np.array):
        self._beta_h=beta_h
        self.problem.beta_h=beta_h

    @property
    def theta_prev(self):
        return self._theta_prev

    @theta_prev.setter
    def theta_prev(self,theta:np.array):
        self._theta_prev=theta
        self.problem.theta_prev=theta


    def solve(self, n_gen=500, pop_size=150, **kwargs):
        self.solver = FLSolver(self.problem,pop_size=pop_size)
        self.res=self.solver.solve(n_gen=n_gen, **kwargs)
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


