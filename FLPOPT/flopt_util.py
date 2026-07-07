
import numpy as np
import numpy as np
import pandas as pd
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.indicators.hv import Hypervolume
from pymoo.indicators.igd import IGD
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.mixed import MixedVariableMating, MixedVariableSampling, MixedVariableDuplicateElimination

def print_solution_details(N,objs, solucao_vars,c,s):
    print(f"Objetivos -> f1 (Energia): {objs[0]:.4e} | f2 (Beta): {objs[1]:.2f} | f3 (Tempo*Rodadas): {objs[2]:.4f}")
    print(f"Variável Global -> T = {solucao_vars['T']:.4f}")
    print("Variáveis Locais:")
    for n in range(N):
        val_f = solucao_vars[f"f_{n}"]
        val_beta = solucao_vars[f"beta_{n}"]
        val_psi = solucao_vars[f"psi_{n}"]
        val_theta = solucao_vars[f"theta_{n}"]
        print(f"  n={n}: f_{n} = {val_f:.4e} | beta_{n} = {f'{val_beta} ' if val_beta else val_beta } | psi_{n} = {val_psi} | PSI_{n}={-np.log2(1 - val_theta):.4f} | theta_{n} = {val_theta:.4f} | T_{n} = {val_psi*c[n]*s[n]/val_f:.4f}")