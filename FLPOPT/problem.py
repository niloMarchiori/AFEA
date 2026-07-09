
import numpy as np
try:
    import cupy as cp
    xp = np
    _USE_CUPY = False
except Exception:
    xp = np
    _USE_CUPY = False
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Real, Integer, Binary

class FederatedLearningProblem(ElementwiseProblem):
    def __init__(self, N, alpha, c, S, f_min, f_max, epsilon_0, theta_prev=0.01, T_min=0.0, T_max=np.inf, unselected_count=None):
        self.N = N
        self.alpha = xp.array(alpha)
        self.c = xp.array(c)
        self.S = xp.array(S)
        self.f_min = xp.array(f_min) *1e9
        self.f_max = xp.array(f_max) *1e9
        self.epsilon_0 = epsilon_0
        self.theta_prev = xp.array(theta_prev)
        self.unselected_count = unselected_count
        if unselected_count is None:
            self.unselected_count = xp.zeros(N)
        
        # Construindo o dicionário de Variáveis Mistas
        vars_dict = {}
        
        # Variável T (Contínua, Única global) - Limite finito para evitar NaN no Pymoo
        vars_dict["T"] = Real(bounds=(T_min, 1e6 if T_max == np.inf else T_max))
        
        psi_upper_bound = max(30, int(5 * (-np.log2(1 - self.epsilon_0))))
        
        for n in range(N):
            # f_n (Contínua)
            vars_dict[f"f_{n}"] = Real(bounds=(self.f_min[n], self.f_max[n]))
            # beta_n (Binária)
            vars_dict[f"beta_{n}"] = Binary()
            # psi_n (Inteira) - N*
            vars_dict[f"psi_{n}"] = Integer(bounds=(1, psi_upper_bound))
            # theta_n (Contínua) - Limitada em [0.01, 0.99] para evitar div/0 e log(0)
            vars_dict[f"theta_{n}"] = Real(bounds=(0.01, 0.9999))
            
        super().__init__(
            vars=vars_dict,
            n_obj=3,          # AGORA TEMOS 3 OBJETIVOS
            n_ieq_constr=3*N +1 # TEMOS 3 RESTRIÇÕES PARA CADA 'n'
        )

    def _evaluate(self, x, out, *args, **kwargs):
        # 1. Extração das variáveis
        T_val = x["T"]
        f_vals = xp.array([x[f"f_{n}"] for n in range(self.N)])
        beta_vals = xp.array([x[f"beta_{n}"] for n in range(self.N)])
        psi_vals = xp.array([x[f"psi_{n}"] for n in range(self.N)])
        theta_vals = xp.array([x[f"theta_{n}"] for n in range(self.N)])

        # Impor limites artificialmente no evaluate
        f_vals = xp.clip(f_vals, self.f_min, self.f_max)
        psi_vals = xp.maximum(psi_vals, 1)
        theta_vals = xp.clip(theta_vals, 0.001, 0.999)

        # 2. Cálculo das Funções G(theta) e Psi(theta)
        # G(theta_n) = - log(1 - epsilon_0) / theta_n
        G_theta = -xp.log2(1 - self.epsilon_0) / theta_vals
        
        # Psi(theta_n) = - log(1 - theta_n)
        Psi_theta = -xp.log2(1 - theta_vals)

        # ====================================
        # FUNÇÕES OBJETIVO
        # ====================================
        # f1: min \sum (beta_n * psi_n * G(theta_n) * (alpha_n/2) * c_n * S_n * f_n^2)
        obj1 = xp.sum(beta_vals * psi_vals * G_theta * (self.alpha / 2) * self.c * self.S * (f_vals**2))
        
        # f2: max \sum beta_n -> min -\sum beta_n
        rec = (self.S / self.S.sum()) * (self.unselected_count)
        obj2 = beta_vals * (1 + rec)
        obj2 = -xp.sum(obj2)
        
        # f3: min G(theta_n) * T
        # Assumindo que queremos minimizar o tempo total ponderado pelos clientes selecionados
        obj3 = (G_theta*beta_vals).max() * T_val
        
        # ====================================
        # RESTRIÇÕES (g <= 0)
        # ====================================
        # g1: (psi_n * c_n * S_n / f_n) <= T  ==>  (psi_n * c_n * S_n / f_n) - T <= 0
        g1 = beta_vals * (psi_vals * self.c * self.S / f_vals) - T_val
        
        # g2: 
        # Se beta_n == 1: psi_n >= Psi(theta_n) ==> Psi(theta_n) - psi_n <= 0
        # Se beta_n == 0: psi_n <= Psi(theta_n) ==> psi_n - Psi(theta_n) <= 0
        g2 = beta_vals * (Psi_theta - psi_vals) + (1 - beta_vals) * (psi_vals - Psi_theta)
        
        # g3: beta_n * theta_n >= beta_n * theta_n^{t-1}  ==>  beta_n * (theta_n^{t-1} - theta_n) <= 0
        g3 = (self.theta_prev * 0.99 - theta_vals)

        g4 = xp.array([1 - beta_vals.sum()])

        # O Pymoo exige que todas as restrições sejam passadas como uma lista/array 1D
        g_all = xp.concatenate([g1, g2, g3, g4])

        # Converter de volta para numpy caso estejamos usando CuPy
        if _USE_CUPY:
            F_np = xp.asnumpy(xp.array([obj1, obj2, obj3]))
            G_np = xp.asnumpy(g_all)
            out["F"] = [float(F_np[0]), float(F_np[1]), float(F_np[2])]
            out["G"] = G_np
        else:
            out["F"] = [obj1, obj2, obj3]
            out["G"] = g_all