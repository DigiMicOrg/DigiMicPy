import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import param, analyse

# This is a basic code for simulating MiCRM

np.random.seed(42)

# parameters
N = 10  # consumer number
M = 5  # resource number
λ = 0.1  # total leakage rate

N_modules = 2 #  module number of consumer to resource
s_ratio = 10.0 

u = param.modular_uptake(N, M, N_modules, s_ratio)  # uptake matrix

lambda_alpha = np.full(M, λ)  # total leakage rate for each resource

m = np.full(N, 0.2)  # mortality rate of N consumers

rho = np.full(M, 0.5)  # input of M resources

omega = np.full(M, 0.5)  # decay rate of M resources

l = param.generate_l_tensor(N, M, N_modules, s_ratio, λ) # a tensor for all consumers' leakage matrics

# ode
def dCdt_Rdt(t, y):
    C = y[:N]
    R = y[N:]
    dCdt = np.zeros(N)
    dRdt = np.zeros(M)
    
    for i in range(N):
        dCdt[i] = sum(C[i] * R[alpha] * u[i, alpha] * (1 - lambda_alpha[alpha]) for alpha in range(M)) - C[i] * m[i]
    
    for alpha in range(M):
        dRdt[alpha] = rho[alpha] - R[alpha] * omega[alpha]
        dRdt[alpha] -= sum(C[i] * R[alpha] * u[i, alpha] for i in range(N))
        dRdt[alpha] += sum(sum(C[i] * R[beta] * u[i, beta] * l[i, beta, alpha] for beta in range(M)) for i in range(N))
    
    return np.concatenate([dCdt, dRdt])

# intial value
C0 = np.full(N,0.01)  # consumer
R0 = np.full(M,1)   # resource
Y0 = np.concatenate([C0, R0])

# time sacle
t_span = (0, 50)
t_eval = np.linspace(*t_span, 300)

# solve ode
sol = solve_ivp(dCdt_Rdt, t_span, Y0, t_eval=t_eval)

# plot
plt.figure(figsize=(10, 5))
for i in range(N):
    plt.plot(sol.t, sol.y[i], label=f'Microbe {i+1}', linewidth=2)
for alpha in range(M):
    plt.plot(sol.t, sol.y[N + alpha], label=f'Resource {alpha+1}', linestyle='dashed', linewidth=2)
plt.xlabel('Time', fontsize=20)
plt.ylabel('Consumer / Resource', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend()
plt.title('Dynamics of Consumers and Resources', fontsize=20)
plt.show()


'''
# system analysis
J = analyse.get_jac(dCdt_Rdt, sol) # Should be (N+M, N+M)
pur = np.random.rand(J.shape[0]) # random vector

t = 5 # time point
rin = analyse.get_Rins(J, pur, t) 

stability = analyse.get_stability(J)
if stability:
    print("Stable.")
else:
    print("Unstable.")

reactivity = analyse.get_reactivity(J, pur)
if reactivity:
    print("The system is reactive.")
else:
    print("The system is not reactive.")

return_rate = analyse.get_return_rate(J)
'''