import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from network import n, n_loads, R, X, netmodel, nu_lcf, nu_mcl, K

p_c_costs = np.array([2.0, 1.0, 2.2, 0.1, 3.0, 4.5]).reshape([1, n_loads])
p_r_costs = -1.0*np.ones([1, n_loads])

p_c_min = np.array([-0.02, -0.01, -0.05, -0.1, -0.03, 0.0]).reshape([n_loads, 1])
p_c_max = np.array([0.02, 0.01, 0.0, 0.0, 0.03, 0.1]).reshape([n_loads, 1])

p_r_min = np.zeros([n_loads, 1])
p_r_max = np.array([0.05, 0.2, 0.03, 0.08, 0.02, 0.025]).reshape([n_loads, 1])

v_min = 0.95
v_max = 1.05

p_r = cp.Variable([n_loads, 1])
p_c = cp.Variable([n_loads, 1])

p_r_lower = p_r >= p_r_min
p_r_upper = p_r <= p_r_max

p_c_lower = p_c >= p_c_min
p_c_upper = p_c <= p_c_max

nu_lower = np.ones([n, 1]) + nu_lcf + 2*R @ K @ (p_r_min + p_c) >= (v_min**2)*np.ones([n, 1])
nu_upper = np.ones([n, 1]) + nu_mcl + 2*R @ K @ (p_r + p_c) <= (v_max**2)*np.ones([n, 1])

objective = cp.Minimize(p_c_costs@p_c + p_r_costs@p_r)
constraints = [p_r_lower, p_r_upper, p_c_lower, p_c_upper, nu_lower, nu_upper]

problem = cp.Problem(objective, constraints)
result = problem.solve()
print(problem.status)
print(p_r.value)

v_wc_min = np.sqrt(np.ones([n, 1]) + nu_lcf + 2*R @ K @ (p_r_min + p_c.value))
print("Minimum Voltages:")
print(v_wc_min)

v_wc_max = np.sqrt(np.ones([n, 1]) + nu_mcl + 2*R @ K @ (p_r.value + p_c.value))
print("Maximum Voltages:")
print(v_wc_max)
print(constraints[4].dual_value)

fig, ax = plt.subplots()
ax.fill_between(range(2, n+2), v_wc_min.flatten(), v_wc_max.flatten(), alpha=0.5, color="#1A851D")
ax.plot(range(2, n+2), v_min*np.ones(n), color="#880404", linestyle="--")
ax.plot(range(2, n+2), v_max*np.ones(n), color="#880404", linestyle="--")
ax.set_ylabel("$V$ (p.u.)", fontname="Times New Roman", fontsize=14)
ax.set_xlabel("Bus", fontname="Times New Roman", fontsize=14)
ax.set_xlim([2, 19])
print(ax.get_xticks())
ax.set_xticks(np.arange(2, n+2, 2))
ax.set_xticklabels(np.arange(2, n+2, 2), fontname="Times New Roman")
yticks = ax.yaxis.get_ticklocs()
yticklabels = ["{:.2f}".format(tick) for tick in yticks]
ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels, fontname="Times New Roman")
#fig.savefig('VoltagePlot.eps', format='eps')
plt.show()