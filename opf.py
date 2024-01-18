import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from network import n, n_loads, R, X, netmodel, nu_lcf, nu_mcl, K, load_bus

p_c_costs = np.array([2.0, 1.0, 2.2, 0.1, 3.0, 4.5]).reshape([1, n_loads])
p_r_costs = -1.0*np.ones([1, n_loads])

p_c_min = np.array([-0.02, -0.01, -0.05, -0.1, -0.03, 0.0]).reshape([n_loads, 1])
p_c_max = np.array([0.02, 0.01, 0.0, 0.0, 0.03, 0.1]).reshape([n_loads, 1])

p_r_min = np.zeros([n_loads, 1])
p_r_max = np.array([0.03, 0.3, 0.04, 0.02, 0.08, 0.005]).reshape([n_loads, 1])

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
print("Renewable dispatch:")
print(p_r.value)
print("Controllable dispatch:")
print(p_c.value)

v_wc_min = np.sqrt(np.ones([n, 1]) + nu_lcf + 2*R @ K @ (p_r_min + p_c.value))
print("Minimum Voltages:")
print(v_wc_min)

v_wc_max = np.sqrt(np.ones([n, 1]) + nu_mcl + 2*R @ K @ (p_r.value + p_c.value))
print("Maximum Voltages:")
print(v_wc_max)
print(constraints[4].dual_value)

#plt.rc('text', usetex=True )
rc = {"font.family" : "serif", 
      "mathtext.fontset" : "stix"}
plt.rcParams.update(rc)
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
fig, ax = plt.subplots()
fig.set_figwidth(7.4)
#ax.fill_between(range(2, n+2), v_wc_min.flatten(), v_wc_max.flatten(), alpha=0.5, color="#1A851D")
ax.fill_between(range(2, n+2), v_wc_min.flatten(), v_wc_max.flatten(), facecolor="#7BBD75")
ax.plot(range(2, n+2), v_min*np.ones(n), color="#880404", linestyle="--")
ax.plot(range(2, n+2), v_max*np.ones(n), color="#880404", linestyle="--")
ax.set_ylabel("$V$ (p.u.)", fontname="Times New Roman", fontsize=14)
ax.set_xlabel("Bus", fontname="Times New Roman", fontsize=14)
ax.set_xlim([2, 19])
print(ax.get_xticks())
ax.set_xticks(np.arange(2, n+2, 2))
ax.set_xticklabels(np.arange(2, n+2, 2), fontname="Times New Roman", fontsize=12)
yticks = ax.yaxis.get_ticklocs()
yticklabels = ["{:.2f}".format(tick) for tick in yticks]
ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels, fontname="Times New Roman", fontsize=12)
fig.savefig('VoltagePlot.eps')

reduced_labels = [1, 2, 3, 4 ,5, 6, 7, 8, 9, 10]
reduced_buses = [0, 2, 3, 5, 8, 10, 14, 15, 16, 17]

x = np.linspace(0, 11, 100)

fig2, ax2 = plt.subplots()
fig2.set_figwidth(7.4)
#ax2.fill_between(range(0, len(load_bus)), v_wc_min.flatten()[load_bus], v_wc_max.flatten()[load_bus], facecolor="#7BBD75")
ax2.bar(reduced_labels, v_wc_max.flatten()[reduced_buses]-v_wc_min.flatten()[reduced_buses], bottom=v_wc_min.flatten()[reduced_buses], width=0.7, facecolor="#1A851D")
ax2.plot(x, v_min*np.ones_like(x), color="#880404", linestyle="--")
ax2.plot(x, v_max*np.ones_like(x), color="#880404", linestyle="--")
bus_markers = [i for i in range(0, len(reduced_buses)+2)]
ax2.plot(bus_markers, np.ones_like(bus_markers), marker='+', markersize=8, color="#444444")
ax2.set_ylabel("$V$ (p.u.)", fontname="Times New Roman", fontsize=14)
ax2.set_xlabel("Bus", fontname="Times New Roman", fontsize=14)
ax2.set_xticks([2, 4, 6, 8, 10])
ax2.set_xticklabels([2, 4, 6, 8, 10], fontname="Times New Roman", fontsize=12)
yticks = ax2.yaxis.get_ticklocs()
yticklabels = ["{:.2f}".format(tick) for tick in yticks]
ax2.set_yticks(yticks)
ax2.set_yticklabels(yticklabels, fontname="Times New Roman", fontsize=12)
ax2.set_xlim([0.5, 10.5])
ax2.set_ylim([0.92, 1.08])
plt.show()