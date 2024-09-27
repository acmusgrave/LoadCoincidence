import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import pandapower as pp

# Network R and X matrices and load contributions to min/max squared voltage imported from network.py
from network import n, n_loads, R, X, net, nu_lcf, nu_mcl, K, load_bus, lcf


p_c_costs = np.array([2.0, 1.0, 2.2, 0.1, 3.0, 4.5]).reshape([1, n_loads])
p_r_firm_costs = -1e-6*np.ones([1, n_loads])
p_r_dne_costs = -1.0*np.ones([1, n_loads])

p_c_min = np.array([-0.02, -0.01, -0.05, -0.1, -0.03, 0.0]).reshape([n_loads, 1])
p_c_max = np.array([0.02, 0.01, 0.0, 0.0, 0.03, 0.1]).reshape([n_loads, 1])

p_r_max = np.array([0.03, 0.3, 0.04, 0.2, 0.5, 0.005]).reshape([n_loads, 1])
p_r_min = 0.2*p_r_max

v_min = 0.95
v_max = 1.05
p_r_firm = cp.Variable([n_loads, 1])
p_r_dne = cp.Variable([n_loads, 1])
p_c = cp.Variable([n_loads, 1])

p_r_firm_lower = p_r_firm >= np.zeros(n_loads).reshape([n_loads, 1])
p_r_firm_upper = p_r_firm <= p_r_min

p_r_dne_lower = p_r_dne >= p_r_firm
p_r_dne_upper = p_r_dne <= p_r_max

p_c_lower = p_c >= p_c_min
p_c_upper = p_c <= p_c_max


# OPF bounds on squared voltage taking into account LCF/MCL, controllable DERs, and renewable bounds
nu_lower = np.ones([n, 1]) + nu_lcf + 2*R @ K @ (p_r_firm + p_c) >= (v_min**2)*np.ones([n, 1])
nu_upper = np.ones([n, 1]) + nu_mcl + 2*R @ K @ (p_r_dne + p_c) <= (v_max**2)*np.ones([n, 1])

objective = cp.Minimize(p_c_costs@p_c + p_r_firm_costs@p_r_firm + p_r_dne_costs@p_r_dne)
constraints = [p_r_firm_lower, p_r_firm_upper, p_r_dne_lower, p_r_dne_upper, p_c_lower, p_c_upper, nu_lower, nu_upper]

problem = cp.Problem(objective, constraints)
result = problem.solve()
print(problem.status)
print("Renewable dispatch:")
print(p_r_dne.value)
print("Controllable dispatch:")
print(p_c.value)

v_wc_min = np.sqrt(np.ones([n, 1]) + nu_lcf + 2*R @ K @ (p_r_firm.value + p_c.value))
print("Minimum Voltages:")
print(v_wc_min)

v_wc_max = np.sqrt(np.ones([n, 1]) + nu_mcl + 2*R @ K @ (p_r_dne.value + p_c.value))
print("Maximum Voltages:")
print(v_wc_max)
print(constraints[4].dual_value)

n_samples = 100
# Generate artificial load data conforming to LCF
load_samples = []
load_indices = list(range(0, n_loads))
p_max = net.load.p_mw
while len(load_samples) < n_samples:
    l_s = p_max*np.random.rand(n_loads)
    valid_sample = True
    for i in load_indices:
        for c in combinations(load_indices, i+1):
            if sum(l_s[list(c)])/sum(p_max[list(c)]) > lcf[i]:
                valid_sample = False
                print("Sample rejected at C({})={}".format(i+1, lcf[i]))
                break
            if not valid_sample:
                break
    if valid_sample:
        print("Sample accepted")
        load_samples.append(l_s)



net.load.q_mvar = np.zeros_like(load_samples[0])
voltage_profiles = []

for sample in load_samples:
    renewable_sample = np.min(np.hstack([p_r_dne.value, p_r_min + (p_r_max-p_r_min)*np.random.rand(n_loads).reshape([-1, 1])]), axis=1).reshape([-1, 1])
    net.load.p_mw = sample.to_numpy().reshape([-1, 1]) - p_c.value - renewable_sample
    pp.runpp(net, numba=False)
    voltage_profiles.append(net.res_bus.vm_pu[1:].to_numpy())


# Setup plots
rc = {"font.family" : "serif", 
      "mathtext.fontset" : "stix"}
plt.rcParams.update(rc)
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]


reduced_labels = [1, 2, 3, 4 ,5, 6, 7, 8, 9, 10]
reduced_buses = [0, 2, 3, 5, 8, 10, 14, 15, 16, 17]

x = np.linspace(0, 11, 100)

fig, ax = plt.subplots()
ax.bar(reduced_labels, v_wc_max.flatten()[reduced_buses]-v_wc_min.flatten()[reduced_buses], bottom=v_wc_min.flatten()[reduced_buses], width=0.7, facecolor="#7BBD7B", edgecolor="#1A851D", linewidth=0.5)
for profile in voltage_profiles:
    ax.plot(reduced_labels, profile[reduced_buses], marker='x', markersize=5, linestyle='', color='#1F2139')
ax.plot(x, v_min*np.ones_like(x), color="#880404", linestyle="--")
ax.plot(x, v_max*np.ones_like(x), color="#880404", linestyle="--")
bus_markers = [i for i in range(0, len(reduced_buses)+2)]
ax.plot(bus_markers, np.ones_like(bus_markers), color="#444444", linewidth=1)
ax.set_ylabel("Nodal Voltage Magnitude [p.u.]", fontname="Times New Roman", fontsize=14)
ax.set_xlabel("Node", fontname="Times New Roman", fontsize=14)
ax.set_xticks(reduced_labels)
ax.set_xticklabels(reduced_labels, fontname="Times New Roman", fontsize=14)
yticks = ax.yaxis.get_ticklocs()
yticklabels = ["{:.2f}".format(tick) for tick in yticks]
ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels, fontname="Times New Roman", fontsize=14)
ax.set_xlim([0.5, 10.5])
ax.set_ylim([0.92, 1.08])
plt.show()