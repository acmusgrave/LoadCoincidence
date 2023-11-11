import pandapower as pp
import numpy as np

net_full = pp.networks.create_cigre_network_lv()

net = pp.create.create_empty_network()

net.bus = net_full.bus[["R" in name for name in net_full.bus.name]]
net.load = net_full.load[["R" in name for name in net_full.load.name]]
net.line = net_full.line[["R" in name for name in net_full.line.name]]
net.trafo = net_full.trafo[["R" in name for name in net_full.trafo.name]]
pp.create_ext_grid(net, 1, vm_pu=1.04)

pp.runpp(net)

# Extract tranformer matrix from Ybus
Ybus = net._ppc["internal"]["Ybus"].todense()
zt = -1/Ybus[0, 1]
rt = zt.real
xt=zt.imag

n = len(net.line)+1
A = np.zeros((n, n))

A[0, 0] = -1
A[1, 0] = 1
A[1, 1] = -1
for i in range(2, n):
    A[i, net.line.iloc[i-1].from_bus-2]=1
    A[i, net.line.iloc[i-1].to_bus-2]=-1
    
F = np.linalg.inv(A)

Zbase = (400**2)/1e6

linediagr = (1/Zbase)*np.diag(net.line.r_ohm_per_km*net.line.length_km)
diagr = np.block([[rt, np.zeros((1, n-1))],[np.zeros((n-1, 1)), linediagr]])
R = F @ diagr @ F.T
print(R)

linediagx = (1/Zbase)*np.diag(net.line.x_ohm_per_km*net.line.length_km)
diagx = np.block([[xt, np.zeros((1, n-1))],[np.zeros((n-1, 1)), linediagr]])
X = F @ diagx @ F.T
print(X)

P = np.zeros((n, 1))
Q = np.zeros((n, 1))

for b, p, q in zip(net.load.bus, net.load.p_mw, net.load.q_mvar):
    P[b-2]=p
    Q[b-2]=q

print(net.res_bus)
print("Transformer resistance: {}, reactance: {}".format(rt, xt))
print(A)

nu = (net.ext_grid.vm_pu[0]**2)*np.ones((n, 1)) - 2*R @ P - 2*X @ Q
print(np.sqrt(nu))