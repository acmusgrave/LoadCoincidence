import pandapower as pp
import numpy as np
import matplotlib.pyplot as plt

# List of bus indices downstream of line of index n
def downstream_buses(net, n):
    downstream = [net.line.iloc[n].to_bus]
    for i in range(0, len(net.line)):
        if net.line.iloc[i].from_bus==net.line.iloc[n].to_bus:
            # downstream.append(net.line.iloc[i].to_bus)
            downstream += downstream_buses(net, i)         
    return downstream
    
# List of line indices upstream of bus of index n
def upstream_lines(net, n):
    upstream = []
    for i in range(0, len(net.line)):
        if net.line.iloc[i].to_bus==n:
            upstream.append(i)
            upstream += upstream_lines(net, net.line.iloc[i].from_bus)
    return upstream
    

net_full = pp.networks.create_cigre_network_lv()

net = pp.create.create_empty_network()

net.bus = net_full.bus[["R" in name for name in net_full.bus.name]]
net.load = net_full.load[["R" in name for name in net_full.load.name]]
net.line = net_full.line[["R" in name for name in net_full.line.name]]
net.trafo = net_full.trafo[["R" in name for name in net_full.trafo.name]]
pp.create_ext_grid(net, 1, vm_pu=1)

#net.load.p_mw[2]=0
#net.load.q_mvar[2]=0

pp.runpp(net)
z_base = (400**2)/1e6

netmodel = pp.create.create_empty_network()
netmodel.bus=net.bus.copy()
netmodel.load=net.load.copy()
netmodel.line=net.line.copy()

# Extract tranformer impedence from Ybus
Ybus = net._ppc["internal"]["Ybus"].todense()
zt = -1/Ybus[0, 1]
rt = zt.real
xt = zt.imag

# Replace tranformer impedence with equivalent line
pp.create_line_from_parameters(netmodel, from_bus=1, to_bus=2, length_km=1, r_ohm_per_km=z_base*rt, x_ohm_per_km=z_base*xt, c_nf_per_km=0, max_i_ka=1.0, index=-1)
netmodel.line.index = netmodel.line.index+1
netmodel.line.sort_index(inplace=True)


n = len(netmodel.line)
A = np.zeros((n, n))

A[0, 0] = -1
for i in range(1, n):
    A[i, netmodel.line.iloc[i].from_bus-2]=1
    A[i, netmodel.line.iloc[i].to_bus-2]=-1
    
F = np.linalg.inv(A)

r_line = (1/z_base)*netmodel.line.r_ohm_per_km*netmodel.line.length_km
R = F @ np.diag(r_line) @ F.T
print(R)

x_line = (1/z_base)*netmodel.line.x_ohm_per_km*netmodel.line.length_km
X = F @ np.diag(x_line) @ F.T
print(X)

buses = np.zeros((n, 1))
P = np.zeros((n, 1))
Q = np.zeros((n, 1))

for b, p, q in zip(netmodel.load.bus, netmodel.load.p_mw, netmodel.load.q_mvar):
    P[b-2]=p
    Q[b-2]=q

print(net.res_bus)
print("Transformer resistance: {}, reactance: {}".format(rt, xt))
print(A)

nu = (net.ext_grid.vm_pu[0]**2)*np.ones((n, 1)) - 2*R @ P - 2*X @ Q
vlin=np.sqrt(nu)
print(vlin)

vmod = net.res_bus.vm_pu[1:].array.reshape(n, 1)

error = 100*(vlin-vmod)/(np.ones((n, 1))-vlin)
print(error)

c = [1, 0.8, 0.7, 0.65, 0.625, 0.6]
m = [0, 0.04, 0.06, 0.07, 0.075, 0.0775]

nu_lcf = np.ones((n, 1))
nu_mcl = np.ones((n, 1))


# TODO: clean up indexing logic here
for i in range(0, len(netmodel.bus)-1):
    lines = upstream_lines(netmodel, i+2)
    for j in range(0, len(lines)):
        downstream = [b-2 for b in downstream_buses(netmodel, lines[j])]
        n_loads = sum(P[downstream]!= 0)[0]
        nu_lcf[i] -= 2*c[n_loads-1]*(r_line[lines[j]]*sum(P[downstream]) + x_line[lines[j]]*sum(Q[downstream]))
        nu_mcl[i] -= 2*m[n_loads-1]*(r_line[lines[j]]*sum(P[downstream]) + x_line[lines[j]]*sum(Q[downstream]))
        
v_lcf = np.sqrt(nu_lcf)
v_mcl = np.sqrt(nu_mcl)

print(v_lcf)
print(v_mcl)


fig, ax = plt.subplots()
ax.plot(v_lcf)
ax.plot(v_mcl)
ax.plot(vmod)
plt.show()
