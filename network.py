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

net.load.p_mw[2]=0.5*net.load.p_mw[2]
net.load.q_mvar[2]=0.5*net.load.q_mvar[2]

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

x_line = (1/z_base)*netmodel.line.x_ohm_per_km*netmodel.line.length_km
X = F @ np.diag(x_line) @ F.T

buses = np.zeros((n, 1))
P = np.zeros((n, 1))
Q = np.zeros((n, 1))

for b, p, q in zip(netmodel.load.bus, netmodel.load.p_mw, netmodel.load.q_mvar):
    P[b-2]=p
    Q[b-2]=q



nu = (net.ext_grid.vm_pu[0]**2)*np.ones((n, 1)) - 2*R @ P - 2*X @ Q
vlin=np.sqrt(nu)

vmod = net.res_bus.vm_pu[1:].array.reshape(n, 1)

error = 100*(vlin-vmod)/(np.ones((n, 1))-vlin)

# Load coincidence factors
# Values from Kersting
lcf = [1.0, 1/1.6, 1/1.8, 1/2.1, 1/2.2, 1/2.3]
# Minimum coincident loads (needs to be demonstrated from data)
# Ignoring MCL characteristics
mcl = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

nu_lcf = np.zeros((n, 1))
nu_mcl = np.zeros((n, 1))

downstream_of_line = [[b-2 for b in downstream_buses(netmodel, ln)] for ln in range(0, n)]


# Calculate load contribution to minimum and maximum squared voltage
for i in range(0, len(netmodel.bus)-1):
    lines = upstream_lines(netmodel, i+2)
    for j in range(0, len(lines)):
        downstream = downstream_of_line[lines[j]]
        n_loads = sum(P[downstream]!= 0)[0]
        nu_lcf[i] -= 2*lcf[n_loads-1]*(r_line[lines[j]]*sum(P[downstream]) + x_line[lines[j]]*sum(Q[downstream]))
        nu_mcl[i] -= 2*mcl[n_loads-1]*(r_line[lines[j]]*sum(P[downstream]) + x_line[lines[j]]*sum(Q[downstream]))
        
v_lcf = np.sqrt(np.ones([n, 1]) + nu_lcf)
v_mcl = np.sqrt(np.ones([n, 1]) + nu_mcl)

load_bus = [i for i, p in zip(range(0, len(P)), P) if p!=0.0]
n_l = len(load_bus)
K = np.zeros([n, n_l])
for i in range(0, n):
    for j in range(0, n_l):
        K[i, j] = i==load_bus[j]

def show_results():
    print(net.res_bus)
    print("Transformer resistance: {}, reactance: {}".format(rt, xt))
    print("A:")
    print(A)
    print("R:")
    print(R)
    print("X:")
    print(X)
    print("vlin:")
    print(vlin)
    print("error:")
    print(error)
    print("v_lcf:")
    print(v_lcf)
    print("v_mcl")
    print(v_mcl)
    fig, ax = plt.subplots()
    ax.fill_between(range(2, n+2), v_lcf.flatten(), v_mcl.flatten(), alpha=0.5, color="green")
    plt.show()

if __name__=="__main__":
    show_results()
