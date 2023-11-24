import cvxpy as cp
import numpy as np
from network import n, R, X, netmodel, nu_lcf, nu_mcl


Pr = cp.Variable([n, 1])
Pc = cp.Variable([n, 1])

