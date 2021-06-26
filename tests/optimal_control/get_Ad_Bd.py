# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 15:03:55 2021

@author: Saber
"""
import numpy as np
import scipy.linalg as la
from helper_funcs import sumOfSines, RK4Step, mat_mul_series, Run_Data
import pickle

m1 = 1
# m2 = 1
M = 5./2
k1 = 100.#1000 #For k>100, x1, xm remain indistinguishably close.
# k2 = 100.
zeta = 1.
b = zeta * np.sqrt((M+m1)*8*k1)

X1i, XMi, dX1i, dXMi = 0,1,2,3
sys_MSMSM = lambda q, u, t: np.array([q[dX1i],
                                      q[dXMi],
                                      (1/m1)*( u[0] + k1*(q[XMi]-q[X1i]) + b*(q[dXMi]-q[dX1i])),
                                      (1/M)*(k1*(q[X1i]-q[XMi]) + b*(q[dX1i]-q[dXMi]))])

# nX = 2
# nXXd= 2*nX
# nS = nXXd+1 # one for the constant term
# nA = 1

sys_Ac = np.array([
    [0, 0, 1., 0],
    [0, 0, 0, 1.],
    [-k1/m1, k1/m1, -b/m1, b/m1],
    [k1/M, (-k1)/M, b/M, -b/M],
     ])

sys_Bc = np.array([
    [0],
    [0],
    [1./m1],
    [0]
    ])

dt_d=0.02
sys_Ad = la.expm(sys_Ac*dt_d) #exp(Ac*Ts)
sys_Bd = mat_mul_series(
    la.pinv2(sys_Ac),
    sys_Ad-np.eye(len(sys_Ad)),
    sys_Bc) #inv(A)*(Ad-I) Bc
    
with open('Ad_Bd_single_Ts0.02.pickle', 'wb') as f:
    pickle.dump((sys_Ad, sys_Bd), f)
