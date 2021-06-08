# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 13:03:04 2021

@author: Saber
"""

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import functools, itertools
from collections import namedtuple
from itertools import product
import time
from helper_funcs import sumOfSines, RK4Step, mat_mul_series, Run_Data
import pickle

######################
# Sys config
######################

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


######################
# Planner config
######################

def get_opt_data(qs, rs=1., 
                 Th=3., replanning_period=1., dt_d=0.005, 
                 sys_Ac=sys_Ac, sys_Bc=sys_Bc):
    
    nXXd, nA = sys_Bc.shape
    nX = int(nXXd/2)
    nS = nXXd+1 #the additional dimension for the constant
    
    qe, qed = qs
    Q_mat = 1e-3*np.eye(nS)
    Q_mat[XMi,XMi] = qe#0
    Q_mat[dXMi,dXMi] = qed#0
    
    R_mat = rs* np.eye(nA)
    N_mat = np.zeros((nS, nA))
    
    # dt_d = 0.005 #discretization period. 0.001<  <0.01
    # Th = 3. #0.2 #plannig horizon
    # replanning_period = 1.#0.2#dt_d
    
    
    nsteps = int(Th/dt_d)
    
    # Precomputed variables
    sys_Ad = la.expm(sys_Ac*dt_d) #exp(Ac*Ts)
    sys_Bd = mat_mul_series(
        la.pinv2(sys_Ac),
        sys_Ad-np.eye(len(sys_Ad)),
        sys_Bc) #inv(A)*(Ad-I) Bc
    Ax = np.pad(sys_Ad, ((0,1),(0,1)), mode='constant', constant_values=0.)


    ######################
    # Reference config
    ######################
    dt = dt_d #0.002
    duration = 10. 
    t = np.arange(0,duration, dt)
    
    freq1 = np.array([0.1, 0.25, 0.55])
    amp1 = 0.1/(freq1*2*np.pi)
    ref, refdot = sumOfSines(t, freq1, amp1)




    ######################
    # Simulation Loop
    ######################
    # u = np.stack((u1, u2), 0)
    u = np.zeros((nA, len(t)))
    q = np.zeros((nXXd,len(t)))
    
    gains = np.zeros((nA, nS, len(t)))
    
    Ae = np.copy(Ax) #(nS,nS)
    Be = np.concatenate(
                (
                    -sys_Bd, 
                    np.zeros((1, sys_Bd.shape[1]))
                ), axis=0) #(nS,nA)
    
    for k in range(len(t[:-1])):
        
        # At each planning time, we form the affine version of Ad, Bd to achieve xd, vd:
        
        r0, rdot0 = ref[k], refdot[k]
        r_vec = np.array([r0]*nX +[rdot0]*nX +[1.])
        e_k = r_vec - np.append(q[:,k], 0)
        
        # if rdot0:
        #     R_mat = 
        if k% int(replanning_period/dt)==0:
            i_Pti = 0 #index to the current Pti (to compute gain)
            # Plan controller gain
            
            # Form A,B in error coordinates
            Ar = np.zeros_like(Ae)
            Ar[:nX, 0] = 1.; Ar[nX:nXXd, nXXd-1] = 1.; Ar[nXXd, nXXd] = 1.
            Ar[:nX, nXXd-1] = dt_d
            
            AeV = mat_mul_series(Ar-Ax, r_vec) 
            
            Ae[:,nXXd] = AeV
            
        
        
            # Compute all P_k backwards from the horizon, then calculate u_k
            # P_{k-1} = A^T P_k A -(A^T P_k B +N) (R+B^T P_k B)^{-1} (B^T P_k A +N^T) +Q
            # terminal condition: P_N = Q
    
            # Takes 0.025s to plan for 500 steps
            Pti = Q_mat
            Pvec = []
            for ti in range(nsteps, 0, -1):
                Pti = mat_mul_series(Ae.T, Pti, Ae) - \
                    mat_mul_series(
                        mat_mul_series(Ae.T, Pti, Be) +N_mat,
                        la.pinv2(
                            (R_mat + mat_mul_series(Be.T, Pti, Be))
                         ),
                        mat_mul_series( Be.T, Pti, Ae) +N_mat.T
                        ) +\
                    Q_mat
                Pvec.append(Pti)
            Pvec.reverse()
    
    
        # F_k = (R +B^T P_{k+1} B)^{-1} (B^T P_{k+1} A +N^T)
        # x_k = [q- [xd,vd], [1] ]
        # u_k = -F_k x_k        
        Pti = Pvec[i_Pti]
        Kti = mat_mul_series(
            la.pinv2(R_mat +\
                   mat_mul_series(Be.T, Pti, Be)
                   ),
            mat_mul_series(Be.T, Pti, Ae) +N_mat.T
            )
    
        U_opt = -mat_mul_series(Kti, e_k)
        
        i_Pti +=1
        
        gains[...,k] = Kti
        u[:,k] = U_opt #Kp*e + Kd*edot
        # u[1,k] = 0. #@@@@
        q[:,k+1] = RK4Step(q[:,k], sys_MSMSM, dt, u=u[:,k], t=t)
    
    x1 = q[X1i,:]
    xm = q[XMi,:]
    f1 = u[0,:]
    # f2 = u[1,:]
    fn1 = k1*(xm-x1)
    # fn2 = -k2*(xm-x2)
    e_sig = ref-xm
    
    err_loss = -dt_d* np.sum(e_sig**2)/(duration*max(amp1))
    eff_loss = -dt_d* np.sum(fn1**2)/(duration*max(amp1))
    
    return [x1, xm, f1, fn1, e_sig], gains, [err_loss, eff_loss]


##############
# Main
##############


t0 = time.time()
optimal_runs = []
Q_space = list(product(np.logspace(-1,3,10), repeat=2))
len_Qspace = len(Q_space)
for i, (qe, qed) in enumerate(Q_space):
    if i%5==1:
        tmp_t = time.time()-t0
        print('Processing %d th combination from %d . ETA: %d seconds'
              %(i, len_Qspace, (len_Qspace/i -1.)*tmp_t))
    s_ts, Kti, loss = get_opt_data([qe, qed])
    optimal_runs.append(
        Run_Data(s_ts, Kti, [qe, qed], loss) )

with open('optimal_runs_dt5ms_p1_pinv2_loss.pickle','wb') as f:
    pickle.dump(optimal_runs, f)