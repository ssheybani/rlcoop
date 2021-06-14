# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 17:35:19 2021

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


# Optimal controller for the following dynamical system, and Q and R function:
    
######################
# Sys config
######################

m1 = 1
m2 = 1
M = 5.#/2
k1 = 100.#1000 #For k>100, x1, xm remain indistinguishably close.
k2 = 100.
zeta = 1.
b = zeta * np.sqrt((M+m1)*8*k1)

# X1i, XMi, dX1i, dXMi = 0,1,2,3
# sys_MSMSM = lambda q, u, t: np.array([q[dX1i],
#                                       q[dXMi],
#                                       (1/m1)*( u[0] + k1*(q[XMi]-q[X1i]) + b*(q[dXMi]-q[dX1i])),
#                                       (1/M)*(k1*(q[X1i]-q[XMi]) + b*(q[dX1i]-q[dXMi]))])

# dyadic
X1i, X2i, XMi, dX1i, dX2i, dXMi = 0,1,2,3,4,5
sys_MSMSM = lambda q, u, t: np.array([q[dX1i],
                                      q[dX2i],
                                      q[dXMi],
                                      (1/m1)*( u[0] + k1*(q[XMi]-q[X1i]) + b*(q[dXMi]-q[dX1i])),
                                      (1/m2)*( u[1] + k2*(q[XMi]-q[X2i]) + b*(q[dXMi]-q[dX2i])),
                                      (1/M)*(k1*(q[X1i]-q[XMi]) + k2*(q[X2i]-q[XMi]) + b*(q[dX1i]-q[dXMi])+ b*(q[dX2i]-q[dXMi]))])


######################
# Reference config
######################
dt = 0.025 #dt_d #0.002
duration = 10. 
t = np.arange(0,duration, dt)

freq1 = np.array([0.1, 0.25, 0.55])
amp1 = 0.1/(freq1*2*np.pi)
ref, refdot = sumOfSines(t, freq1, amp1)
    
refresh_k_steps = int(0.3/dt)

with open('K2controller.pickle', 'rb') as f:
    w_inv, b_inv = pickle.load(f)
    
    w_inv = w_inv.reshape(1,4)
    b_inv = b_inv.reshape(1,4)
# Usage
    # Kti = K*w_inv +b_inv
    # r_vec = [r, r, r', r']
    # e_k = r_vec - [x1, xm, x1', xm']
    # U_opt = -mat_mul_series(Kti, e_k)
    
######################
# Simulation Loop
######################
# u = np.stack((u1, u2), 0)
u = np.zeros((2, len(t)))
q = np.zeros((6,len(t)))
k_both = np.zeros((2,len(t)))


for k in range(len(t[:-1])):
    r0, rdot0 = ref[k], refdot[k]
    r_vec = np.array([r0, r0, rdot0, rdot0])
    x1_view = np.array([q[X1i,k], q[XMi,k], q[dX1i,k], q[dXMi,k]])
    x2_view = np.array([q[X2i,k], q[XMi,k], q[dX2i,k], q[dXMi,k]])
    e1_k = r_vec - x1_view
    e2_k = r_vec - x2_view
    
    if k%refresh_k_steps==0:
        K_both = np.random.uniform(low=-2, high=2, size=2).reshape(2,1)
        K_fb = K_both*w_inv +b_inv
    u1 = - np.dot(K_fb[0,:], e1_k)
    u2 = - np.dot(K_fb[1,:], e2_k)
    u[:,k] = np.asarray([u1,u2])
    q[:,k+1] = RK4Step(q[:,k], sys_MSMSM, dt, u=u[:,k], t=t)
    k_both[:,k] = K_both.squeeze()

x1 = q[0,:]
x2 = q[1,:]
xm = q[2,:]
f1 = u[0,:]
f2 = u[1,:]
fn1 = k1*(xm-x1)
fn2 = -k2*(xm-x2)


######################
# Analysis
######################

plt.figure(1)
plt.plot(t, x1, label = r'x_1')
plt.plot(t, x2, label = r'x_2')
plt.plot(t, xm, label = r'x_m')
plt.plot(t, ref, 'k--', label = r'ref')
# plt.plot(t, e_sig, 'c:', label='e')
plt.legend()

fig2, ax2 = plt.subplots(2,1, sharey= True, sharex = True)
ax2[0].plot(t, f1, 'r', label = 'applied force')
ax2[0].plot(t, -fn1, 'g--', label = '(inverted) normal force')
ax2[0].plot(t, fn1+f1, 'b:', label = 'diff normal and applied')
ax2[0].axhline(0, lw=0.3, color='k')
ax2[0].legend()

ax2[1].plot(t, f2, 'r')
ax2[1].plot(t, -fn2, 'g--')
ax2[1].plot(t, fn2+f2, 'b:')
# ax2[1].set_ylim([-3,3])
ax2[1].axhline(0, lw=0.3, color='k')

fig3, ax3 = plt.subplots(2,1, sharey= True, sharex = True)
ax3[0].plot(t, f1, 'r--', label = 'applied force')
ax3[0].plot(t, k_both[0,:], 'g', label = 'metacontroller')
# ax3[0].plot(t, -fn1, 'g', label = '(inverted) normal force')
# ax3[0].plot(t, fn1+f1, 'b:', label = 'diff normal and applied')
ax3[0].axhline(0, lw=0.3, color='k')
ax3[0].legend()

ax3[1].plot(t, f2, 'r--', label = 'applied force')
ax3[1].plot(t, k_both[1,:], 'g', label = 'metacontroller')
ax3[1].axhline(0, lw=0.3, color='k')
ax3[1].legend()

plt.show()
