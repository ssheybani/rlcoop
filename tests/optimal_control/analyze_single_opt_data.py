# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 16:59:55 2021

@author: Saber
"""

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import time
from helper_funcs import sumOfSines, RK4Step, mat_mul_series, Run_Data
import pickle
import seaborn as sns

with open('optimal_runs_dt5ms_p1_pinv2_loss.pickle', 'rb') as f:
    optimal_runs = pickle.load(f)
    
    
######################
# Reference config
######################
dt_d = 0.005
dt = dt_d #0.002
duration = 10. 
t = np.arange(0,duration, dt)

freq1 = np.array([0.1, 0.25, 0.55])
amp1 = 0.1/(freq1*2*np.pi)
ref, refdot = sumOfSines(t, freq1, amp1)


n_runs = len(optimal_runs)
n_row, n_col = int(np.sqrt(n_runs)), int(np.sqrt(n_runs))


fig, ax = plt.subplots(n_row,n_col, figsize=(2*n_row,2*n_col), sharex=True)
for i in range(n_row):
    for j in range(n_col):
        run_num = i*n_col+j
        run_tt = optimal_runs[run_num]
        x1, xm, f1, fn1, e_sig = run_tt.signals
        ax[i,j].plot(t, x1, label = r'x_1')
        # plt.plot(t, x2, label = r'x_2')
        ax[i,j].plot(t, xm, label = r'x_m')
        ax[i,j].plot(t, ref, 'k--', label = r'ref')
        ax[i,j].plot(t, e_sig, 'c:', label='e')
        if j==0:
            ax[i,j].set_ylabel('qe/r= %.1f' %run_tt.e_costs[0])
        if i==(n_row-1):
            ax[i,j].set_xlabel('qed/r= %.1f' %run_tt.e_costs[1])
        
ax[n_row-1,n_col-1].legend()
plt.savefig('tracking_traj.png', dpi=300)


fig2, ax2 = plt.subplots(n_row,n_col, figsize=(2*n_row,2*n_col), sharex=True)
for i in range(n_row):
    for j in range(n_col):
        run_num = i*n_col+j
        run_tt = optimal_runs[run_num]
        x1, xm, f1, fn1, e_sig = run_tt.signals
        ax2[i,j].plot(t, f1, 'r', label = 'applied force')
        ax2[i,j].plot(t, -fn1, 'g--', label = '(inverted) normal force')
        ax2[i,j].plot(t, fn1+f1, 'b:', label = 'diff normal and applied')
        ax2[i,j].axhline(0, lw=0.3, color='k')
        if j==0:
            ax2[i,j].set_ylabel('qe/r= %.1f' %run_tt.e_costs[0])
        if i==(n_row-1):
            ax2[i,j].set_xlabel('qed/r= %.1f' %run_tt.e_costs[1])
ax2[n_row-1,n_col-1].legend()



k_ticks = ['k_e1', 'k_em', 'k_ed1', 'k_edm', 'kc', 'err', 'eff']
bar_colors = [plt.cm.tab10(0)]*5+[plt.cm.tab10(1), plt.cm.tab10(2)]
y_pos = np.arange(len(k_ticks))

fig3, ax3 = plt.subplots(n_row,n_col, figsize=(2*n_row,2*n_col), sharex=True)
for i in range(n_row):
    for j in range(n_col):
        run_num = i*n_col+j
        run_tt = optimal_runs[run_num]
        gain_vec = np.mean(run_tt.controllers, axis=-1).squeeze()
        losses = run_tt.loss
        losses = [50*losses[0],20*losses[1]]
        gain_std = np.std(run_tt.controllers, axis=-1).squeeze()
        # sns.barplot(data=gain_vec, ci=None, ax=ax3[i,j])
        ax3[i,j].bar(y_pos, list(gain_vec)+losses, color=bar_colors, align='center')
        # ax3[i,j].set_xticks(y_pos)#, k_ticks)
        # ax3[i,j].set_xticklabels(k_ticks)
        if j==0:
            ax3[i,j].set_ylabel('qe/r= %.1f' %run_tt.e_costs[0])
        if i==(n_row-1):
            ax3[i,j].set_xlabel('qed/r= %.1f' %run_tt.e_costs[1])
        ax3[i,j].set_ylim([-30,15])
    
plt.savefig('controllers_scaled.png', dpi=300)
# ax3[n_row-1,n_col-1].legend()

######################
# Plot the data    
######################

# plt.figure(1)
# plt.plot(t, x1, label = r'x_1')
# # plt.plot(t, x2, label = r'x_2')
# plt.plot(t, xm, label = r'x_m')
# plt.plot(t, ref, 'k--', label = r'ref')
# plt.plot(t, e_sig, 'c:', label='e')
# plt.legend()

# fig2, ax2 = plt.subplots(1,1, sharey= True, sharex = True)

# ax2.legend()

# ax2[1].plot(t, f2, 'r')
# ax2[1].plot(t, -fn2, 'g--')
# ax2[1].plot(t, fn2+f2, 'b:')
# # ax2[1].set_ylim([-3,3])
# ax2[1].axhline(0, lw=0.3, color='k')

# fig3, ax3 = plt.subplots(2,1, sharey= True, sharex = True)
# ax3[0].plot(t, f1, 'r--', label = 'applied force')
# ax3[0].plot(t, gains[0, 2], 'g', label = 'p gain')
# ax3[0].plot(t, gains[0, 5], 'b', label = 'd gain')
# # ax3[0].plot(t, -fn1, 'g', label = '(inverted) normal force')
# # ax3[0].plot(t, fn1+f1, 'b:', label = 'diff normal and applied')
# ax3[0].axhline(0, lw=0.3, color='k')
# ax3[0].legend()

# ax3[1].plot(t, f2, 'r--', label = 'applied force')
# ax3[1].plot(t, gains[1, 2], 'g', label = 'p gain')
# ax3[1].plot(t, gains[1, 5], 'b', label = 'd gain')
# ax3[1].axhline(0, lw=0.3, color='k')
# ax3[1].legend()

