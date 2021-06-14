# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 12:49:08 2021

@author: Saber
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import importlib
import argparse
import functools
import sys,os


import context
from context import rlcoop, DATA_PATH, CONFIG_PATH
# from rlcoop.algos import torch_trainer
from rlcoop.envs import env_msd
# from rlcoop.util import buffers #, nn_models 
# from rlcoop.agents import rl_agent, benchmark_agents, train_agents
from rlcoop.envs import gen_env_data

"""
200k episodes
Each episode: 10s 
Each 200 episodes become one csv file.

for i in range(1000)
    Generate 200 episodes: For j in range(200)
        Generate 1 episode: while env.done is False
        Dump the episode data (400 rows) in a temp variable.
    Dump the 200 episode data into a csv file. (print ETA)

 A run file has the following configs: action refractory period, 
 metacontroller2controller.

"""

ts_dict = {'t':0,
           'e1':1, 'e1dot':2, 'fn1':3, 'act1':4, 'f1':5, 
           'e2':6, 'e2dot':7, 'fn2':8, 'act2':9,'f2':10}



# def gen_episode(env, action2k, refresh_k_steps, renew_traj=True):
    
#     # Unwrap some variables
#     w_inv, b_inv = action2k
#     sdict = env.svec_dict
#     fdict = env.fvec_dict
#     X1i, X2i, XMi = sdict['x1'],sdict['x2'],sdict['xm']
#     dX1i, dX2i, dXMi = sdict['dx1'],sdict['dx2'],sdict['dxm']
#     Fn1i, Fn2i, F1i, F2i, = fdict['fn1'], fdict['fn2'], fdict['f1'], fdict['f2']
    
#     (rrdot, q, fvec) = env.reset(renew_traj=renew_traj, max_freq='max')    
#     ts_data = []
    
#     while env.get_time() < env._max_episode_steps-1:
#         t = env.get_time()
        
#         # Observe
#         r0, rdot0 = rrdot
#         r_vec = np.array([r0, r0, rdot0, rdot0])
#         x1_view = np.array([q[X1i], q[XMi], q[dX1i], q[dXMi]])
#         x2_view = np.array([q[X2i], q[XMi], q[dX2i], q[dXMi]])
#         e1_vec = r_vec - x1_view
#         e2_vec = r_vec - x2_view
        
#         # Get forces
#         if t%refresh_k_steps==0:
#             actions = np.random.uniform(low=-2, high=2, size=2).reshape(2,1)
#             K_fb = actions*w_inv +b_inv
#         f1 = - np.dot(K_fb[0,:], e1_vec)
#         f2 = - np.dot(K_fb[1,:], e2_vec)
        
#         # Collect the features for recording
#         e1,e1d = e1_vec[1], e1_vec[3]
#         e2,e2d = e2_vec[1], e2_vec[3]
#         fn1, fn2 = fvec[Fn1i], fvec[Fn2i]

#         ts_data.append([t*env.tstep, 
#                         e1, e1d, fn1, actions[0], f1, 
#                         e2, e2d, fn2, actions[1], f2])
#         # Environment step
#         (rrdot, q, fvec), _, _, _ = env.step([f1, f2])

#     return ts_data



    
seed = 1234
max_freq = 0.5
dt_d = 0.02
env1 = env_msd.TrackMSDMDSM_minimal(config_file=CONFIG_PATH+'/env_v4_config.ini',
                            tstep=dt_d, dyadic=True, seed_=seed, max_freq=max_freq)
with open('K2controller.pickle', 'rb') as f:
    w_inv, b_inv = pickle.load(f)
    
    w_inv = w_inv.reshape(1,4)
    b_inv = b_inv.reshape(1,4)
refresh_k_steps = 0.3/env1.tstep
test_data = gen_env_data.gen_episode(env1, (w_inv, b_inv), refresh_k_steps, renew_traj=True)
test_data = np.array(test_data, dtype=np.float)

# plt.plot(env1.traj[0,:-1], label='ref')
# plt.plot(env1.traj[0,:-1]-test_data[:, 1], label='cursor')
# plt.legend()


"""Plot the data"""
t = env1.traj_time[:-1]
ref = env1.traj[0,:-1]
cursor = env1.traj[0,:-1]-test_data[:, ts_dict['e1']]
f1 = test_data[:, ts_dict['f1']]
f2 = test_data[:, ts_dict['f2']]
fn1 = test_data[:, ts_dict['fn1']]
fn2 = test_data[:, ts_dict['fn2']]
action1 = test_data[:, ts_dict['act1']]
action2 = test_data[:, ts_dict['act2']]

plt.figure(1)
plt.plot(t, cursor, label = r'x_m')
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
ax3[0].plot(t, action1, 'g', label = 'metacontroller')
# ax3[0].plot(t, -fn1, 'g', label = '(inverted) normal force')
# ax3[0].plot(t, fn1+f1, 'b:', label = 'diff normal and applied')
ax3[0].axhline(0, lw=0.3, color='k')
ax3[0].legend()

ax3[1].plot(t, f2, 'r--', label = 'applied force')
ax3[1].plot(t, action2, 'g', label = 'metacontroller')
ax3[1].axhline(0, lw=0.3, color='k')
ax3[1].legend()
