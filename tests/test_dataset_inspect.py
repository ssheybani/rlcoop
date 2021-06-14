# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 12:01:15 2021

@author: Saber
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from pathlib import Path

import context
from context import rlcoop, DATA_PATH, CONFIG_PATH
# from rlcoop.algos import torch_trainer
from rlcoop.envs import env_msd
# from rlcoop.util import buffers #, nn_models 
# from rlcoop.agents import rl_agent, benchmark_agents, train_agents
from rlcoop.envs import gen_env_data


"""Load the data"""

dataset_name = 'ds_run_12.00.49'
dataset_dir = DATA_PATH + dataset_name

ep_trials = []
for filename in os.listdir(dataset_dir):
    with open(dataset_dir+'/'+filename, 'rb') as f:
        file_eps = pickle.load(f)
        ep_trials += file_eps
    


"""Plot the data"""
ts_dict = {'t':0,
            'e1':1, 'e1dot':2, 'fn1':3, 'act1':4, 'f1':5, 
            'e2':6, 'e2dot':7, 'fn2':8, 'act2':9,'f2':10}
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