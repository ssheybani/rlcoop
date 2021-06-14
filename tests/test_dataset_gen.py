
import numpy as np
# import matplotlib.pyplot as plt
import pickle
from pathlib import Path

# import importlib
# import argparse
# import functools
# import sys,os
import time, datetime


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
Each 200 episodes become one pickle file.

for i in range(1000)
    Generate 200 episodes: For j in range(200)
        Generate 1 episode: while env.done is False
        Dump the episode data (400 rows) in a temp numpy array.
    Append each episode data to a list.
    Dump the list containing 200 episode data (numpy arrays) into a pickle. (print ETA)

 A run file has the following configs: action refractory period, 
 metacontroller2controller. how many episodes in each file, how many files.

"""

  
"""
Set the config
"""
seed = 1234
max_freq = 0.5
dt_d = 0.02
duration = 10

cont_refresh_time = 0.3

# run config
n_file_eps = 10
n_files = 10

"""
Create the objects
"""
env1 = env_msd.TrackMSDMDSM_minimal(config_file=CONFIG_PATH+'/env_v4_config.ini',
                            tstep=dt_d, duration=duration, dyadic=True, 
                            seed_=seed, max_freq=max_freq)
with open('K2controller.pickle', 'rb') as f:
    w_inv, b_inv = pickle.load(f)
    
    w_inv = w_inv.reshape(1,4)
    b_inv = b_inv.reshape(1,4)
refresh_k_steps = cont_refresh_time/env1.tstep

"""
Run loop
"""
datinow = datetime.datetime.now();
run_token = 'ds_run_'+datinow.strftime("%H.%M.%S")
dataset_path = DATA_PATH + run_token
Path(dataset_path).mkdir(parents=True, exist_ok=True)
t0 = time.time()
for i_files in range(n_files):
    if i_files%10==1:
        tmp_t = time.time()-t0
        print('Processing %d th file from %d . ETA: %d seconds'
              %(i_files, n_files, (n_files/i_files -1.)*tmp_t))
    file_eps = []
    for i_eps in range(n_file_eps):
        ep_data = gen_env_data.gen_episode(
            env1, (w_inv, b_inv), refresh_k_steps, renew_traj=True)
        ep_data = np.array(ep_data, dtype=np.float)
        file_eps.append(ep_data)
    with open(dataset_path+'/experience_chunk_'+str(i_files)+'.pickle','wb') as file_handle:
        pickle.dump(file_eps, file_handle)


# """Plot the data"""
# ts_dict = {'t':0,
#            'e1':1, 'e1dot':2, 'fn1':3, 'act1':4, 'f1':5, 
#            'e2':6, 'e2dot':7, 'fn2':8, 'act2':9,'f2':10}
# t = env1.traj_time[:-1]
# ref = env1.traj[0,:-1]
# cursor = env1.traj[0,:-1]-test_data[:, ts_dict['e1']]
# f1 = test_data[:, ts_dict['f1']]
# f2 = test_data[:, ts_dict['f2']]
# fn1 = test_data[:, ts_dict['fn1']]
# fn2 = test_data[:, ts_dict['fn2']]
# action1 = test_data[:, ts_dict['act1']]
# action2 = test_data[:, ts_dict['act2']]

# plt.figure(1)
# plt.plot(t, cursor, label = r'x_m')
# plt.plot(t, ref, 'k--', label = r'ref')
# # plt.plot(t, e_sig, 'c:', label='e')
# plt.legend()

# fig2, ax2 = plt.subplots(2,1, sharey= True, sharex = True)
# ax2[0].plot(t, f1, 'r', label = 'applied force')
# ax2[0].plot(t, -fn1, 'g--', label = '(inverted) normal force')
# ax2[0].plot(t, fn1+f1, 'b:', label = 'diff normal and applied')
# ax2[0].axhline(0, lw=0.3, color='k')
# ax2[0].legend()

# ax2[1].plot(t, f2, 'r')
# ax2[1].plot(t, -fn2, 'g--')
# ax2[1].plot(t, fn2+f2, 'b:')
# # ax2[1].set_ylim([-3,3])
# ax2[1].axhline(0, lw=0.3, color='k')

# fig3, ax3 = plt.subplots(2,1, sharey= True, sharex = True)
# ax3[0].plot(t, f1, 'r--', label = 'applied force')
# ax3[0].plot(t, action1, 'g', label = 'metacontroller')
# # ax3[0].plot(t, -fn1, 'g', label = '(inverted) normal force')
# # ax3[0].plot(t, fn1+f1, 'b:', label = 'diff normal and applied')
# ax3[0].axhline(0, lw=0.3, color='k')
# ax3[0].legend()

# ax3[1].plot(t, f2, 'r--', label = 'applied force')
# ax3[1].plot(t, action2, 'g', label = 'metacontroller')
# ax3[1].axhline(0, lw=0.3, color='k')
# ax3[1].legend()
