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
import pandas as pd

import context
from context import rlcoop, DATA_PATH, CONFIG_PATH
# from rlcoop.algos import torch_trainer
from rlcoop.envs import env_msd
# from rlcoop.util import buffers #, nn_models 
# from rlcoop.agents import rl_agent, benchmark_agents, train_agents
from rlcoop.envs import gen_env_data
from rlcoop.util import helper_funcs

"""Load the data"""

n_ep_to_load = 5000



dataset_name = 'ds_run_17.59.14'
dataset_dir = DATA_PATH + dataset_name

ep_trials = []
for filename in os.listdir(dataset_dir):
    if len(ep_trials)<n_ep_to_load:
        with open(dataset_dir+'/'+filename, 'rb') as f:
            file_eps = pickle.load(f)
            ep_trials += file_eps

# ep_trials = np.asarray(ep_trials)
ts_dict = {'t':0,
            'e1':1, 'e1dot':2, 'fn1':3, 'act1':4, 'f1':5, 
            'e2':6, 'e2dot':7, 'fn2':8, 'act2':9,'f2':10}


"""
Sample rollouts

Assuming that ep_trials contains the data that one would usse as a batch, 
create the rollouts in a form that the policy gradient algorithm with TD(lambda)
updates can use.

"""


