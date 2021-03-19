import gym
import sys, time
import numpy as np; 
from numpy import asarray as narr
import matplotlib.pyplot as plt
from collections import namedtuple
from copy import deepcopy
import torch
from torch import nn, optim


# Import custom scripts
sys.path.append('configs/')
sys.path.append('agents/')
sys.path.append('util/')
import env_old, env_reach, env_spring, env_track_dyadic
from util import buffers, nn_models, torch_trainer
from agents import base, rl_agent, train_agents, benchmark_agents


BMHyperparams = namedtuple('BMHyperparams',
                ('batch_size','learning_rate', 'buffer_max_size',
                 'experience_sift_tol', 'target_int', 'gamma'))
#Benchmark x-axis related
n_episodes = 200 #recommended: 150
n_intervals = 10 #15 # recommended: 15
n_eval = 20

# Buffer-related
experience_sift_tol = 0. #0.01
buffer_max_size = 300000# 1000 episodes

# Algo hyperparams
learning_rate=0.0001; momentum = 0.9
gamma = 0.9 #0.5 #0.2#0.1 #0.8 # Future reward discount rate
batch_size= 256 #1024 #
xaxis_params = (n_episodes, n_intervals, n_eval)


#------- Create the agents
buffer1 = buffers.CyclicBuffer(buffer_max_size)#, tag=experience_sift_tol)
#ReplayMemoryList(buffer_max_size, tag=experience_sift_tol) 
#
buffer2 = deepcopy(buffer1)

    #------- Create the NN models
# Input:
# r, r', x, x', fn, fndot, f
# Output: q
actor_net1 = nn_models.NetL1G(6) #NetRelu1L1G(6, n_hidden=128)# NetReGlReGlLG(6, n_hidden=128) #NetRelu3L1G(6, n_hidden=64) #NetL1G(6) #N
critic_net1 = nn_models.NetReGlReGlL(6,1) # NetGlu2L1(6,1) #nn_models.NetRelu3L1(6, 1)#nn_models.NetRelu1L1(6, 1)

actor_net2 = deepcopy(actor_net1); critic_net2 = deepcopy(critic_net1)

actor1_opt = optim.Adam(actor_net1.parameters(), lr=5e-5, amsgrad=True)
critic1_opt = optim.Adam(critic_net1.parameters(), lr=1e-4)

scheduler1 = None #optim.lr_scheduler.StepLR(optimizer1, step_size=100, gamma=0.8)

actor2_opt = optim.Adam(actor_net2.parameters(), lr=1e-4)
critic2_opt = optim.Adam(critic_net2.parameters(), lr=1e-4)

class Logger(): 
    def __init__(self): 
        self.entropy_ts = []
        self.adv_ts = []
        self.logprobs_ts = []
        self.actor_weight_ts = [[],[],[]]
        self.critic_weight_ts = [[],[],[]]
        self.eff_adv_ratio_ts = []
        self.actorloss_ts = []


# global_logger = Logger()
# global_logger.entropy_ts = []
# global_logger.adv_ts = []
# global_logger.logprobs_ts = []
# global_logger.actor_weight_ts = [[],[]]
# global_logger.critic_weight_ts = [[],[]]
# global_logger.eff_adv_ratio_ts = []
# global_logger.actorloss_ts = []

# global_logger2 = Logger()
# global_logger2.entropy_ts = []
# global_logger2.adv_ts = []
# global_logger2.logprobs_ts = []
# global_logger2.actor_weight_ts = [[],[]]
# global_logger2.critic_weight_ts = [[],[]]
# global_logger2.eff_adv_ratio_ts = []
# global_logger2.actorloss_ts = []

criterion = nn.MSELoss()
# nn_mod1 = torch_trainer.ACTrainer(actor_net1, critic_net1, actor1_opt, critic1_opt, criterion, scheduler=scheduler1, logger=global_logger) 
# nn_mod2 = torch_trainer.ACTrainer(actor_net2, critic_net2, actor2_opt, critic2_opt, criterion)


force_max=20.; #force_min=-20.
sigma = 0. #0.3;
tau = 0.09 #0.05 #
muscle1 = base.MuscleModel(sigma, force_max, ts=0.025, tau=tau)
muscle2 = deepcopy(muscle1)

seed = 1234
max_freq = 0.5 #0.2; 
max_freq_dyadic = 0.3
spring_init_k = .7
env_single_spring = env_spring.PhysicalTrackSpring(seed_=seed,
                                     max_freq=max_freq, 
                                     spring_k=spring_init_k) 

env_dyadic = env_track_dyadic.PhysicalTrackDyad(seed_=seed,
                                     max_freq=max_freq_dyadic)


env_single_spring_wa = env_spring.PhysicalTrackSpringWA(seed_=seed,
                                     max_freq=max_freq, 
                                     spring_k=spring_init_k) 

env_dyadic_wa = env_track_dyadic.PhysicalTrackDyadWA(seed_=seed,
                                     max_freq=max_freq_dyadic)

env_v3 = env_track_dyadic.PhysicalTrackDyad_v3(seed_=seed,
                                               config_file='configs/env_v3_config.ini',
                                     max_freq=max_freq_dyadic)
# env = env_reach.PhysicalReach(seed_=seed, max_freq=max_freq) #env_old.PhysicalDyads(seed_=seed, max_freq=max_freq)
#___________________________________________
