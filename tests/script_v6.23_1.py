# Date: Jan 26

# Using force as a control feature for the PD agents

import numpy as np
import pickle
import importlib
import argparse
import functools


import context
from context import rlcoop
from rlcoop.algos import torch_trainer
from rlcoop.envs import env_track_dyadic
from rlcoop.util import buffers, nn_models 
from rlcoop.agents import rl_agent, benchmark_agents, train_agents

import random

import sys, time, datetime
from copy import deepcopy
import torch
from torch import nn, optim


from fixed_exp_constants import * 

#---------- Parse the arguments
parser = argparse.ArgumentParser(description='RL on TrackSlider Experiment')

# Add the arguments
parser.add_argument('-cp',
                       type=float,
                       help='Cost coefficient for positive forces (pushing). Suggestions: 0.1, 1, 10.')

parser.add_argument('-cn',
                       type=float,
                       help='Cost coefficient for negative forces (pulling). Suggestions: 0.1, 1, 10.')

parser.add_argument('--n_rec_actor',
                       type=int,
                       default=0,
                       help='The number of layers from the actor network to log.')

parser.add_argument('--data_subdir',
                       type=str,
                       default='',
                       help='The subdirectory to save data in.')

parser.add_argument('--exp_name',
                       type=str,
                       default=None,
                       help='The experiment name to use in the name of the saved files.')


parser.add_argument('--n_ep',
                       type=int,
                       default=10000,
                       help='The number of episodes to train the agent.')
#----------


# Execute the parse_args() method
args = parser.parse_args()

c_positivef1 = args.cp
c_negativef1 = args.cn
cost_token = 'cp'+str(c_positivef1)+'cn'+str(c_negativef1)
n_rec_actor = args.n_rec_actor
rec_points = [1, 200, 500, 1000, 1500, 2000, 3000, 5000, 10000]
exp_name = args.exp_name
if exp_name is None:
    exp_name = parser.prog
data_subdir = args.data_subdir+'cp'+str(c_positivef1)+'cn'+str(c_negativef1)+'/'
# data_subsubdir = 
n_episodes = args.n_ep

seed = 1234
max_freq, max_freq_dyadic = 0.5, 0.3
env_v3 = env_track_dyadic.PhysicalTrackDyad_v3(seed_=seed,
                                               config_file=parent_path+'configs/env_v3_config.ini',
                                     max_freq=max_freq_dyadic)
env = env_v3#env_dyadic_wa #env_single_spring_wa

device = torch.device("cpu")

# For the algo
n_futuresteps = 8#20
gamma=0.7
trust_region = 0.5
tau_eint = 0.5; alpha_eint = env.tstep/(env.tstep+tau_eint)
lr = 3e-4
wd = 3e-5

loss_coef = 100
batch_size= 256 #1024 #

agent1cls = rl_agent.PPO5PID2Agent

a_ftr_dict = agent1cls.ftr_pos_dict
#For the actor network
ac_unmasked_ftrs = [a_ftr_dict['e'], a_ftr_dict['e\''], a_ftr_dict['fn']]
ac_ftr_normalizer = 7*np.asarray([2., 1, 0.02]) #based on std of the dummy agent
ac_disc_bounds = [-3., -1., -0.3, -0.1, 0., 0.1, 0.3, 1., 3.]

# ac_masked_ftrs=[a_ftr_dict['r'], a_ftr_dict['r\''], a_ftr_dict['r\"'],
#                 a_ftr_dict['e_int'], a_ftr_dict['none'], a_ftr_dict['f_own']
# ac_ftr_normalizer = 7*np.asarray([1, 1, 0.5, 2, 1, 0.5, 1, 0, 1]) #based on std of the dummy agent

#[-3.6, -2.8, -2.1, -1.5, -1., -0.6, -0.3, -0.1, 0, 0.1, 0.3, 0.6, 1., 1.5, 2.1, 2.8, 3.6]
#[-0.75, -0.25, 0., 0.25, 0.75]

k2 = 0.0#2
k3 = 0.0#05

kernel3d = [
    [
    [0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0.],
    [0., 0., k3, 0., 0.],
    [0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0.],
    ],
    [
    [0., 0., 0., 0., 0.],
    [0., 0., k3, 0., 0.],
    [0., k3, k2, k3, 0.],
    [0., 0., k3, 0., 0.],
    [0., 0., 0., 0., 0.],
    ],
    [
    [0., 0., k3, 0., 0.],
    [0., k3, k2, k3, 0.],
    [k3, k2, 1., k2, k3],
    [0., k3, k2, k3, 0.],
    [0., 0., k3, 0., 0.],
    ],
    [
    [0., 0., 0., 0., 0.],
    [0., 0., k3, 0., 0.],
    [0., k3, k2, k3, 0.],
    [0., 0., k3, 0., 0.],
    [0., 0., 0., 0., 0.],
    ],
    [
    [0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0.],
    [0., 0., k3, 0., 0.],
    [0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0.],
    ]
]

kernel3d = np.asarray(kernel3d)/np.sum(kernel3d)

n_hidden_l=None #256
nn_init_mu, nn_init_std = 0.3, 0.1
nn_out_std=0.1

#For the linear controller terms

ctrl_ftrs = [a_ftr_dict['e'], a_ftr_dict['e\''], a_ftr_dict['e_int'] ]#, a_ftr_dict['fn'] ]
pid_scalers = 20*np.asarray([1.,  0.1, 0.1])#, -0.02])
nout = len(ctrl_ftrs)
# torch.tensor([1.,  0.1, 0.1, 0.02], device=device, dtype=torch.float32)
# #[1.,  0.05, 0.025, -0.02])
#     [6.,  0.3, 0.15, 0.02*6]) # will be exponentiated inside the agent._gain2cmd method
def pid_interp_te(te, scalers=None):
    xtt = te>1; f_xtt = te+20
    ytt = te<0; f_ytt = te+1
    ztt = (te<=1) * (te>=0); f_ztt = 20*te+1
    te_c = f_xtt*xtt + f_ytt*ytt + f_ztt*ztt
    
    if scalers is not None:
        return scalers *te_c
    return te_c

pid_interp = functools.partial(pid_interp_te, scalers=pid_scalers)
ac_masked_ftrs = list(set([i for i in range(9)]) - set(ac_unmasked_ftrs) )
# np.asarray([6.,  0.6, 0.3]) 
#np.asarray([2.30258509,  0.11512925, -0.05756463])
#100*np.asarray([1., 0.1, 0.01], dtype=np.float32)
#2*np.asarray([1., 0.05, -0.025],dtype=np.float32)
# pd_scalers = [30., 30*0.025, -30*0.025]
# pd_scalers = 2*np.asarray([1., 0.05, -0.025],dtype=np.float32)

#[-0.5000, -0.1000,  0.1000,  0.5000]
#5*np.asarray([1., 1., 0.1, 3., 1., 0.1, 1., 1., 1.])

# spring_init_k = .01


#------- Create the NN models
# Input:
# r, r', x, x', fn, fndot, f
# Output: q

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


try_num=1; 
while try_num<=3:
    clean_exit= True
    print('Try ', try_num)
    
    n_features = 9 
    actor_net1 = nn_models.NetContPID_OhSoft(n_features, nout, kernel3d, n_hl=n_hidden_l,
                init_mu=nn_init_mu, init_std=nn_init_std, out_std=nn_out_std,
                 masked_ftrs=ac_masked_ftrs, ftr_normalizer=ac_ftr_normalizer,
                disc_bounds=ac_disc_bounds, device=device).to(device)

    actor1_opt = optim.Adam([
            {'params': actor_net1.parameters(), 'lr': lr},
        ], lr=3e-5, weight_decay=wd, amsgrad=True)
#     actor1_opt = optim.Adam([
#             {'params': actor_net1.fc1.parameters(), 'lr': 1e-6},
#             {'params': actor_net1.fc_mu.parameters(), 'lr': 1e-5}
#         ], lr=3e-5, amsgrad=True)
#             {'params': actor_net1.fc15.parameters(), 'lr': 1e-4},


    actor_net1.eval(); #Put it to evaluation mode
    
    n_rec_critic = 0
    
    actor_net2 = deepcopy(actor_net1);
    actor_net2.eval(); #Put it to evaluation mode
    
    scheduler1 = None #optim.lr_scheduler.StepLR(optimizer1, step_size=100, gamma=0.8)
    
    actor2_opt = optim.Adam(actor_net2.parameters(), lr=lr, weight_decay=wd, amsgrad=True)

    nn_mod1 = torch_trainer.MCPPOTrainer(
        actor_net1, actor1_opt, 
        n_actions=nout, trust_region=trust_region, nsteps=n_futuresteps, 
        scheduler=scheduler1, device=device, gamma=gamma,
        logger=logger1, log_weights=(n_rec_actor,n_rec_critic) ,f_ftr_idx=-1, loss_coef=loss_coef)
    
    nn_mod2 = torch_trainer.MCPPOTrainer(
        actor_net2, actor2_opt, 
        n_actions=nout, trust_region=trust_region, nsteps=n_futuresteps, 
        scheduler=None, device=device, gamma=gamma,
        logger=logger1, log_weights=(n_rec_actor,n_rec_critic) ,f_ftr_idx=-1, loss_coef=loss_coef)


#     nn_mod1._log_weights = nn_mod1._log_params #@@@@@@@@@@@ Monitor parameters
    
    #---------------------------------
    # # Load a single pretrained agent
#     pretrained_fname1 = 'exp3.94bc0.01_0.005Jan04-02-02-26results.p' #working controller

#     _, _, xagent1, _ = pickle.load(open("data/"+pretrained_fname1, "rb"))
#     critic1_statedict = deepcopy(xagent1.nn_mod.critic_net.state_dict())
#     actor1_statedict = deepcopy(xagent1.nn_mod.actor_net.state_dict())

#     nn_mod1.critic_net.load_state_dict(critic1_statedict)
#     nn_mod1.target_vnet.load_state_dict(deepcopy(critic1_statedict))
#     nn_mod1.actor_net.load_state_dict(actor1_statedict)
#     nn_mod1.target_pnet.load_state_dict(deepcopy(actor1_statedict))
    
#     spring_init_k = 0.01
    #---------------------------------


    # Run name and specs
    exp_name = exp_name 
    datinow = datetime.datetime.now();
    cost_token = cost_token 
    run_token = cost_token+'_'+datinow.strftime("%H.%M.%S") 

    target_int = 1
#     target_int_actor = 1
    update_interval = int(min(batch_size/4, (env.duration-1)/env.tstep)) 
    # 4 passes on each datapoint or batch on average. At least one per episode.
    
    
    c_effort2 = 0. #-.25
    c_negativef2 = None

    hyperparams = None
    # BMHyperparams(batch_size, learning_rate, 
                    buffer_max_size, experience_sift_tol, target_int, gamma)

    agent1 = agent1cls(
        nn_mod1, buffer1, muscle1, perspective=0, hyperparams=hyperparams, 
        c_error=1., c_positivef=c_positivef1, c_negativef=c_negativef1, 
        ctrl_ftrs=ctrl_ftrs, pid_interp=pid_interp, pid_scalers=pid_scalers, force_rms=1., alpha_eint=alpha_eint)
    agent2 = rl_agent.idle_agent(nn_mod2, buffer2, muscle2, perspective=1, c_error=1.,
                                 c_effort=c_effort2, c_negativef=c_negativef2, force_rms=1.) 
    nn_mod1.set_agent(agent1); #nn_mod2.set_agent(agent2)

    agent1.set_train_hyperparams(hyperparams)
    # agent2.set_train_hyperparams(hyperparams)

    algo = train_agents.train_single5
#     env.spring_k = spring_init_k





    #------------ Benchmark: 

    #algo, hp, env, agent1, agent2, xaxis_params
    hp = hyperparams
    t0 = time.time()

    # Unzip arguments
    _, _, n_eval = xaxis_params

    n_episodes = n_episodes; n_intervals = int(n_episodes/100)
    int_episodes=int(n_episodes/n_intervals)

    x, yp, yc1 = np.zeros(n_intervals+1), np.zeros(n_intervals+1), np.zeros(n_intervals+1)

    x[0] = 0
    yp[0], yc1[0] = benchmark_agents.single_eval(env, agent1, agent2, n_episodes=5, normalizer=True, ret_effort=True)

    # Evaluate the created policy once every int_episodes episodes
    for i in range(n_intervals):
        print('--------------------------------')
        print('Episode {0}: Error = {1:2.3f}, Effort = {2:2.3f}'.format(i*int_episodes, yp[i], yc1[i]))
#         print('Episode', i*int_episodes, ': Objectives = %.3f %.3f'%yp[i] %yc[i])
        
        curr_perf = yp[i]
        for j in range(int_episodes):
            agent1, agent2, curr_perf = algo(
                env, agent1, agent2, update_interval=update_interval, 
                current_performance=yp[i], ret_perf=True)

            if j%20==19:
                print('curr_perf = ', curr_perf)
            
            if hp.target_int <= 1:
                agent1.update_target_nets()
                agent2.update_target_nets()
            elif (i*int_episodes+j) % hp.target_int == 0:
                agent1.update_target_nets()
            elif (i*int_episodes+j) % hp.target_int == int(hp.target_int/2):
                agent2.update_target_nets()

#             if (i*int_episodes+j) %100 ==99:  # Reduce spring_k
#                 env.spring_k = max(env.spring_k-0.1, 0.01)
#                 print('env.spring_k = ', env.spring_k)
            if (i*int_episodes+j) in rec_points:
                x_s, y_s, agent1_s, agent2_s = deepcopy(x), [deepcopy(yp), deepcopy(yc1)], deepcopy(agent1), deepcopy(agent2)
                agent1_s.buffer = buffers.CyclicBuffer(1); agent2_s.buffer = buffers.CyclicBuffer(1);
                try:
                    os.makedirs('data/'+data_subdir)
                except FileExistsError:
                    pass
                fname = exp_name+run_token+'-'+'{:04d}'.format(i*int_episodes+j)
                pickle.dump((x_s, y_s, agent1_s, agent2_s), open( "data/"+data_subdir+fname, "wb" ) )
           

        x[i+1] = (i+1)*int_episodes
        yp[i+1], yc1[i+1] = benchmark_agents.single_eval(env, agent1, agent2, n_episodes=n_eval, normalizer=True, ret_effort=True)

        if i>=1:
            if yp[i+1]<-1. and yp[i]<-1. and yp[i-1]<-1.:
                print('The search has diverged too much. Restarting the search.')
                clean_exit = False
                break
                
        
        ct = time.time()
        estimated_time_left = (n_intervals-i)*(ct-t0)/((i+1)*60)
        print('Estimated time left: %.1f' %estimated_time_left, 'minutes')
    
    if clean_exit is False:
        try_num+=1
        continue
    else:
        break
        
    
print('Total Duration: ', (ct- t0)/60, ' Minutes')
if clean_exit is False:
    print('No successful runs')
else:
    agent1.buffer = buffers.CyclicBuffer(1)
    agent2.buffer = buffers.CyclicBuffer(1)
    try:
        os.makedirs('data/'+data_subdir)
    except FileExistsError:
        pass
    fname = exp_name+run_token+'-'+'{:04d}'.format(i*int_episodes+j)
    pickle.dump((x, [yp,yc1], agent1, agent2), open( "data/"+data_subdir+fname, "wb" ) )
    print('______________________________________')