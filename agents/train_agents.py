# Simulate training for both agents for one episode
# Originally written for agents of the class DQNAgent.

# from train_agents.py
# Simulate training for both agents for one episode
# Originally written for agents of the class DQNAgent.
import scipy
import numpy as np
import torch #for single4

    
    
def train_single5(env, agent1, agent2, update_interval=25, current_performance=-100., ret_perf=False):
    # calculate eps using current performance
    old_observations = env.reset(renew_traj=True, max_freq='random'); #old_observations = np.asarray(observations)
    agent1.reset(); agent2.reset()
    
#     eps = 1- scipy.special.expit(-16*np.log10(-current_performance)-12)
    eps=1.
            
    state1_old = agent1.observe(old_observations)#, agent2.observe(old_observations)
    f1, action_lp1 = agent1.get_force(state1_old, eps=eps, verbose=True)
    f2, action_lp2 = 0., 0. #agent2.get_force(state2_old, eps=1., verbose=True)

    reward_old = 0.
    
    cum_reward = 0.
    
    while True:
        t = env.get_time()
         
        observations, reward_new, done, _ = env.step([f1, f2]) #@@@@@@@@@
        cum_reward += reward_new
        
        state1 = agent1.observe(observations)
#             , \            agent2.observe(observations)
        
        # Add the experience
        if t>1: #because the first reward_old is incorrect
            agent1.add_experience((state1_old, action_lp1, state1, 
                                  torch.tensor([[reward_old, reward_new]], device=agent1.nn_mod.device, dtype=torch.float32)
                                  ))
    #             agent2.add_experience((state2_old, action2, state2, reward))
        state1_old = state1#, state2
        reward_old = reward_new

        f1, action_lp1 = agent1.get_force(state1, eps=eps, verbose=True)
            
#         else:
#             f1 = agent1.muscle.get_force(action1)
#             f2 = 0. #agent2.muscle.get_force(action2)
            
        # Take one training step for each agent
        if t%update_interval==0:
            agent1.train_step(current_performance)

        if done is True:
            break
    if ret_perf is True:
        return agent1, agent2, cum_reward /t
    else:
        return agent1, agent2
    


def train_dyad5(env, agent1, agent2, update_interval=25, current_performance=-100., ret_perf=False):
    
    eps = 1. #- scipy.special.expit(-16*np.log10(-current_performance)-12)
     
    old_observations = env.reset(renew_traj=True, max_freq='max'); #old_observations = np.asarray(observations)
    agent1.reset(); agent2.reset()
    
    state1_old = agent1.observe(old_observations); state2_old = agent2.observe(old_observations)
    f1, action_lp1 = agent1.get_force(state1_old, eps=eps, verbose=True)
    f2, action_lp2 = agent2.get_force(state2_old, eps=eps, verbose=True)
    
    reward_old=0.
    cum_rewards = 0.
    
    while True:
        t = env.get_time()
#         print('f1, f2', f1, f2) #@@@@@@@@@@
         
        observations, reward_new, done, _ = env.step([f1, f2]) #@@@@@@@@@
        cum_rewards +=reward_new
        
        
        state1 = agent1.observe(observations)
        state2 = agent2.observe(observations)
            
#             utility1 = agent1.compute_utility(reward, f1)
#             utility2 = agent2.compute_utility(reward, f2)
                    
        # Add the experience
#             err = observations[0]-observations[2]
#             if abs(err)> env.max_err*agent1.buffer.tag:
        if t>1:    
            agent1.add_experience((state1_old, action_lp1, state1, 
                                  torch.tensor([[reward_old, reward_new]], device=agent1.nn_mod.device, dtype=torch.float32)
                                  ))
            agent2.add_experience((state2_old, action_lp2, state2, 
                                  torch.tensor([[reward_old, reward_new]], device=agent2.nn_mod.device, dtype=torch.float32)
                                  ))
        state1_old = state1; state2_old = state2
        reward_old = reward_new

        f1, action_lp1 = agent1.get_force(state1, eps=eps, verbose=True)
        f2, action_lp2 = agent2.get_force(state2, eps=eps, verbose=True)
            
#         else:
#             f1 = agent1.muscle.get_force(action1)
#             f2 = agent2.muscle.get_force(action2)
            
        # Take one training step for each agent
        if t%update_interval==0:
            agent1.train_step(current_performance)
            agent2.train_step(current_performance)
            
        if done is True:
            break
    if ret_perf is True:
        return agent1, agent2, cum_rewards /t
    else:
        return agent1, agent2


