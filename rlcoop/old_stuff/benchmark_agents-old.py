# Benchmark functions
# Originally written for agents from class DQNAgent
import numpy as np
import time

def policy_ts(env, agent1, agent2, duration=50):
    # Behavioral profile of the agents
    # Returns time series of events in an episode
    # Arguments:
    # normalize: if True, the reward is normalized by the number of time steps.
    eps=0; verbose=True
    
#         ep_loss = torch.zeros(env.observation_space.shape[0])
    observations = env.reset(renew_traj=True);# 
    agent1.reset(); agent2.reset()
    
    old_observations = np.asarray(observations)
    
    state1_old = agent1.observe(old_observations); state2_old = agent2.observe(old_observations)
    
    f1, action1 = agent1.get_force(state1_old, eps=eps, verbose=verbose)
    f2, action2 = agent2.get_force(state2_old, eps=eps, verbose=verbose)
#         ftr_vec = observations+[f1, f2]
    
    t_ts=[]
    r_ts, x_ts = [],[]
    f1_ts, f2_ts = [],[]
    u1_ts,u2_ts = [],[];  cum_reward = 0.
#     q10_ts, q11_ts, q20_ts, q21_ts = [],[],[],[]
    
    while True:
        t = env.get_time()
        observations, reward, done, _ = env.step([f1, f2])
        
        t_ts.append(t)
        r_ts.append(observations[0]); x_ts.append(observations[2]) # Logging
        f1_ts.append(f1) # Logging; 
        f2_ts.append(f2) # Logging
    
        cum_reward += reward#reward
        u1_ts.append(agent1.compute_utility(reward, f1))
        u2_ts.append(agent2.compute_utility(reward, f2))#reward
        
        state1 = agent1.observe(observations); state2 = agent2.observe(observations)
        f1, action1 = agent1.get_force(state1, eps=eps, verbose=verbose)
        f2, action2 = agent2.get_force(state2, eps=eps, verbose=verbose)

        if done is True:
            break

    cum_reward = cum_reward /t
    a1_cum_reward = np.mean(u1_ts)
    a2_cum_reward = np.mean(u2_ts)
    
    t_ts = [t_i*env.tstep for t_i in t_ts]
    return t_ts, (r_ts,x_ts), (f1_ts, f2_ts), (u1_ts, u2_ts, cum_reward, a1_cum_reward, a2_cum_reward)

#     return t_ts, (r_ts,x_ts), (f1_ts, f2_ts), (q10_ts, q11_ts, q20_ts, q21_ts), (u1_ts, u2_ts, cum_reward, a1_cum_reward, a2_cum_reward)

def single_eval(env, agent1, agent2, n_episodes=100, normalizer=True):
    # Empirical Model Evaluation
    # For assessing the performance of two RLAgent agents playing 
    # on DyadSlider.
    # Arguments:
    # n_episodes: the number of episodes used for averaging the quality of the policy
    # normalize: if True, the reward is normalized by the number of time steps.

    reward_vals = np.zeros(n_episodes)


    for i_episode in range(n_episodes):
        cum_reward = 0.
        observations = env.reset(renew_traj=True);
        agent1.reset(); agent2.reset()
        state1_old = agent1.observe(observations)#, agent2.observe(old_observations)
        f1 = agent1.get_force(state1_old, eps=0)
        f2 = 0. #agent2.get_force(observations, eps=0)
#         ftr_vec = observations+[f1, f2]

        while True:
            t = env.get_time()
            observations, reward, done, _ = env.step([f1, f2]) 
            cum_reward += reward
            
            state1 = agent1.observe(observations)
            
            f1 = agent1.get_force(state1, eps=0)
            f2 = 0. #agent2.get_force(observations, eps=0)
            
            if done is True:
                break
        if normalizer is True:
            reward_vals[i_episode] = cum_reward /t
        else:
            reward_vals[i_episode] = cum_reward
        
    return np.mean(reward_vals, axis=0)

def dyad_eval(env, agent1, agent2, n_episodes=100, normalizer=True):
    # Empirical Model Evaluation
    # For assessing the performance of two RLAgent agents playing 
    # on DyadSlider.
    # Arguments:
    # n_episodes: the number of episodes used for averaging the quality of the policy
    # normalize: if True, the reward is normalized by the number of time steps.

    reward_vals = np.zeros(n_episodes)


    for i_episode in range(n_episodes):
        cum_reward = 0.
        observations = env.reset(renew_traj=True);
        agent1.reset(); agent2.reset()
        state1_old = agent1.observe(observations); state2_old = agent2.observe(observations)
        f1 = agent1.get_force(state1_old, eps=0)
        f2 = agent2.get_force(state2_old, eps=0)
#         ftr_vec = observations+[f1, f2]

        while True:
            t = env.get_time()
            observations, reward, done, _ = env.step([f1, f2]) 
            cum_reward += reward
            
            state1 = agent1.observe(observations)
            state2 = agent2.observe(observations)
            
            f1 = agent1.get_force(state1, eps=0)
            f2 = agent2.get_force(state2, eps=0)
            
            if done is True:
                break
        if normalizer is True:
            reward_vals[i_episode] = cum_reward /t
        else:
            reward_vals[i_episode] = cum_reward
        
    return np.mean(reward_vals, axis=0)
    
    
def benchmark(algo, hp, env, agent1, agent2, xaxis_params):
    # Creates time series for algorithm quality across episodes
    
#     dyad_eval = dyad_eval
    
    t0 = time.time()
    
    # Unzip arguments
    n_episodes, n_intervals, n_eval = xaxis_params
    int_episodes=int(n_episodes/n_intervals)
    
    x, y = np.zeros(n_intervals+1), np.zeros(n_intervals+1)

    x[0] = 0
    y[0] = dyad_eval(env, agent1, agent2, n_episodes=1, normalizer=True)
    
    # Evaluate the created policy once every int_episodes episodes
    for i in range(n_intervals):
        print('Episode ', i*int_episodes, ': Objective (normalized by duration)= ', y[i])
        
        for j in range(int_episodes):
            agent1, agent2 = algo(env, agent1, agent2)
            
            if hp.target_int <= 1:
                agent1.update_target_qnet()
                agent2.update_target_qnet()
            elif i*int_episodes+j % hp.target_int == 0:
                agent1.update_target_qnet()
            elif i*int_episodes+j % hp.target_int == int(hp.target_int/2):
                agent2.update_target_qnet()
            
        x[i+1] = (i+1)*int_episodes
        y[i+1] = dyad_eval(env, agent1, agent2, n_episodes=n_eval, normalizer=True)
        
        ct = time.time()
        estimated_time_left = (n_intervals-i)*(ct-t0)/(i+1)
        print('Estimated time left: ', estimated_time_left)
        
    print('Total Duration: ', (ct- t0)/60, ' Minutes')
    return x,y, agent1, agent2