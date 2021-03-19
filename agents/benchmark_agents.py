# Benchmark functions
# Originally written for agents from class DQNAgent
import numpy as np
import time


def benchmark_force(env):
    # Ideal force ts and performance given the trajectory and environment
    
    observations = env.reset(renew_traj=False);# 
    r, rd, rdd, x, *unused = observations
    f1 = env.obj_mass*rdd + env.obj_fric * rd; f2=0
    
    t_ts=[]
    r_ts, x_ts = [],[]
    f1_ts, f2_ts = [],[]
    cum_reward = 0.
    
    while True:
        t = env.get_time()
        observations, reward, done, _ = env.step([f1, f2])
        
        r, rd, rdd, x, *unused = observations
        f1 = env.obj_mass*rdd + env.obj_fric * rd; f2=0
        
        t_ts.append(t)
        r_ts.append(r); x_ts.append(x) # Logging
        f1_ts.append(f1) # Logging; 
        f2_ts.append(f2) # Logging
    
        cum_reward += reward#reward

        if done is True:
            break

    cum_reward = cum_reward /t
    
    t_ts = [t_i*env.tstep for t_i in t_ts]
    return t_ts, (r_ts,x_ts), (f1_ts, f2_ts), (0, 0, cum_reward, 0, 0)



def policy_ts(env, agent1, agent2, renew_traj=True):
    # Behavioral profile of the agents
    # Returns time series of events in an episode
    # Arguments:
    # normalize: if True, the reward is normalized by the number of time steps.
    # if the feature vector includes second derivatives, x_idx should be set to 3.
    eps=0; verbose=True
    
    pdict = env.ftr_pos_dict
    r_idx, x_idx, fn1_idx, fn2_idx, f1_idx, f2_idx = \
        pdict['r'],pdict['x'], pdict['fn1'],\
        pdict['fn2'], pdict['f1'],pdict['f2']
        
    
#         ep_loss = torch.zeros(env.observation_space.shape[0])
    observations = env.reset(renew_traj=renew_traj);# 
    agent1.reset(); agent2.reset()
    
    old_observations = np.asarray(observations)
    
    state1_old = agent1.observe(old_observations); state2_old = agent2.observe(old_observations)
    
    f1, action1 = agent1.get_force(state1_old, eps=eps, verbose=verbose)
    f2, action2 = agent2.get_force(state2_old, eps=eps, verbose=verbose)
#         ftr_vec = observations+[f1, f2]
    
    t_ts=[]
    r_ts, x_ts = [],[]
    f1_ts, f2_ts = [],[]
    fn1_ts, fn2_ts = [],[]
    act1_ts, act2_ts = [],[]
    u1_ts,u2_ts = [],[];  cum_reward = 0.
#     q10_ts, q11_ts, q20_ts, q21_ts = [],[],[],[]
    
    while True:
        t = env.get_time()
        observations, reward, done, _ = env.step([f1, f2])
        
        t_ts.append(t)
        f1_ts.append(f1); f2_ts.append(f2)
        act1_ts.append(action1[0].numpy())
        if action2!=0: # An idle agent has action=0 all the time
            act2_ts.append(action2[0].numpy())
        r_ts.append(observations[r_idx]); x_ts.append(observations[x_idx]) # Logging
        fn1_ts.append(observations[fn1_idx]); fn2_ts.append(observations[fn2_idx]);
    
        cum_reward += reward#reward
#         u1_ts.append(agent1.compute_utility(reward, f1))
#         u2_ts.append(agent2.compute_utility(reward, f2))
        u1_ts.append(agent1._compute_effort(f1))
        u2_ts.append(agent2._compute_effort(f2))
        
        state1 = agent1.observe(observations); state2 = agent2.observe(observations)
        f1, action1 = agent1.get_force(state1, eps=eps, verbose=verbose)
        f2, action2 = agent2.get_force(state2, eps=eps, verbose=verbose)

        if done is True:
            break

    cum_reward = cum_reward /t
    a1_cum_reward = np.mean(u1_ts)
    a2_cum_reward = np.mean(u2_ts)
    
    t_ts = [t_i*env.tstep for t_i in t_ts]
    return t_ts, (r_ts,x_ts), (f1_ts, f2_ts), \
            (u1_ts, u2_ts, cum_reward, a1_cum_reward, a2_cum_reward),\
            (act1_ts, act2_ts), (fn1_ts, fn2_ts)

#     return t_ts, (r_ts,x_ts), (f1_ts, f2_ts), (q10_ts, q11_ts, q20_ts, q21_ts), (u1_ts, u2_ts, cum_reward, a1_cum_reward, a2_cum_reward)

def single_eval_depr(env, agent1, agent2, n_episodes=100, normalizer=True):
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

def single_eval(env, agent1, agent2, n_episodes=100, normalizer=True, ret_effort=False):
    # Empirical Model Evaluation
    # For assessing the performance of two RLAgent agents playing 
    # on DyadSlider.
    # Arguments:
    # n_episodes: the number of episodes used for averaging the quality of the policy
    # normalize: if True, the reward is normalized by the number of time steps.

    reward_vals = np.zeros(n_episodes)

    effort_vals = np.zeros(n_episodes)
    
    for i_episode in range(n_episodes):
        cum_reward = 0.; cum_effort = 0.;
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
            
            effort = agent1._compute_effort(f1)
            cum_reward += reward
            cum_effort += effort
            
            if done is True:
                break
        if normalizer is True:
            reward_vals[i_episode] = cum_reward /t
            effort_vals[i_episode] = cum_effort /t
        else:
            reward_vals[i_episode] = cum_reward
        
    return np.mean(reward_vals, axis=0), np.mean(effort_vals, axis=0)



def dyad_eval_depr(env, agent1, agent2, n_episodes=100, normalizer=True, ret_utils=False):
    # Empirical Model Evaluation
    # For assessing the performance of two RLAgent agents playing 
    # on DyadSlider.
    # Arguments:
    # n_episodes: the number of episodes used for averaging the quality of the policy
    # normalize: if True, the reward is normalized by the number of time steps.

    reward_vals = np.zeros(n_episodes)
    util1_vals = np.zeros(n_episodes)
    util2_vals = np.zeros(n_episodes)

    for i_episode in range(n_episodes):
        cum_reward = 0.; cum_util1 = 0;cum_util2 = 0;
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
            
            if ret_utils is True:
                cum_util1 += agent1.compute_utility(reward, f1)
                cum_util2 += agent2.compute_utility(reward, f2)
            
            if done is True:
                break
        if normalizer is True:
            reward_vals[i_episode] = cum_reward /t
            util1_vals[i_episode] = cum_util1 /t
            util2_vals[i_episode] = cum_util2 /t
        else:
            reward_vals[i_episode] = cum_reward
            util1_vals[i_episode] = cum_util1
            util2_vals[i_episode] = cum_util2
    
    if ret_utils is True:
        return np.mean(reward_vals), np.mean(util1_vals), np.mean(util2_vals)
    return np.mean(reward_vals)
    
    
def dyad_eval(env, agent1, agent2, n_episodes=100, normalizer=True, ret_effort=False):
    # Empirical Model Evaluation
    # For assessing the performance of two RLAgent agents playing 
    # on DyadSlider.
    # Arguments:
    # n_episodes: the number of episodes used for averaging the quality of the policy
    # normalize: if True, the reward is normalized by the number of time steps.

    reward_vals = np.zeros(n_episodes)
    util1_vals = np.zeros(n_episodes)
    util2_vals = np.zeros(n_episodes)

    for i_episode in range(n_episodes):
        cum_reward = 0.; cum_util1 = 0;cum_util2 = 0;
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
            
            if ret_effort is True:
                cum_util1 += agent1._compute_effort(f1)
                cum_util2 += agent2._compute_effort(f2)
            
            if done is True:
                break
        if normalizer is True:
            reward_vals[i_episode] = cum_reward /t
            util1_vals[i_episode] = cum_util1 /t
            util2_vals[i_episode] = cum_util2 /t
        else:
            reward_vals[i_episode] = cum_reward
            util1_vals[i_episode] = cum_util1
            util2_vals[i_episode] = cum_util2
    
    if ret_effort is True:
        return np.mean(reward_vals), np.mean(util1_vals), np.mean(util2_vals)
    return np.mean(reward_vals)


def benchmark(algo, hp, env, agent1, agent2, xaxis_params):
    # Creates time series for algorithm quality across episodes
    
#     dyad_eval = dyad_eval
    
    t0 = time.time()
    
    # Unzip arguments
    n_episodes, n_intervals, n_eval = xaxis_params
    int_episodes=int(n_episodes/n_intervals)
    
    x, y = np.zeros(n_intervals+1), np.zeros(n_intervals+1)

    x[0] = 0
    y[0] = dyad_eval(env, agent1, agent2, n_episodes=1, normalizer=True, ret_utils=False)
    
    # Evaluate the created policy once every int_episodes episodes
    for i in range(n_intervals):
        print('--------------------------------')
        print('Episode ', i*int_episodes, ': Objective (normalized by duration)= %.3f'%y[i])
        
        for j in range(int_episodes):
            agent1, agent2 = algo(env, agent1, agent2)
            
            if hp.target_int <= 1:
                agent1.update_target_nets() #update_target_qnet()
                agent2.update_target_nets()
            elif i*int_episodes+j % hp.target_int == 0:
                agent1.update_target_nets()
            elif i*int_episodes+j % hp.target_int == int(hp.target_int/2):
                agent2.update_target_nets()
            
        x[i+1] = (i+1)*int_episodes
        y[i+1] = dyad_eval(env, agent1, agent2, n_episodes=n_eval, normalizer=True, ret_utils=ret_utils)
        
        ct = time.time()
        estimated_time_left = (n_intervals-i)*(ct-t0)/(i+1)
        print('Estimated time left: ', estimated_time_left)
        
    print('Total Duration: ', (ct- t0)/60, ' Minutes')
    print('______________________________________')
    return x,y, agent1, agent2


def benchmark_w_utils(algo, hp, env, agent1, agent2, xaxis_params):
    # Creates time series for algorithm quality across episodes
    
    ret_utils = True
    
    t0 = time.time()
    
    # Unzip arguments
    n_episodes, n_intervals, n_eval = xaxis_params
    int_episodes=int(n_episodes/n_intervals)
    
    x, y = np.zeros(n_intervals+1), np.zeros(n_intervals+1)
    util1, util2 = np.zeros(n_intervals+1), np.zeros(n_intervals+1)

    x[0] = 0
    y[0], util1[0], util2[0] = dyad_eval(env, agent1, agent2, n_episodes=1, normalizer=True, ret_utils=ret_utils)
    
    # Evaluate the created policy once every int_episodes episodes
    for i in range(n_intervals):
        print('--------------------------------')
        print('Episode ', i*int_episodes, ': Objective = %.3f'%y[i], 
              ' Agent 1 Utility = %.3f'%util1[i], 
              ': Agent 2 Utility = %.3f'%util2[i])
        
        for j in range(int_episodes):
            agent1, agent2 = algo(env, agent1, agent2)
            
            if hp.target_int <= 1:
                agent1.update_target_nets() #update_target_qnet()
                agent2.update_target_nets()
            elif i*int_episodes+j % hp.target_int == 0:
                agent1.update_target_nets()
            elif i*int_episodes+j % hp.target_int == int(hp.target_int/2):
                agent2.update_target_nets()
            
        x[i+1] = (i+1)*int_episodes
        y[i+1], util1[i+1], util2[i+1] = dyad_eval(env, agent1, agent2, n_episodes=n_eval, normalizer=True, ret_utils=ret_utils)
        
        ct = time.time()
        estimated_time_left = (n_intervals-i)*(ct-t0)/(i+1)
        print('Estimated time left: ', estimated_time_left)
        
    print('Total Duration: ', (ct- t0)/60, ' Minutes')
    print('______________________________________')
    return x,y, agent1, agent2
