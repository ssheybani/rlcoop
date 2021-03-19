#______________________________________________
# Old versions

def train_single(env, agent1, agent2, update_interval=25):
    #required imports: none
    old_observations = env.reset(renew_traj=True, max_freq='random'); #old_observations = np.asarray(observations)
    agent1.reset(); agent2.reset()
    
    state1_old = agent1.observe(old_observations)#, agent2.observe(old_observations)
    f1, action1 = agent1.get_force(state1_old, eps=1., verbose=True)
    f2, action2 = 0., 0. #agent2.get_force(state2_old, eps=1., verbose=True)

    while True:
        t = env.get_time()
#         print('f1, f2', f1, f2) #@@@@@@@@@@
         
        observations, reward, done, _ = env.step([f1, f2]) #@@@@@@@@@
        
        if t%1 ==0:
            state1 = agent1.observe(observations)
#             , \            agent2.observe(observations)
            
            utility1 = agent1.compute_utility(reward, f1)
#         , \            agent2.compute_utility(reward, f2)
                    
        # Add the experience
            err = observations[0]-observations[2]
            if abs(err)> env.max_err*agent1.buffer.tag:
                agent1.add_experience((state1_old, action1, state1, utility1))
#             agent2.add_experience((state2_old, action2, state2, utility2))
            state1_old = state1#, state2
            
            f1, action1 = agent1.get_force(state1, eps=1., verbose=True)
            
        else:
            f1 = agent1.muscle.get_force(action1)
            f2 = 0. #agent2.muscle.get_force(action2)
            
        # Take one training step for each agent
        if t%update_interval==0:
            agent1.train_step()
#             agent2.train_step() #@@@@@@@@@@@
#             print('t=', t)
#             print('a1fc2 grad = ', agent1.nn_mod.actor_net.fc2.weight)
#         if t==0:
#             print('Agent 1 actor weights norm: ', 
#                   agent1.nn_mod.actor_net.fc4.weight[0].norm())
#             ,                  agent1.nn_mod.actor_net.fc1.weight[1][0:4])
#                   agent1.nn_mod.optim.state_dict()['state'])
        # Generate action for next env interaction
#         f1, action1 = agent1.get_force(state1, eps=1., verbose=True)
#         f2, action2 = agent2.get_force(state2, eps=1., verbose=True)
#         old_observations = observations
#         state1_old, state2_old = state1, state2

        if done is True:
            break
    return agent1, agent2


def train_single2(env, agent1, agent2, update_interval=25, current_performance=-100.):
    #required imports: none
    old_observations = env.reset(renew_traj=True, max_freq='random'); #old_observations = np.asarray(observations)
    agent1.reset(); agent2.reset()
    
    state1_old = agent1.observe(old_observations)#, agent2.observe(old_observations)
    f1, action1 = agent1.get_force(state1_old, eps=1., verbose=True)
    f2, action2 = 0., 0. #agent2.get_force(state2_old, eps=1., verbose=True)

    while True:
        t = env.get_time()
#         print('f1, f2', f1, f2) #@@@@@@@@@@
         
        observations, reward, done, _ = env.step([f1, f2]) #@@@@@@@@@
        
        if t%1 ==0:
            state1 = agent1.observe(observations)
#             , \            agent2.observe(observations)
                    
        # Add the experience
            err = observations[0]-observations[2]
            if abs(err)> env.max_err*agent1.buffer.tag:
                agent1.add_experience((state1_old, action1, state1, reward))
#             agent2.add_experience((state2_old, action2, state2, reward))
            state1_old = state1#, state2
            
            f1, action1 = agent1.get_force(state1, eps=1., verbose=True)
            
        else:
            f1 = agent1.muscle.get_force(action1)
            f2 = 0. #agent2.muscle.get_force(action2)
            
        # Take one training step for each agent
        if t%update_interval==0:
            agent1.train_step(current_performance)

        if done is True:
            break
    return agent1, agent2


def train_single3(env, agent1, agent2, update_interval=25, current_performance=-100., ret_perf=False):
    # calculate eps using current performance
    old_observations = env.reset(renew_traj=True, max_freq='random'); #old_observations = np.asarray(observations)
    agent1.reset(); agent2.reset()
    
#     eps = 1- scipy.special.expit(-16*np.log10(-current_performance)-12)
    eps=1.
            
    state1_old = agent1.observe(old_observations)#, agent2.observe(old_observations)
    f1, action1 = agent1.get_force(state1_old, eps=eps, verbose=True)
    f2, action2 = 0., 0. #agent2.get_force(state2_old, eps=1., verbose=True)

    cum_reward = 0.
    
    while True:
        t = env.get_time()
#         print('f1, f2', f1, f2) #@@@@@@@@@@
         
        observations, reward, done, _ = env.step([f1, f2]) #@@@@@@@@@
        cum_reward += reward
        
        if t%1 ==0:
            state1 = agent1.observe(observations)
#             , \            agent2.observe(observations)
                    
        # Add the experience
#             err = observations[0]-observations[2]
#             if abs(err)> env.max_err*agent1.buffer.tag:
            agent1.add_experience((state1_old, action1, state1, reward))
#             agent2.add_experience((state2_old, action2, state2, reward))
            state1_old = state1#, state2
            
            f1, action1 = agent1.get_force(state1, eps=eps, verbose=True)
            
        else:
            f1 = agent1.muscle.get_force(action1)
            f2 = 0. #agent2.muscle.get_force(action2)
            
        # Take one training step for each agent
        if t%update_interval==0:
            agent1.train_step(current_performance)

        if done is True:
            break
    if ret_perf is True:
        return agent1, agent2, cum_reward /t
    else:
        return agent1, agent2
    

def train_single4(env, agent1, agent2, update_interval=25, current_performance=-100., ret_perf=False):
    # calculate eps using current performance
    old_observations = env.reset(renew_traj=True, max_freq='random'); #old_observations = np.asarray(observations)
    agent1.reset(); agent2.reset()
    
#     eps = 1- scipy.special.expit(-16*np.log10(-current_performance)-12)
    eps=1.
            
    state1_old = agent1.observe(old_observations)#, agent2.observe(old_observations)
    f1, action1 = agent1.get_force(state1_old, eps=eps, verbose=True)
    f2, action2 = 0., 0. #agent2.get_force(state2_old, eps=1., verbose=True)

    cum_reward = 0.
    
    while True:
        t = env.get_time()
#         print('f1, f2', f1, f2) #@@@@@@@@@@
         
        observations, reward, done, _ = env.step([f1, f2]) #@@@@@@@@@
        cum_reward += reward
        
        if t%1 ==0:
            state1 = agent1.observe(observations)
#             , \            agent2.observe(observations)
                    
        # Add the experience
#             err = observations[0]-observations[2]
#             if abs(err)> env.max_err*agent1.buffer.tag:
            agent1.add_experience((state1_old, action1, state1, 
                                  torch.tensor([reward], device=agent1.nn_mod.device, dtype=torch.float32)
                                  ))
#             agent2.add_experience((state2_old, action2, state2, reward))
            state1_old = state1#, state2
            
            f1, action1 = agent1.get_force(state1, eps=eps, verbose=True)
            
        else:
            f1 = agent1.muscle.get_force(action1)
            f2 = 0. #agent2.muscle.get_force(action2)
            
        # Take one training step for each agent
        if t%update_interval==0:
            agent1.train_step(current_performance)

        if done is True:
            break
    if ret_perf is True:
        return agent1, agent2, cum_reward /t
    else:
        return agent1, agent2
    



def train_dyad(env, agent1, agent2, update_interval=25):
    #required imports: none
    old_observations = env.reset(renew_traj=True); #old_observations = np.asarray(observations)
    agent1.reset(); agent2.reset()
    
    state1_old = agent1.observe(old_observations); state2_old = agent2.observe(old_observations)
    f1, action1 = agent1.get_force(state1_old, eps=1., verbose=True)
    f2, action2 = agent2.get_force(state2_old, eps=1., verbose=True)

    while True:
        t = env.get_time()
#         print('f1, f2', f1, f2) #@@@@@@@@@@
         
        observations, reward, done, _ = env.step([f1, f2]) #@@@@@@@@@
        
        if t%1 ==0:
            state1 = agent1.observe(observations)
            state2 = agent2.observe(observations)
            
            utility1 = agent1.compute_utility(reward, f1)
            utility2 = agent2.compute_utility(reward, f2)
                    
        # Add the experience
#             err = observations[0]-observations[2]
#             if abs(err)> env.max_err*agent1.buffer.tag:
            agent1.add_experience((state1_old, action1, state1, utility1))
            agent2.add_experience((state2_old, action2, state2, utility2))
            state1_old = state1; state2_old = state2
            
            f1, action1 = agent1.get_force(state1, eps=1., verbose=True)
            f2, action2 = agent2.get_force(state2, eps=1., verbose=True)
            
        else:
            f1 = agent1.muscle.get_force(action1)
            f2 = agent2.muscle.get_force(action2)
            
        # Take one training step for each agent
        if t%update_interval==0:
            agent1.train_step()
            agent2.train_step()

        if done is True:
            break
    return agent1, agent2



def train_dyad2(env, agent1, agent2, update_interval=25, current_performance=-100.):
    #required imports: none
    old_observations = env.reset(renew_traj=True, max_freq='max'); #old_observations = np.asarray(observations)
    agent1.reset(); agent2.reset()
    
    state1_old = agent1.observe(old_observations); state2_old = agent2.observe(old_observations)
    f1, action1 = agent1.get_force(state1_old, eps=1., verbose=True)
    f2, action2 = agent2.get_force(state2_old, eps=1., verbose=True)

    while True:
        t = env.get_time()
#         print('f1, f2', f1, f2) #@@@@@@@@@@
         
        observations, reward, done, _ = env.step([f1, f2]) #@@@@@@@@@
        
        if t%1 ==0:
            state1 = agent1.observe(observations)
            state2 = agent2.observe(observations)
            
#             utility1 = agent1.compute_utility(reward, f1)
#             utility2 = agent2.compute_utility(reward, f2)
                    
        # Add the experience
#             err = observations[0]-observations[2]
#             if abs(err)> env.max_err*agent1.buffer.tag:
            agent1.add_experience((state1_old, action1, state1, reward))
            agent2.add_experience((state2_old, action2, state2, reward))
            state1_old = state1; state2_old = state2
            
            f1, action1 = agent1.get_force(state1, eps=1., verbose=True)
            f2, action2 = agent2.get_force(state2, eps=1., verbose=True)
            
        else:
            f1 = agent1.muscle.get_force(action1)
            f2 = agent2.muscle.get_force(action2)
            
        # Take one training step for each agent
        if t%update_interval==0:
            agent1.train_step(current_performance)
            agent2.train_step(current_performance)
#             print('t=', t)
#             print('a1fc2 grad = ', agent1.nn_mod.actor_net.fc2.weight)
#         if t==0:
#             print('Agent 1 actor weights norm: ', 
#                   agent1.nn_mod.actor_net.fc4.weight[0].norm())
#             ,                  agent1.nn_mod.actor_net.fc1.weight[1][0:4])
#                   agent1.nn_mod.optim.state_dict()['state'])
        # Generate action for next env interaction
#         f1, action1 = agent1.get_force(state1, eps=1., verbose=True)
#         f2, action2 = agent2.get_force(state2, eps=1., verbose=True)
#         old_observations = observations
#         state1_old, state2_old = state1, state2

        if done is True:
            break
    return agent1, agent2


def train_dyad3(env, agent1, agent2, update_interval=25, current_performance=-100., ret_perf=False):
    #required imports: none
    
    eps = 1- scipy.special.expit(-16*np.log10(-current_performance)-12)
     
    old_observations = env.reset(renew_traj=True, max_freq='max'); #old_observations = np.asarray(observations)
    agent1.reset(); agent2.reset()
    
    state1_old = agent1.observe(old_observations); state2_old = agent2.observe(old_observations)
    f1, action1 = agent1.get_force(state1_old, eps=eps, verbose=True)
    f2, action2 = agent2.get_force(state2_old, eps=eps, verbose=True)
    
    cum_rewards = 0.
    while True:
        t = env.get_time()
#         print('f1, f2', f1, f2) #@@@@@@@@@@
         
        observations, reward, done, _ = env.step([f1, f2]) #@@@@@@@@@
        cum_rewards +=reward
        
        if t%1 ==0:
            state1 = agent1.observe(observations)
            state2 = agent2.observe(observations)
            
#             utility1 = agent1.compute_utility(reward, f1)
#             utility2 = agent2.compute_utility(reward, f2)
                    
        # Add the experience
#             err = observations[0]-observations[2]
#             if abs(err)> env.max_err*agent1.buffer.tag:
            agent1.add_experience((state1_old, action1, state1, reward))
            agent2.add_experience((state2_old, action2, state2, reward))
            state1_old = state1; state2_old = state2
            
            f1, action1 = agent1.get_force(state1, eps=eps, verbose=True)
            f2, action2 = agent2.get_force(state2, eps=eps, verbose=True)
            
        else:
            f1 = agent1.muscle.get_force(action1)
            f2 = agent2.muscle.get_force(action2)
            
        # Take one training step for each agent
        if t%update_interval==0:
            agent1.train_step(current_performance)
            agent2.train_step(current_performance)
#             print('t=', t)
#             print('a1fc2 grad = ', agent1.nn_mod.actor_net.fc2.weight)
#         if t==0:
#             print('Agent 1 actor weights norm: ', 
#                   agent1.nn_mod.actor_net.fc4.weight[0].norm())
#             ,                  agent1.nn_mod.actor_net.fc1.weight[1][0:4])
#                   agent1.nn_mod.optim.state_dict()['state'])
        # Generate action for next env interaction
#         f1, action1 = agent1.get_force(state1, eps=1., verbose=True)
#         f2, action2 = agent2.get_force(state2, eps=1., verbose=True)
#         old_observations = observations
#         state1_old, state2_old = state1, state2

        if done is True:
            break
    if ret_perf is True:
        return agent1, agent2, cum_reward /t
    else:
        return agent1, agent2

def train_dyad4(env, agent1, agent2, update_interval=25, current_performance=-100., ret_perf=False):
    
    eps = 1. #- scipy.special.expit(-16*np.log10(-current_performance)-12)
     
    old_observations = env.reset(renew_traj=True, max_freq='max'); #old_observations = np.asarray(observations)
    agent1.reset(); agent2.reset()
    
    state1_old = agent1.observe(old_observations); state2_old = agent2.observe(old_observations)
    f1, action1 = agent1.get_force(state1_old, eps=eps, verbose=True)
    f2, action2 = agent2.get_force(state2_old, eps=eps, verbose=True)
    
    cum_rewards = 0.
    while True:
        t = env.get_time()
#         print('f1, f2', f1, f2) #@@@@@@@@@@
         
        observations, reward, done, _ = env.step([f1, f2]) #@@@@@@@@@
        cum_rewards +=reward
        
        if t%1 ==0:
            state1 = agent1.observe(observations)
            state2 = agent2.observe(observations)
            
#             utility1 = agent1.compute_utility(reward, f1)
#             utility2 = agent2.compute_utility(reward, f2)
                    
        # Add the experience
#             err = observations[0]-observations[2]
#             if abs(err)> env.max_err*agent1.buffer.tag:
            agent1.add_experience((state1_old, action1, state1, 
                                  torch.tensor([reward], device=agent1.nn_mod.device, dtype=torch.float32)
                                  ))
            agent2.add_experience((state2_old, action2, state2, 
                                  torch.tensor([reward], device=agent2.nn_mod.device, dtype=torch.float32)
                                  ))
            state1_old = state1; state2_old = state2
            
            f1, action1 = agent1.get_force(state1, eps=eps, verbose=True)
            f2, action2 = agent2.get_force(state2, eps=eps, verbose=True)
            
        else:
            f1 = agent1.muscle.get_force(action1)
            f2 = agent2.muscle.get_force(action2)
            
        # Take one training step for each agent
        if t%update_interval==0:
            agent1.train_step(current_performance)
            agent2.train_step(current_performance)
#             print('t=', t)
#             print('a1fc2 grad = ', agent1.nn_mod.actor_net.fc2.weight)
#         if t==0:
#             print('Agent 1 actor weights norm: ', 
#                   agent1.nn_mod.actor_net.fc4.weight[0].norm())
#             ,                  agent1.nn_mod.actor_net.fc1.weight[1][0:4])
#                   agent1.nn_mod.optim.state_dict()['state'])
        # Generate action for next env interaction
#         f1, action1 = agent1.get_force(state1, eps=1., verbose=True)
#         f2, action2 = agent2.get_force(state2, eps=1., verbose=True)
#         old_observations = observations
#         state1_old, state2_old = state1, state2

        if done is True:
            break
    if ret_perf is True:
        return agent1, agent2, cum_reward /t
    else:
        return agent1, agent2

