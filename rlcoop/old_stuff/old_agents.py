#___________________________________________________    
#### Old agent classes



class DQNAgent(DyadSliderAgent):
    # imports: numpy as np
    # softmax from scipy.special, 
    # FloatTensor as tarr from torch
    
    #__________________________ Interface methods
    def __init__(self, rl, pdcoef, buffer, perspective, sigma,
                 hyperparams=None, force_rms=1., role=None, c_negativef=None, **kwargs):
        super().__init__(force_max=20., force_min=-20.,
                       perspective=perspective, **kwargs)
        self.rl = rl
        self.pd = pdcoef
        self.buffer = buffer
        self.sigma = sigma
        self.force_rms = force_rms
        self.hp = hyperparams
        self.role=role
        self.c_negativef = c_negativef
    
    def set_train_hyperparams(self, hyperparams):
        self.hp = hyperparams
        
    def add_experience(self, trans_tuple):
        self.buffer.push(*trans_tuple)
    
    def train_step(self):
        if len(self.buffer)>1:
            # Run one step of SGD using the whole batch data
            self.rl.step(self.buffer)
    
    def get_force(self, env_state, eps=1., verbose=False):
        
        # role=0: PID controller
        # role=1: do nothing.
        
        if self.role is None:
            force, qvals = self._qbased_force(env_state, eps=eps)
            if verbose is True:
                return force, qvals
            return force
        elif self.role == 1:
            return 0.
        elif self.role == 0:
            return self._apply_pid(env_state) #case: role=0
        else:
            raise ValueError('role should be in {0,1,None}')
        
    def update_target_qnet(self):
        self.rl.update_target_qnet()
    
    
    #__________________________ Methods used in train_step() by the rl torch model
    def get_force_batch(self, state_batch):
        force0_batch = np.zeros((len(state_batch),1)); force1_batch = np.zeros((len(state_batch),1))
        i=0;
        for sample_s in state_batch:
            force0_batch[i] = self._apply_pid(sample_s)
            force1_batch[i] = self._addnoise_cap(0.)
        
        return (force0_batch, force1_batch)
        
    
    def compute_utility(self, reward, force):
        return self.c_error*reward - self._compute_effort(force)
    
    #__________________________ Methods used by the agent itself
    
    def _compute_effort(self, force):
        scaled_force = force/self.force_rms
        if self.c_negativef is not None:
            if scaled_force<0:
                return self.c_negativef*(-scaled_force)
            else:
                return self.c_effort* scaled_force
        else:
            return self.c_effort* abs(scaled_force)
    
    
    def _qbased_force(self, env_state, eps=0.):
        # returns the appropriate role given the q
        
        force0 = self._apply_pid(env_state)
        force1 = self._addnoise_cap(0.)
        
        # Assuming the input to qnet is [state, action]
        q0 = self.rl.net(torch.cat( (tarr(env_state), tarr([force0])) )) 
        q1 = self.rl.net(torch.cat( (tarr(env_state), tarr([force1])) ))
        
        q0 = float(q0); q1 = float(q1)
        qvals = [q0, q1]
        
        if eps>0. and random.random()<eps:
            probs = softmax([q0,q1])
            if random.random()< probs[0]:
                return force0, qvals
            else:
                return force1, qvals
        else:
            if q0>q1:
                return force0, qvals
            else:
                return force1, qvals
    
    
    def _addnoise_cap(self, force):
        mlt_noise = force*np.random.normal(0, self.sigma)
        add_noise = np.random.normal(0, self.sigma)
        noisy_force = force+ mlt_noise+add_noise
        force_capped = self.force_max*np.tanh((2./self.force_max) *noisy_force)
        return force_capped
        
        
    def _apply_pid(self, env_state):
        # env_state can be any 1dimensional indexable sequence
        
        e = env_state[0]-env_state[2]
        ed = env_state[1]-env_state[3]
        if self.perspective !=0:
            e = -e; ed=-ed
        force = np.dot(self.pd, [e,ed])
        capped_noisy_force = self._addnoise_cap(force)
        return capped_noisy_force

class ACAgent(DyadSliderAgent):
    # imports: numpy as np
    # FloatTensor as tarr from torch
    
    #__________________________ Interface methods
    def __init__(self, nn_mod, buffer, muscle, perspective,
                 hyperparams=None, force_rms=1., c_negativef=None, **kwargs):
        super().__init__(perspective=perspective, **kwargs)
        self.nn_mod = nn_mod # nn_mod is an object, containing the neural net and the methods for training it.
        self.buffer = buffer
        self.muscle = muscle
        self.force_rms = force_rms
        self.hp = hyperparams
        self.c_negativef = c_negativef
    
    def set_train_hyperparams(self, hyperparams):
        self.hp = hyperparams
        
    def add_experience(self, trans_tuple):
        self.buffer.push(*trans_tuple)
    
    def train_step(self):
        if len(self.buffer)>1:
            # Run one step of SGD using the whole batch data
            self.nn_mod.step(self.buffer)
    
#     def state_estimator(self, env_state, isbatch=False):
#         # Estimates positional error from sensory inputs
#         if isbatch:
#             e = env_state[:, 0]-env_state[:, 2]
#             edot = env_state[:, 1]-env_state[:, 3]
#             state = env_state; state[:,2], state[:,3] = e, edot 
# #             f_partner = 
#         else:
#             e = env_state[0]-env_state[2]
#             edot = env_state[1]-env_state[3]
#             state = env_state; state[2], state[3] = e, edot 
#         return state
    
    
    def observe(self, environment_state):
        r, r_dot, x, x_dot, \
        force_interaction, force_interaction_dot = environment_state

        if self.perspective % 2 == 1:
            r, r_dot, x, x_dot = -r, -r_dot, -x, -x_dot

        error = r - x
        error_dot = r_dot - x_dot
        
#         norm_coef = 1.
        observation = np.array([r, r_dot, error, error_dot,
                                force_interaction, force_interaction_dot])
        return observation
    
        
    def get_force(self, state, eps=0., verbose=False):
        
        action = self._act(state, eps)
        force = float(self.muscle.get_force(action))

        if verbose is True:
            return force, action
        return force

    def update_target_qnet(self):
        self.nn_mod.update_target_vnet()
    
    
    #__________________________ Methods used in train_step() by the rl torch model
    
    def compute_utility(self, reward, force):
        return self.c_error*reward - self._compute_effort(force)
    
    #__________________________ Methods used by the agent itself
    
    def _compute_effort(self, force):
        scaled_force = force/self.force_rms
        if self.c_negativef is not None:
            if scaled_force<0:
                return self.c_negativef*(scaled_force**2)
            else:
                return self.c_effort* (scaled_force**2)
        else:
            return self.c_effort* (scaled_force**2)#abs(scaled_force)
    
    
    def _act(self, observation, eps):
#         obs_tar = torch.FloatTensor(observation)#.to(device)
        obs_tar = torch.as_tensor(observation, dtype=torch.float32)
        
        if obs_tar.dim()==1:
            obs_tar = obs_tar.view(1,-1)
#         other_ftrs = ...
        dist = self.nn_mod.actor_net(obs_tar)
        
#         if random.random()<0.01:
#             print('Action Std = ', dist.stddev)  #@@@@@@@@@@@@@@@@
        
        if eps>0.:
            action = dist.sample().item()
        elif eps ==0:
            action = dist.mean.detach().numpy()
#         print('dist, action = ', dist, action)
        return action

    def reset(self):
        self.muscle.reset()
        
class ACAgent2(ACAgent):
    # imports: numpy as np
    # FloatTensor as tarr from torch
    # Including critic representations in the feature vector
    
    # Works with ACTrainer2
    
    def _act(self, observation, eps):
        obs_tar = torch.as_tensor(observation, dtype=torch.float32)
        if obs_tar.dim()==1:
            obs_tar = obs_tar.view(1,-1)
        _, critic_repr = self.nn_mod.target_vnet(obs_tar, return_hidden=True)
        # Calculate the feature representations from state predictor.
         #(use the current muscle force for control input)
        
        # Concatenate feature representations
        features = torch.cat((obs_tar, self.nn_mod.critic_ftr_scale* critic_repr), 1)

        dist = self.nn_mod.actor_net(features)
        
#         if random.random()<0.01:
#             print('Action Std = ', dist.stddev)  #@@@@@@@@@@@@@@@@
        
        if eps>0.:
            action = dist.sample().item()
        elif eps ==0:
            action = dist.mean.detach().numpy()
#         print('dist, action = ', dist, action)
        return action


class ACAgent3(ACAgent):
    # Adding own force to the feature vector.
    # Adding target network for the policy.
    
    # imports: numpy as np
    # FloatTensor as tarr from torch
    def observe(self, environment_state):
        r, r_dot, x, x_dot, \
        force_interaction, force_interaction_dot = environment_state

        if self.perspective % 2 == 1:
            r, r_dot, x, x_dot = -r, -r_dot, -x, -x_dot

        error = r - x
        error_dot = r_dot - x_dot
        
        own_force = float(self.muscle.muscle_force)
#         norm_coef = 1.
        observation = np.array([r, r_dot, error, error_dot,
                                force_interaction, force_interaction_dot, own_force])
        return observation
    
    
    def _act(self, observation, eps):
#         obs_tar = torch.FloatTensor(observation)#.to(device)
        obs_tar = torch.as_tensor(observation, dtype=torch.float32)
        
        if obs_tar.dim()==1:
            obs_tar = obs_tar.view(1,-1)
#         other_ftrs = ...
        dist = self.nn_mod.target_pnet(obs_tar)
        
#         if random.random()<0.01:
#             print('Action Std = ', dist.stddev)  #@@@@@@@@@@@@@@@@
        
        if eps>0.:
            action = dist.sample().item()
        elif eps ==0:
            action = dist.mean.detach().numpy()
#         print('dist, action = ', dist, action)
        return action 

    def update_target_nets(self):
        self.nn_mod.update_target_nets()
    

class PPOAgent(ACAgent3):
    # Also return log_probs for the actions, to be used later for the PPO loss function.
    def get_force(self, state, eps=0., verbose=False):
        
        action = self._act(state, eps) #action is a np array of [action, logprob]
        
        force = float(self.muscle.get_force(action[0]))

        if verbose is True:
            return force, action
        return force
    
    def _act(self, observation, eps):
#         obs_tar = torch.FloatTensor(observation)#.to(device)
        obs_tar = torch.as_tensor(observation, dtype=torch.float32)
        
        if obs_tar.dim()==1:
            obs_tar = obs_tar.view(1,-1)
#         other_ftrs = ...
        dist = self.nn_mod.target_pnet(obs_tar)
        
#         if random.random()<0.01:
#             print('Action Std = ', dist.stddev)  #@@@@@@@@@@@@@@@@
        
        if eps>0.:
            action = dist.sample().item()
        elif eps ==0:
            action = dist.mean.detach().numpy()
#         print('dist, action = ', dist, action)
        log_prob = dist.log_prob(torch.tensor(action)).detach().item()
    
        return np.array([action, log_prob])

class PPO2Agent(PPOAgent):
    # Works with PPO2Trainer, which computes a separate advantage function for the effort cost.
    # This agent class computes effort on a batch of states.
    
    def __init__(self, nn_mod, buffer, muscle, perspective,
                 hyperparams=None, force_rms=1., c_positivef=0., 
                 c_negativef=0., **kwargs):
        
        super().__init__(nn_mod, buffer, muscle, perspective,
                 hyperparams, force_rms, **kwargs)
        self.c_positivef = c_positivef
        self.c_negativef = c_negativef
    
    
    
    def train_step(self, current_performance):
        if len(self.buffer)>1:
            # Run one step of SGD using the whole batch data
            self.nn_mod.step(self.buffer, current_performance)
            
    def _compute_effort(self, force):
#         scaled_force = force/self.force_rms
        coef = self.c_positivef if force>0 else self.c_negativef
        return coef*(force**2)
    
    def observe(self, environment_state):
        r, r_dot, x, x_dot, \
            f1,f2 = environment_state

        if self.perspective % 2 == 1:
            r, r_dot, x, x_dot = -r, -r_dot, -x, -x_dot
            f_own, f_other = f2, f1
        else:
            f_own, f_other = f1, f2

        error = r - x
        error_dot = r_dot - x_dot
        
        observation = np.array([r, r_dot, error, error_dot,
                                f_other, 0., f_own])
        return observation
    
class PPO3Agent(PPO2Agent):
    # Very small change: just giving eps to the target_pnet
    def _act(self, observation, eps):
#         obs_tar = torch.FloatTensor(observation)#.to(device)
        obs_tar = torch.as_tensor(observation, dtype=torch.float32)
        
        if obs_tar.dim()==1:
            obs_tar = obs_tar.view(1,-1)
#         other_ftrs = ...
        dist = self.nn_mod.target_pnet(obs_tar, eps=eps)
        
        if random.random()<0.00003:
            print('Action Std = ', dist.scale.item())  #@@@@@@@@@@@@
        
        if eps>0.:
            action = dist.sample().item()
        elif eps ==0:
            action = dist.loc.detach().item() #numpy()
#         print('dist, action = ', dist, action)
        log_prob = dist.log_prob(torch.tensor(action)).detach().item()
    
        return np.array([action, log_prob])
    
    
class PPO4Agent(PPO3Agent):
    # Modified the observe method to account for r", x" which are added to env_track_dyadic_w_a
    def observe(self, environment_state):
        r, r_d, r_dd, x, x_d, x_dd, \
            f1,f2 = environment_state

        if self.perspective % 2 == 1:
            r, r_d, r_dd, x, x_d, x_dd = -r, -r_d,-r_dd, -x, -x_d, -x_dd
            f_own, f_other = f2, f1
        else:
            f_own, f_other = f1, f2

        error = r - x
        error_d = r_d - x_d
        error_dd = r_dd - x_dd
        
        observation = np.array([r, r_d, r_dd, error, error_d, error_dd,
                                f_other, 0., f_own])
        return observation
    
    
class PPO4PDAgent(PPO4Agent):
    # Modified the act and get_force methods to work with the actions being the PID gains
    
    def __init__(self, nn_mod, buffer, muscle, perspective,
                 hyperparams=None, force_rms=1., c_positivef=0., 
                 c_negativef=0., ctrl_ftrs=slice(3,5), pd_scalers=[1., 0.1], **kwargs):
        
        super().__init__(nn_mod, buffer, muscle, perspective,
                 hyperparams, force_rms, c_positivef, 
                 c_negativef, **kwargs)
        self.ctrl_ftrs = ctrl_ftrs
        self.pd_scalers = np.asarray(pd_scalers)
    
    def _act(self, observation, eps):
#         obs_tar = torch.FloatTensor(observation)#.to(device)
        obs_tar = torch.as_tensor(observation, dtype=torch.float32)
        
        if obs_tar.dim()==1:
            obs_tar = obs_tar.view(1,-1)
#         other_ftrs = ...
        dist = self.nn_mod.target_pnet(obs_tar, eps=eps)
        
        if random.random()<0.00003:
            print('Action Std = ', dist.stddev.detach())  #@@@@@@@@@@@@
        
        if eps>0.:
            action = dist.sample()
        elif eps ==0:
            action = dist.loc
#         print('dist, action = ', dist, action)
        log_prob = dist.log_prob(action).detach().item()
    
        return [action.detach().numpy(), log_prob]
    
    def _gain2cmd(self, rel_gains, state):
        err_ftrs = state[self.ctrl_ftrs]
        abs_gains = self.pd_scalers* rel_gains
        cmd = np.inner(err_ftrs, abs_gains)
        return cmd
        
    def get_force(self, state, eps=0., verbose=False):
        
        rel_gains_logprob = self._act(state, eps) #gains_logprob is a np array of [rel_gains, logprob]
        cmd = self._gain2cmd(rel_gains_logprob[0], state)
        force = float(self.muscle.get_force(cmd))

        if verbose is True:
            return force, rel_gains_logprob
        return force

    
    
class PPO5PDAgent(PPO4PDAgent):
    # Modified everything to work with GPU
        
    def __init__(self, nn_mod, buffer, muscle, perspective,
                 hyperparams=None, force_rms=1., c_positivef=0., 
                 c_negativef=0., ctrl_ftrs=slice(3,5), pd_scalers=[1., 0.1], **kwargs):
        
        super().__init__(nn_mod, buffer, muscle, perspective,
                 hyperparams, force_rms, c_positivef, 
                 c_negativef, **kwargs)
        self.ctrl_ftrs = ctrl_ftrs
        self.pd_scalers = torch.tensor(pd_scalers, device=self.nn_mod.device, dtype=torch.float32)
        
    def observe(self, environment_state):
        r, r_d, r_dd, x, x_d, x_dd, \
            f1,f2 = environment_state

        if self.perspective % 2 == 1:
            r, r_d, r_dd, x, x_d, x_dd = -r, -r_d,-r_dd, -x, -x_d, -x_dd
            f_own, f_other = f2, f1
        else:
            f_own, f_other = f1, f2

        error = r - x
        error_d = r_d - x_d
        error_dd = r_dd - x_dd
        
        observation = torch.tensor([[r, r_d, r_dd, error, error_d, error_dd,
                                f_other, 0., f_own]], dtype=torch.float32, 
                                   device=self.nn_mod.device)
        return observation 
    
    
    def _act(self, state_te, eps):
        # state_te is a tensor of observations
        
        if state_te.dim()==1:
            state_te = state_te.view(1,-1)

            
        dist = self.nn_mod.target_pnet(state_te, eps=eps)
        
#         if dist.dim()==1:
#             raise ValueError('dist: '+str(dist.shape)) #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        
        if random.random()<0.00003:
            print('Action Std = ', dist.stddev.detach())  #@@@@@@@@@@@@
        
        if eps>0.:
            action = dist.sample()
        elif eps ==0:
            action = dist.loc
        log_prob = dist.log_prob(action).detach()
    
        return [action.detach(), log_prob]
    
    def _gain2cmd(self, rel_gains, state):
        err_ftrs = state[self.ctrl_ftrs]
        abs_gains = self.pd_scalers* rel_gains
        cmd = torch.dot(err_ftrs, abs_gains)
        return cmd
        
    def get_force(self, state, eps=0., verbose=False):
        
        rel_gains_logprob = self._act(state, eps) #gains_logprob is a np array of [rel_gains, logprob]
        cmd = self._gain2cmd(rel_gains_logprob[0].squeeze(), state.squeeze()).cpu().numpy()
        force = float(self.muscle.get_force(cmd))

        if verbose is True:
            return force, rel_gains_logprob
        return force
    

    def compute_effort_batch(self, f_batch):
        # receives a batch of states and returns a batch of effort associated with each.
        coefs = torch.where(f_batch>0, self.c_positivef, self.c_negativef)#.to(self.nn_mod.device)
        effort_batch = coefs* (f_batch**2)
        return effort_batch
        
        
    
class PPO5PDAgent_wfn(PPO5PDAgent):
    # Modified everything to work with GPU
        
    def __init__(self, nn_mod, buffer, muscle, perspective,
                 hyperparams=None, force_rms=1., c_positivef=0., 
                 c_negativef=0., ctrl_ftrs=slice(3,5), 
                 pd_scalers=[1., 0.1], mass=0.1, fric=0.1, **kwargs):
        
        super().__init__(nn_mod, buffer, muscle, perspective,
                 hyperparams, force_rms, c_positivef, 
                 c_negativef, ctrl_ftrs, pd_scalers, **kwargs)
        self.mass = mass
        self.fric = fric
        
    def observe(self, environment_state):
        r, r_d, r_dd, x, x_d, x_dd, \
            f1,f2 = environment_state

        if self.perspective % 2 == 1:
            r, r_d, r_dd, x, x_d, x_dd = -r, -r_d,-r_dd, -x, -x_d, -x_dd
            f_own, f_other = f2, f1
        else:
            f_own, f_other = f1, f2
        
        fn = f_own - self.mass*x_dd -self.fric*x_d
        error = r - x
        error_d = r_d - x_d
        error_dd = r_dd - x_dd
        
        observation = torch.tensor([[r, r_d, r_dd, error, error_d, error_dd,
                                fn, 0., f_own]], dtype=torch.float32, 
                                   device=self.nn_mod.device)
        return observation 
    
    
class PPO5PDAgent_env3(PPO5PDAgent):
        
    def observe(self, environment_state):
        r, r_d, r_dd, x, x_d, x_dd, \
            fn1,fn2, f1,f2 = environment_state

        if self.perspective % 2 == 1:
            r, r_d, r_dd, x, x_d, x_dd = -r, -r_d,-r_dd, -x, -x_d, -x_dd
            fn = fn2
            f_own = f2
        else:
            fn = fn1
            f_own = f1
        
        error = r - x
        error_d = r_d - x_d
        error_dd = r_dd - x_dd
        
        observation = torch.tensor([[r, r_d, r_dd, error, error_d, error_dd,
                                fn, 0., f_own]], dtype=torch.float32, 
                                   device=self.nn_mod.device)
        return observation 

    def _gain2cmd(self, rel_gains, state):
        err_ftrs = state[self.ctrl_ftrs]
#         print(self.pd_scalers.dtype, rel_gains.dtype)
        abs_gains = torch.exp(self.pd_scalers* rel_gains)
#         print(err_ftrs.dtype, abs_gains.dtype)
        cmd = torch.dot(err_ftrs, abs_gains)
        return cmd

