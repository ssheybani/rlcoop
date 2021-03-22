# Agent class
import numpy as np
import sys,os, inspect
# SCRIPT_DIR = os.path.realpath(os.path.dirname(inspect.getfile(inspect.currentframe())))
# PARENT_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, '..'))
# sys.path.append(os.path.join(PARENT_PATH,'configs'))
# sys.path.append(os.path.join(PARENT_PATH,'agents'))
# sys.path.append(os.path.join(PARENT_PATH,'algos'))
# sys.path.append(os.path.join(PARENT_PATH,'util'))
# sys.path.append(os.path.join(PARENT_PATH,'envs'))

import torch
from torch import FloatTensor as tarr
from scipy.special import softmax
import random
# import warnings 

    
    
class PPO5PIDAgent():
    
    """
    Properties:
        Linked objects:
            hp, buffer, muscle, nn_mod
        Force limits and scale:
            force_max, force_min, force_rms
        Controller-related
            ctrl_ftrs, pid_scalers, alpha_eint
        Cost-related:
            c_effort, c_error, c_positivef, c_negativef
        State-related:
            force, force_int
        Perspective within a dyad:
            perpective
        
            ftr_pos_dict
    
    Methods:
        
        Construction
            __init__
            reset
            set_train_hyperparams
        
        Used in Simulation (train_agents.py)
            add_experience
            observe
            get_force
                _act
                _gain2cmd
            train_step
                compute_effort_batch (used in torch_trainer)
        Used in Benchmarking (benchmark_agent or the main script)
            update_target_nets

        Used in analysis
            compute_effort
            compute_utility
    """
    ###############
    # Construction
    ###############
    ftr_pos_dict = {'r':0, 'r\'':1, 'r\"':2,
            'e':3, 'e\'':4, 'e_int':5,
            'fn':6, 'none':7, 'f_own':8}
    
    def __init__(self, nn_mod, buffer, muscle, perspective,
                 hyperparams=None, force_rms=1., 
                 c_error=1, c_positivef=0., c_negativef=0., 
                 ctrl_ftrs=slice(3,5), alpha_eint=0.05,
                 pid_interp=None,
                 pid_scalers=[1., 0.1, 0.1], **kwargs):
    
        # Linked objects:
        self.buffer, self.nn_mod, self.muscle = buffer, nn_mod, muscle
        self.hp = hyperparams
    # Force limits and scale (deprecated and should be only in the muscle object):
#             self.force_max, self.force_min, self.force_rms = force_max, force_min, force_rms
    # Controller-related
        self.ctrl_ftrs = ctrl_ftrs
        if type(pid_scalers)==torch.Tensor:
            self.pid_scalers = pid_scalers
        else:
            self.pid_scalers = torch.tensor(pid_scalers, device=self.nn_mod.device, dtype=torch.float32)
        self.alpha_eint = alpha_eint
    # Cost-related:
#             self.c_effort = c_effort
        self.c_error = c_error
        self.c_positivef, self.c_negativef = c_positivef, c_negativef
    # Perspective within a dyad:
        self.perspective = perspective
    # State-related:
        self.e_int = 0
        
        self.ftr_pos_dict = PPO5PIDAgent.ftr_pos_dict

#             if pid_interp is not None:
#                 warning.warn('PID Interpolator not set!)
        self.pid_interp = pid_interp

#             self.pid_interp = torch.exp(self.pid_scalers* rel_gains)


    def reset(self):
        self.muscle.reset()
        self.eint = 0
    
    def set_train_hyperparams(self, hyperparams):
        self.hp = hyperparams


    ###############
    # Used in Simulation (train_agents.py)
    ###############
    
    def add_experience(self, trans_tuple):
        self.buffer.push(*trans_tuple)

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
        # error_dd = r_dd - x_dd
        self.e_int = self.alpha_eint*error +\
                            (1-self.alpha_eint)*self.e_int

        observation = torch.tensor([[r, r_d, r_dd, error, error_d, self.e_int,
                                fn, 0., f_own]], dtype=torch.float32, 
                                   device=self.nn_mod.device)
        return observation 
        

    def get_force(self, state, eps=0., verbose=False):
        
        rel_gains_logprob = self._act(state, eps) #gains_logprob is a np array of [rel_gains, logprob]
        cmd = self._gain2cmd(rel_gains_logprob[0].squeeze(), state.squeeze()).cpu().numpy()
        force = float(self.muscle.get_force(cmd))

        if verbose is True:
            return force, rel_gains_logprob
        return force


    def _act(self, state_te, eps):
        # state_te is a tensor of observations
        if state_te.dim()==1:
            state_te = state_te.view(1,-1)
        dist = self.nn_mod.target_pnet(state_te, eps=eps)
#         if dist.dim()==1:
#             raise ValueError('dist: '+str(dist.shape)) #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$        
        if eps>0.: #do some exploration
            action = dist.sample()
        elif eps ==0:
            action = dist.loc
        log_prob = dist.log_prob(action).detach()
    
        return [action.detach(), log_prob]


    def _gain2cmd(self, rel_gains, state):
        err_ftrs = state[self.ctrl_ftrs]
#         abs_gains = torch.exp(self.pid_scalers* rel_gains)
        abs_gains = self.pid_interp(rel_gains)
#         print('len(rel_gains): ', len(rel_gains))
        cmd = torch.dot(err_ftrs, abs_gains)
        return cmd


    def train_step(self, current_performance):
        if len(self.buffer)>1:
            # Run one step of SGD using the whole batch data
            self.nn_mod.step(self.buffer, current_performance)

    def compute_effort_batch(self, f_batch):
        #(used in torch_trainer)
        # receives a batch of states and returns a batch of effort associated with each.
        coefs = torch.where(f_batch>0, self.c_positivef, self.c_negativef)#.to(self.nn_mod.device)
        effort_batch = coefs* (f_batch**2)
        return effort_batch
            

    ###############
    # Used in Benchmarking (benchmark_agent or the main script)
    ###############
    def update_target_nets(self):
        self.nn_mod.update_target_nets()

    ###############
    # Used in analysis
    ###############
    
    def _compute_effort(self, force):
#         scaled_force = force/self.force_rms
        coef = self.c_positivef if force>0 else self.c_negativef
        return coef*(force**2)

    def compute_utility(self, reward, force):
        return self.c_error*reward - self._compute_effort(force)
    
    
    
class PPO5PID2Agent(PPO5PIDAgent):
    ftr_pos_dict = {'r':0, 'r\'':1, 'r\"':2,
            'e':3, 'e\'':4, 'e_int':5,
            'fn':6, 'none':7, 'f_own':8}
    def __init__(self, nn_mod, buffer, muscle, perspective,
                 hyperparams=None, force_rms=1., 
                 c_error=1, c_positivef=0., c_negativef=0., 
                 ctrl_ftrs=slice(3,5), alpha_eint=0.05,
                 pid_interp=None,
                 pid_scalers=[1., 0.1, 0.1], **kwargs):
        
        super().__init__(nn_mod, buffer, muscle, perspective,
                 hyperparams=hyperparams, force_rms=force_rms, 
                 c_error=c_error, c_positivef=c_positivef, c_negativef=c_negativef, 
                 ctrl_ftrs=ctrl_ftrs, alpha_eint=alpha_eint,
                 pid_interp=pid_interp,
                 pid_scalers=pid_scalers, **kwargs)
        
        self.f_descaler = 1./nn_mod.actor_net.ftr_normalizer[-1]
        
    def _gain2cmd(self, rel_gains, state):
        s_ctrl_ftrs = state[self.ctrl_ftrs]
#         abs_gains = torch.exp(self.pid_scalers* rel_gains)
#         abs_gains = self.pid_scalers* torch.relu(rel_gains[:-1])
        abs_gains = self.pid_scalers* torch.relu(rel_gains)
#         print('len(rel_gains): ', len(rel_gains))
#         print('self.f_descaler: ', self.f_descaler)
        cmd = torch.dot(s_ctrl_ftrs, abs_gains) #+ self.pid_scalers[-1]*(rel_gains[-1]-0.3) #* self.f_descaler
        return cmd
        
        
class idle_agent(PPO5PIDAgent):
    
    def __init__(self, *args, **kwargs):
        pass
    
    def get_force(self, state, eps=0., verbose=False):
        if verbose:
            return 0., 0.
        return 0.
    
    def observe(self, environment_state):
        length = len(environment_state)
        return [0]*length
    
    def reset(self):
        pass
    
    def _compute_effort(self, force):
        return 0.

    def compute_utility(self, reward, force):
        return 0.
    
    def add_experience(self, trans_tuple):
        pass
    
    def train_step(self):
        pass
    
    def update_target_nets(self):
        pass





