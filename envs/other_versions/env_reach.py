import sys, os
import importlib, time

lib_dir = 'util/'
default_config = 'configs/env_old_config.ini'
sys.path.append(lib_dir)

import configparser
from helper_funcs import rk4_step
import trajectory_tools
# importlib.reload(trajectory_tools)
from trajectory_tools import Trajectory
import numpy as np
import gym
from gym import spaces

class PhysicalReach():
    
#     allowable_keys = ('obj_mass', 'obj_fric', 'tstep', 'duration', 'failure_error',
#                       'f_bound', 'max_ref', 'max_freq')

    # internal methods
    def __init__(self, env_id=None, seed_=None, config_file=None, **kwargs):
        # Returns an instantiated dyadic game environment. 
        # Given that the environment is dynamic, the current time step can be queried
        # from variable "step_i".
        
        # env_id can be either 'soft' or 'hard'.
        # 'soft' corresponds to soft force constraints, i.e. the task doesn't end 
        # (the carried object does not break) if normal force range is violated.
        # 'hard' is the other case.
        
        
        # There are 2 mechanisms to set/modify env parameters: config file, kwargs.
        # config file is the address to the INI config file.
            # - The file does not need to contain all parameters but only those that are modified
            # - Sections are not important
            # - Only scalars and strings are allowed as values.
        
        # Read params from config file.
        Config = configparser.ConfigParser()
        Config.read(default_config) # read the default config file.
        
        for section in Config.sections():
            for key in Config.options(section):
                if True: #key in self.allowable_keys:
                    val = Config.get(section, key)
                    try:
                        val = float(val)
                    except:
                        pass
                    setattr(self, key, val)       
        
        if config_file!=None:
            Config = configparser.ConfigParser()
            Config.read(config_file) # read the default config file.
        
            for section in Config.sections():
                for key in Config.options(section):
                    if True: #key in self.allowable_keys:
                        val = Config.get(section, key)
                        try:
                            val = float(val)
                        except:
                            pass
                        setattr(self, key, val)
        
        # Read the kwargs and override params from config file.
        for key in kwargs:
            if True: #key in self.allowable_keys:
                setattr(self, key, kwargs[key])

    
    
    
        self.max_err = self.max_ref*self.failure_error
        
        act_high = np.array([self.force_max, self.force_max])
        self.action_space = spaces.Box(-act_high, act_high, dtype=np.float32)
        
        obs_high = np.array([self.max_ref * 2,
                             2*np.pi*self.max_freq*self.max_ref * 2,
                             (self.max_ref+self.max_err) * 2,
                             np.finfo(np.float32).max,
                             self.force_max,
                             np.finfo(np.float32).max],
                            dtype=np.float32)
        
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)
        
        self.nS = 4
        # Initialize state variables
        self.step_i =0
        self.x, self.v, self.f1_old, self.f2_old = 0., 0., 0., 0.
        self.done = False
        self.traj_creator = Trajectory(self.tstep, seed_=seed_)
        self.traj_time, self.traj = None, None
        self._max_episode_steps = int(self.duration/self.tstep)
        
        
    
    def get_time(self):
        return self.step_i
    
    def get_episode_duration(self):
        return self.duration
    
    def _dynamic_sys(self, x, u, t=0):
        return np.array([x[1],(-self.obj_fric*x[1]+x[2])/self.obj_mass, u])
    
    def _update_state(self, net_f, net_df, t, tstep):

        obj_state = [self.x, self.v, net_f]
        self.x, self.v, self.net_f = rk4_step(self._dynamic_sys, obj_state, net_df, t, tstep)
    

    def _update_fn(self, f1, f2):
        
        # Returns normal force and its first derivative
        
        raise NotImplementedError
    
    
    def reward(self, r, x, t):
        err = (-((r-x)/self.max_ref)**2)
        return err
        
    
    # interface functions
    
#     def seed(self, seed_):
#         # Make sure the all the random generators in the environment use this seed.
#         np.random.seed(seed_)
#         self.traj_creator.s
#         # Do not use Python's native random generator for consistency
#         raise NotImplementedError
        
    def reset(self, renew_traj=True):
        # Resets the initial state and the reference trajectories. 
        # Call this function before a new episode starts.
#         return initial observations
        if (renew_traj is False and self.traj is None) or \
        (renew_traj is True):
            # generate a traj
            self.traj_time, self.traj = self.traj_creator.generate_random(self.duration, \
                n_traj=1, max_amp=self.max_ref, traj_max_f=self.max_freq, rel_amps=None, fixed_effort=False, \
                obj_mass=self.obj_mass, obj_fric=self.obj_fric, n_deriv=1, ret_specs=False)
        
        self.x, self.v  = (2*np.random.rand()-1)*self.max_ref, 0. #self.traj[0][0], self.traj[1][0]
        self.step_i = 0
        self.f1_old, self.f2_old = 0., 0.
        self.done = False
        
        return [self.x, self.v, self.x, self.v, 0., 0.]
            
    
    def is_terminal(self, r):
        if self.step_i== self._max_episode_steps-1:
            return True 
        # If the positional error is too large, end the episode
        if r-self.x > self.max_err:
            return True
        return False
        
        
    def step(self, action):
#         return (reference, state, normal force), reward, done, _
    # action is a tuple of two forces
    # Calling step() after the episode is done will return None.
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        if self.done is True:
            print('Warning: Episode has ended. Reset the environment!')
            return None
        
        self.step_i +=1
        t = self.traj_time[self.step_i]
        r, r_dot = self.traj[0][self.step_i], self.traj[1][self.step_i] #Set reference
        
        self.done = self.is_terminal(r) # Check terminal
        
        # Update object state
        f1, f2 = action[0], action[1]
        net_f = f1-f2
        net_df = net_f - (self.f1_old-self.f2_old)
        self._update_state(net_f, net_df, t, self.tstep)
        
        # Calculate normal force and its derivative
        fn = min(f1, f2); fn_old = min(self.f1_old, self.f2_old);
        fn_dot = (fn-fn_old)/self.tstep 
        
        self.f1_old = f1; self.f2_old = f2;
        
#         self.done = self.is_terminal() # Check terminal due to new positional error 
        
        return [r, r_dot, self.x, self.v, fn, fn_dot], self.reward(r,self.x, t), self.done, None
    
    
    def render(self):
        # Not schedulled to be implemented at this phase. 
        raise NotImplementedError
        
    def close(self):
        pass