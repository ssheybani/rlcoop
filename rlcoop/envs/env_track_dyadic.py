# This is a slightly modified version of the env_spring
# intended for the pairs of agents that were pretrained on the env_spring.

import sys, os, inspect
import importlib, time

import rlcoop
from rlcoop.util import helper_funcs, trajectory_tools

import configparser
import numpy as np
import gym
from gym import spaces

    
class PhysicalTrackDyad_v3():
# With the springs between the object and the controllers

#     allowable_keys = ('obj_mass', 'obj_fric', 'tstep', 'duration', 'failure_error',
#                       'f_bound', 'max_ref', 'max_freq')

    # internal methods
    def __init__(self, config_file, env_id=None, seed_=None, **kwargs):
        # Returns an instantiated dyadic game environment. 
        # Given that the environment is dynamic, the current time step can be queried
        # from variable "step_i".
        
        # env_id can be either 'soft' or 'hard'.
        # 'soft' corresponds to soft force constraints, i.e. the task doesn't end 
        # (the carried object does not break) if normal force range is violated.
        # 'hard' is the other case. => NOT IMPLEMENTED.
        
        
        # There are 2 mechanisms to set/modify env parameters: config file, kwargs.
        # config file is the address to the INI config file.
            # - The file does not need to contain all parameters but only those that are modified
            # - Sections are not important
            # - Only scalars and strings are allowed as values.
        
        # Read params from config file.
        Config = configparser.ConfigParser()
        Config.read(config_file) # read the config file.
        
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
                             ((2*np.pi*self.max_freq)**2)*self.max_ref * 2,
                             (self.max_ref+self.max_err) * 2,
                             np.finfo(np.float32).max,
                             np.finfo(np.float32).max,
                             self.force_max,
                             np.finfo(np.float32).max],
                            dtype=np.float32)
        
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)
        
        self.nS = np.nan
        # Initialize state variables
        self.step_i =0
        self.svec = np.zeros(6, dtype=float)
        self.p1, self.p2, self.x, self.v = self.svec[0], self.svec[2], self.svec[4], self.svec[5]
        self.a = 0.
        # p1, p2: joystic positions; x,v,a: the position (and v,a) of the object
        self.done = False
        self.traj_creator = trajectory_tools.Trajectory(self.tstep, seed_=seed_)
        self.traj_time, self.traj = None, None
        self._max_episode_steps = int(self.duration/self.tstep)


        k1, k2 = self.joy1_spring, self.joy2_spring
        m1, m2, MM = self.joy1_mass, self.joy2_mass, self.obj_mass
        self.A_mat = np.array([
            [0., 1., 0., 0., 0., 0.],
            [-k1/m1, 0., 0., 0., k1/m1, 0.],
            [0., 0., 0., 1., 0., 0.],
            [0., 0., -k2/m2, 0., k2/m2, 0.],
            [0., 0., 0., 0., 0., 1.],
            [k1/MM, 0., k2/MM, 0., (-k1-k2)/MM, -self.obj_fric/MM]
            ])
        
        self.B_mat = np.array([
            [0.,0.],
            [1./m1,0.],
            [0.,0.],
            [0.,-1/m2],
            [0.,0.],
            [0.,0.]
            ])

        self._dynamic_sys = lambda X,U,_: np.matmul(self.A_mat, X)+np.matmul(self.B_mat, U)
#         self.spring_k = spring_k
        self.ftr_pos_dict = {'r':0, 'r\'':1, 'r\"':2,
                'x':3, 'x\'':4, 'x\"':5,
                'fn1':6, 'fn2':7, 'f1':8, 'f2':9}
        
    
    def get_time(self):
        return self.step_i
    
    def get_episode_duration(self):
        return self.duration
    
#     x,v,a, p1, p2 = self._update_states(f1, f2, t)
    def _update_state(self, f1, f2, t):
        # Updates the dynamic states of the system of joysticks and the object
        # self.svec: np.array([p1, p1dot, p2, p2dot, x, v])
        input_vec = np.array([f1,f2])
        
        self.a = self._dynamic_sys(self.svec, input_vec, t)[-1]
        
        self.svec = helper_funcs.rk4_step(self._dynamic_sys, self.svec, input_vec, t, self.tstep)
        self.x, self.v = self.svec[4], self.svec[5]
        
        return self.svec[0], self.svec[2],  self.x, self.v, self.a
        

    def _update_fn(self, f1, f2):
        raise NotImplementedError
        
    
    def reward(self, r,x):
        error = (r-x)/self.max_ref
        reward_ = -(error**2) #-100*abs(fn) #
        
        if reward_ <-10.:
            return -10.
        return reward_
    
    
    # interface functions
    
#     def seed(self, seed_):
#         # Make sure the all the random generators in the environment use this seed.
#         np.random.seed(seed_)
#         self.traj_creator.s
#         # Do not use Python's native random generator for consistency
#         raise NotImplementedError
        
    def reset(self, renew_traj=True, max_freq='max'):
        # Resets the initial state and the reference trajectories. 
        # Call this function before a new episode starts.
#         return initial observations
        #max_freq can be 'max' or 'random'
        
        if max_freq=='max':
            traj_max_f = self.max_freq
        elif max_freq=='random':
            traj_max_f = self.max_freq * np.random.rand()
        elif isinstance(max_freq, (float)):
            traj_max_f = max_freq
        else:
            raise ValueError
        if (renew_traj is False and self.traj is None) or \
        (renew_traj is True):
            # generate a traj
            self.traj_time, self.traj = self.traj_creator.generate_random(self.duration, \
                n_traj=1, max_amp=self.max_ref, traj_max_f=traj_max_f, rel_amps=None, fixed_effort=False, \
                obj_mass=self.obj_mass, obj_fric=self.obj_fric, n_deriv=2, ret_specs=False)
        
        self.x, self.v, self.a  = self.traj[0][0], self.traj[1][0], self.traj[2][0]
        # self.p1, self.p2 = self.x, self.x
        self.svec = np.array([self.x, self.v, self.x, self.v, self.x, self.v])
        self.step_i = 0
        self.done = False
        
        return [self.x, self.v, self.a, self.x, self.v, self.a, 0., 0., 0., 0.]
            
    
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
        r, r_d, r_dd = self.traj[0][self.step_i], self.traj[1][self.step_i], self.traj[2][self.step_i] #Set reference
        
        self.done = self.is_terminal(r) # Check if the state is terminal
        
        # Update joystick states
        f1, f2 = action[0], action[1]
        
        p1,p2, x,v,a = self._update_state(f1, f2, t)
        
        fn1 = -self.joy1_spring*(p1-x)
        fn2 = self.joy2_spring*(p2-x)

        return [r, r_d, r_dd, x, v, a, fn1, fn2, f1, f2], self.reward(r,x), self.done, None
    
    
    def render(self):
        # Not schedulled to be implemented at this phase. 
        raise NotImplementedError
        
    def close(self):
        pass