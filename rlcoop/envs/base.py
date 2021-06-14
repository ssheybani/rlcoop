import sys, os, inspect
import importlib, time

import rlcoop
from rlcoop.util import helper_funcs, trajectory_tools

import configparser
import numpy as np
import gym
from gym import spaces


class PhysicalTracking():

    """
    Base class for trackings experiments 
    
    A moving target is to be tracked where the control force is exerted to a 
    mechanical system of spring-damper-mass.
    
    Attributes
    ----------
        System dynamics : function handle
            function receiving X, U and returning X'
            Or A_mat, B_mat.
            Or the constants for the masses, springs, dampers. 
        Config related: float
            max_freq, force_max, ...
        Run-related: float
            duration, tstep, time1, traj, step_i
        ftr_pos_dict: dict
    
    Methods
    ----------        
        Construction:
            __init__
            reset
            close
        
        Used in Simulation (train_agents.py):
            step
                _update_state
            reward
            is_terminal
            
        render
        
    """

#     allowable_keys = ('obj_mass', 'obj_fric', 'tstep', 'duration', 'failure_error',
#                       'f_bound', 'max_ref', 'max_freq')

    # internal methods
    def __init__(self, config_file, dyadic=True, env_id=None, seed_=None, **kwargs):

        
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
        
        # obs_high = np.array([self.max_ref * 2,
        #                      2*np.pi*self.max_freq*self.max_ref * 2,
        #                      ((2*np.pi*self.max_freq)**2)*self.max_ref * 2,
        #                      (self.max_ref+self.max_err) * 2,
        #                      np.finfo(np.float32).max,
        #                      np.finfo(np.float32).max,
        #                      self.force_max,
        #                      np.finfo(np.float32).max],
        #                     dtype=np.float32)
        
        # self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)
        
        self.dyadic = dyadic 
        
        if dyadic:
            self.nA = 2
            self.nS = 6
            self.svec_dict = {'x1':0, 'x2':1, 'xm':2,
                         'dx1':3, 'dx2':4, 'dxm':5}
            
        else:
            self.nA = 1
            self.nS = 4
            self.svec_dict = {'x1':0, 'xm':2,
                         'dx1':3, 'dxm':5}
        self.fvec_dict = {'fn1':0, 'fn2':1, 'f1':2, 'f2':3}

        # Initialize state variables
        self.step_i =0
        self.svec = np.zeros(self.nS, dtype=float)
        self.done = False
        self.traj_creator = trajectory_tools.Trajectory(self.tstep, seed_=seed_)
        self.traj_time, self.traj = None, None
        self._max_episode_steps = int(self.duration/self.tstep)

        # default A_mat,B_mat        
        self.A_mat = np.zeros((self.nS, self.nS))
        self.B_mat = np.zeros((self.nS, self.nA))
        self._dynamic_sys = lambda X,U,_: np.matmul(self.A_mat, X)+np.matmul(self.B_mat, U)

        # k1, k2 = self.joy1_spring, self.joy2_spring # for outputting normal force
    
    def get_time(self):
        return self.step_i
    
    def get_episode_duration(self):
        return self.duration
    
#     x,v,a, p1, p2 = self._update_states(f1, f2, t)
    def _update_state(self, f1, f2, t):
        # Updates the dynamic states of the system of joysticks and the object
        # Reimplement for single
        input_vec = np.array([f1,f2])
        
        self.svec = helper_funcs.rk4_step(self._dynamic_sys, self.svec, input_vec, t, self.tstep)
        return self.svec
        

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
                obj_mass=self.obj_mass, obj_fric=0., n_deriv=2, ret_specs=False)
        
        r0, rdot0 = self.traj[0][0], self.traj[1][0]
        self.svec = np.zeros(self.nS)
        X1i, XMi = self.svec_dict['x1'], self.svec_dict['xm']
        dX1i, dXMi = self.svec_dict['dx1'], self.svec_dict['dxm']
        self.svec[X1i], self.svec[XMi] = r0, r0
        self.svec[dX1i], self.svec[dXMi] = rdot0, rdot0
        if self.dyadic:
            X2i, dX2i = self.svec_dict['x2'], self.svec_dict['dx2']
            self.svec[X2i] = r0
            self.svec[dX2i] = rdot0
            
        self.step_i = 0
        self.done = False
        
        return ([r0, rdot0], self.svec, [0., 0., 0., 0.])
        
            
    
    def is_terminal(self, r,x):
        if self.step_i== self._max_episode_steps-1:
            return True 
        # If the positional error is too large, end the episode
        if r-x > self.max_err:
            return True
        return False
        
        
    def step(self, action):
        """
        Note: reimplement for each of the single and dyadic task separately
        (This decision is for the sake of faster simulation)
        
        Parameters
        ----------
            action : tuple of size 2
                f1, f2

        Returns
        -------
            outcomes: tuple of size 4
            (reference, state, forces), reward, done, _
            Calling step() after the episode is done will return None.

        """
#         return 
    # 
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        if self.done is True:
            print('Warning: Episode has ended. Reset the environment!')
            return None
        
        self.step_i +=1
        t = self.traj_time[self.step_i]
        r, r_d = self.traj[0][self.step_i], self.traj[1][self.step_i] #Set reference
        # r_dd = self.traj[2][self.step_i]
        
        raise NotImplementedError()
        
        # # Update joystick states
        # f1, f2 = action[0], action[1]
        # state_vec = self._update_state(f1, f2, t)
        
        # fn1, fn2 = 0., 0.
        # x = state_vec[self.svec_dict['xm']]
        
        # self.done = self.is_terminal(r,x) # Check if the state is terminal
        
        # return ([r, r_d], state_vec, [fn1, fn2, f1, f2]), self.reward(r,x), self.done, None
    
    
    def render(self):
        # Not schedulled to be implemented at this phase. 
        raise NotImplementedError
        
    def close(self):
        pass