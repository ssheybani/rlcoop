import sys, os, inspect
import importlib, time

import rlcoop
from rlcoop.util import helper_funcs, trajectory_tools

import configparser
import numpy as np
import gym
from gym import spaces


class TrackMSD():

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
    def __init__(self, config_file, dyadic=False, env_id=None, seed_=None, **kwargs):

        
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
        # self.p1, self.p2, self.x, self.v = self.svec[0], self.svec[2], self.svec[4], self.svec[5]
        # self.a = 0.
        # p1, p2: joystic positions; x,v,a: the position (and v,a) of the object
        self.done = False
        self.traj_creator = trajectory_tools.Trajectory(self.tstep, seed_=seed_)
        self.traj_time, self.traj = None, None
        self._max_episode_steps = int(self.duration/self.tstep)

        
        k1, k2 = self.joy1_spring, self.joy2_spring
        # m1, m2, MM = self.joy1_mass, self.joy2_mass, self.obj_mass
        
        if dyadic:
            # correct the matrices, based on the new task model. @@@@@@@@@@@@
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
        else:
            # correct the matrices to the single version@@@@@@@@@@@@@
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
        # if dyadic: self.svec: np.array([p1, p1dot, p2, p2dot, x, v])
        # if single: self.svec: np.array([p1, p1dot, x, v])
        input_vec = np.array([f1,f2])
        
        # self.a = self._dynamic_sys(self.svec, input_vec, t)[-1]
        
        self.svec = helper_funcs.rk4_step(self._dynamic_sys, self.svec, input_vec, t, self.tstep)
        return self.svec
        # self.x, self.v = self.svec[4], self.svec[5]
        
        # return self.svec[0], self.svec[2],  self.x, self.v, self.a
        

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
        """
        
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
        r, r_d, r_dd = self.traj[0][self.step_i], self.traj[1][self.step_i] #Set reference
        # r_dd = self.traj[2][self.step_i]
        self.done = self.is_terminal(r) # Check if the state is terminal
        
        # Update joystick states
        f1, f2 = action[0], action[1]
        
        # if dyadic:
        #     state_vec = self._update_state(f1, f2, t)
        #     p1,p2, x,v,a = self._update_state(f1, f2, t)
        #     fn2 = 0
        # else:
        #     f2=0
        #     p1,p2, x,v,a = self._update_state(f1, f2, t)
        #     fn2 = self.joy2_spring*(p2-x)
        state_vec = self._update_state(f1, f2, t)
        
        fn1 = -self.joy1_spring*(p1-x)
        fn2 = -self.joy2_spring*(p1-x)
        
        return ([r, r_d], state_vec, [fn1, fn2, f1, f2]), self.reward(r,x), self.done, None
    
    
    def render(self):
        # Not schedulled to be implemented at this phase. 
        raise NotImplementedError
        
    def close(self):
        pass