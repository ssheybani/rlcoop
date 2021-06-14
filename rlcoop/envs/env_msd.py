# This is a slightly modified version of the env_spring
# intended for the pairs of agents that were pretrained on the env_spring.

import sys, os, inspect
import importlib, time

import rlcoop
from rlcoop.util import helper_funcs, trajectory_tools
from rlcoop.envs.base import PhysicalTracking

import configparser
import numpy as np
import gym
from gym import spaces

class TrackMSDM(PhysicalTracking):
    """
    Single
    """
    
    def __init__(self, config_file, dyadic=False, env_id=None, seed_=None, **kwargs):
        
        super().__init__(config_file, dyadic=dyadic, **kwargs)
        
        m1, k1, b = self.joy1_mass, self.joy1_spring, self.joy1_damper
        M = self.obj_mass
        
        self.A_mat = np.array([
            [0, 0, 1., 0],
            [0, 0, 0, 1.],
            [-k1/m1, k1/m1, -b/m1, b/m1],
            [k1/M, (-k1)/M, b/M, -b/M],
             ])
        
        self.B_mat = np.array([
            [0],
            [0],
            [1./m1],
            [0]
            ])

        self._dynamic_sys = lambda X,U,_: np.matmul(self.A_mat, X)+np.matmul(self.B_mat, U)
        self.XMi = self.svec_dict['xm']
        self.X1i = self.svec_dict['x1']
        
    def _update_state(self, f1, f2, t):
        """ Updates the dynamic states of the system of joysticks and the 
        object """
        input_vec = np.array([f1])
        self.svec = helper_funcs.rk4_step(self._dynamic_sys, self.svec, input_vec, t, self.tstep)
        return self.svec
    
    def step(self, action):
    
        """
        step function for signle agent.
        
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
        
#         return (reference, state, normal force), reward, done, _
    # action is a tuple of two forces
    # Calling step() after the episode is done will return None.
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        if self.done is True:
            print('Warning: Episode has ended. Reset the environment!')
            return None
        
        self.step_i +=1
        t = self.traj_time[self.step_i]
        r, r_d = self.traj[0][self.step_i], self.traj[1][self.step_i] #Set reference
        
        # Update joystick states
        f1, _ = action[0], _
        f2=0
        state_vec = self._update_state(f1, f2, t)
        
        xm = state_vec[self.XMi]
        x1 = state_vec[self.X1i]
        fn1 = self.joy1_spring*(xm-x1)
        fn2 = 0
        
        self.done = self.is_terminal(r,xm) # Check if the state is terminal
        
        return ([r, r_d], state_vec, [fn1, fn2, f1, f2]), self.reward(r,xm), self.done, None
    
    
class TrackMSDMDSM(PhysicalTracking):
    """
    Dyadic
    """
    def __init__(self, config_file, dyadic=True, env_id=None, seed_=None, **kwargs):
        super().__init__(config_file, dyadic=dyadic, **kwargs)
        
        m1, k1, b = self.joy1_mass, self.joy1_spring, self.joy1_damper
        m2, k2 = self.joy2_mass, self.joy2_spring
        M = self.obj_mass
        
        self.A_mat = np.array([
            [0, 0, 0, 1., 0, 0],
            [0, 0, 0, 0, 1., 0],
            [0, 0, 0, 0, 0, 1.],
            [-k1/m1, 0, k1/m1, -b/m1, 0, b/m1],
            [0, -k2/m2, k2/m2, 0, -b/m2, b/m2],
            [k1/M, k2/M, (-k1-k2)/M, b/M, b/M, -2*b/M],
             ])
        
        self.B_mat = np.array([
            [0, 0],
            [0, 0],
            [0, 0],
            [1./m1, 0],
            [0, 1./m2],
            [0, 0]
            ])
        self._dynamic_sys = lambda X,U,_: np.matmul(self.A_mat, X)+np.matmul(self.B_mat, U)
        
        self.XMi = self.svec_dict['xm']
        self.X1i = self.svec_dict['x1']
        self.X2i = self.svec_dict['x2']
        
        
    def step(self, action):
        """
        step function for the dydic environment.
        
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
        
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        if self.done is True:
            print('Warning: Episode has ended. Reset the environment!')
            return None
        
        self.step_i +=1
        t = self.traj_time[self.step_i]
        r, r_d = self.traj[0][self.step_i], self.traj[1][self.step_i] #Set reference
        
        # Update joystick states
        f1, f2 = action[0], action[1]
        state_vec = self._update_state(f1, f2, t)
        
        xm = state_vec[self.XMi]
        x1 = state_vec[self.X1i]
        x2 = state_vec[self.X2i]
        fn1 = self.joy1_spring*(xm-x1)
        fn2 = self.joy1_spring*(xm-x2)
        
        self.done = self.is_terminal(r,xm) # Check if the state is terminal
        
        return ([r, r_d], state_vec, [fn1, fn2, f1, f2]), self.reward(r,xm), self.done, None
    
    
    
class TrackMSDMDSM_minimal(TrackMSDMDSM):
    
    def step(self, action):
        """
        A minimal version of TrackMSDMDSM.
        For done and reward, None is returned.
        """
        
        self.step_i +=1
        t = self.traj_time[self.step_i]
        r, r_d = self.traj[0][self.step_i], self.traj[1][self.step_i] #Set reference
        
        # Update joystick states
        f1, f2 = action[0], action[1]
        state_vec = self._update_state(f1, f2, t)
        
        xm = state_vec[self.XMi]
        x1 = state_vec[self.X1i]
        x2 = state_vec[self.X2i]
        fn1 = self.joy1_spring*(xm-x1)
        fn2 = self.joy1_spring*(xm-x2)
        
        # self.done = self.is_terminal(r,xm) # Check if the state is terminal
        
        return ([r, r_d], state_vec, [fn1, fn2, f1, f2]), None, None, None