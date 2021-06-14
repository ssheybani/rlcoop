import numpy as np
import matplotlib.pyplot as plt
import pickle
import importlib
import argparse
import functools
import sys,os


import context
from context import rlcoop, DATA_PATH, CONFIG_PATH
# from rlcoop.algos import torch_trainer
from rlcoop.envs import env_msd

def gen_episode(env, action2k, refresh_k_steps, renew_traj=True):
    
    # Unwrap some variables
    w_inv, b_inv = action2k
    sdict = env.svec_dict
    fdict = env.fvec_dict
    X1i, X2i, XMi = sdict['x1'],sdict['x2'],sdict['xm']
    dX1i, dX2i, dXMi = sdict['dx1'],sdict['dx2'],sdict['dxm']
    Fn1i, Fn2i, F1i, F2i, = fdict['fn1'], fdict['fn2'], fdict['f1'], fdict['f2']
    
    (rrdot, q, fvec) = env.reset(renew_traj=renew_traj, max_freq='max')    
    ts_data = []
    
    while env.get_time() < env._max_episode_steps-1:
        t = env.get_time()
        
        # Observe
        r0, rdot0 = rrdot
        r_vec = np.array([r0, r0, rdot0, rdot0])
        x1_view = np.array([q[X1i], q[XMi], q[dX1i], q[dXMi]])
        x2_view = np.array([q[X2i], q[XMi], q[dX2i], q[dXMi]])
        e1_vec = r_vec - x1_view
        e2_vec = r_vec - x2_view
        
        # Get forces
        if t%refresh_k_steps==0:
            actions = np.random.uniform(low=-2, high=2, size=2).reshape(2,1)
            K_fb = actions*w_inv +b_inv
        f1 = - np.dot(K_fb[0,:], e1_vec)
        f2 = - np.dot(K_fb[1,:], e2_vec)
        
        # Collect the features for recording
        e1,e1d = e1_vec[1], e1_vec[3]
        e2,e2d = e2_vec[1], e2_vec[3]
        fn1, fn2 = fvec[Fn1i], fvec[Fn2i]

        ts_data.append([t*env.tstep, 
                        e1, e1d, fn1, actions[0], f1, 
                        e2, e2d, fn2, actions[1], f2])
        # Environment step
        (rrdot, q, fvec), _, _, _ = env.step([f1, f2])

    return ts_data