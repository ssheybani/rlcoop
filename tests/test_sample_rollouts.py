# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 12:01:15 2021

@author: Saber
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from pathlib import Path
import pandas as pd

import context
from context import rlcoop, DATA_PATH, CONFIG_PATH
# from rlcoop.algos import torch_trainer
from rlcoop.envs import env_msd
# from rlcoop.util import buffers #, nn_models 
# from rlcoop.agents import rl_agent, benchmark_agents, train_agents
from rlcoop.envs import gen_env_data
from rlcoop.util import helper_funcs

"""Load the data"""

n_ep_to_load = 5000



dataset_name = 'ds_run_17.59.14'
dataset_dir = DATA_PATH + dataset_name

ep_trials = []
for filename in os.listdir(dataset_dir):
    if len(ep_trials)<n_ep_to_load:
        with open(dataset_dir+'/'+filename, 'rb') as f:
            file_eps = pickle.load(f)
            ep_trials += file_eps

# ep_trials = np.asarray(ep_trials)
ts_dict = {'t':0,
            'e1':1, 'e1dot':2, 'fn1':3, 'act1':4, 'f1':5, 
            'e2':6, 'e2dot':7, 'fn2':8, 'act2':9,'f2':10}


"""
Sample rollouts

Assuming that ep_trials contains the data that one would usse as a batch, 
create the rollouts in a form that the policy gradient algorithm with TD(lambda)
updates can use.


Write two convenience function: 
    One that splits epsidoes at fixed locations.
    One that splits at given indices


"""

class Scalarizer():
    """
    Objective scalarizer. Computes the discounted return of a time-serie of 
        several objectives.
    """
    
    def __init__(self, scalarization_type, objective_specs, obj_seq_len, 
                 cost_type='quadratic',
                 cost_weight_type=None, cost_weight_pn=None):
        """
        Sets the global attributes of the scalarizer
        
        Parameters
        ----------
        
        scalarization_type: str
            Either 'basic' or 'custom'
            If 'basic', the standard scalarization formula will be used.
            If custom, the objective_specs for each objective should include
            offset, discount, obj_weight in the shape (obj_seq_len)
        
        objective_specs: sequence of shape (n_objectives)
            if scalarization_type is 'basic', the sequence should be of shape (n_objectives, 3)
                The specifications: gamma, offset, obj_weight
                    gamma: float in [0,1]
                        discount factor. can be 0.
                    offset: float
                        Will be used as: cost_type(obj) -offset
                        
                    obj_weight: float or sequence of size 2
                        if sequence of size 2, the first element is considered for 
                        the positive values of the objective and the second element 
                        for negative values.
            if scalarization_type is 'custom', the sequence should be of 
            len (n_objectives) with each element being a tuple of 3 sequences 
            of shape (obj_seq_len).
   
        obj_seq_len: int
            The length of the objective time series. Used to precompute the 
            discount matrix
            
        
        cost_type: str
            The function to apply on the input signal before scalarization.
            Either of 'quadratic', 'abs' or None.
        cost_weight_type: None or sequence of str of shape (n_obj)
            the str elements can be either 'symmetric' or 'asymmetric'
            if None, it is assumed that all are 'symmetric'
        cost_weight_pn: None or sequence of float of shape  (n_obj, 2)
            For each objective, if asymmetrical, contains the cost of a positive 
            value and a negative value, respectively.
            

    
        Returns
        ----------
        None
        """
        seq_len = obj_seq_len
        n_obj = len(objective_specs)
        self.n_obj = n_obj
        self.cost_type = cost_type
        self.cost_weight_type = cost_weight_type
        
        self.discount = np.zeros((seq_len, n_obj))
        self.obj_weight = np.zeros((seq_len, n_obj))
        self.offset = np.zeros((seq_len, n_obj))
        self.c_pos, self.c_neg = np.zeros(n_obj), np.zeros(n_obj)
        
        
        if scalarization_type=='basic':
            
            for i, obj_spec in enumerate(objective_specs):
                offset, gamma, obj_weight = obj_spec
                
                self.offset[:,i] = offset

                self.discount[:,i] = [gamma**j for j in range(seq_len)]
                self.discount[:,i] = self.discount[:,i]/np.sum(self.discount[:,i])

                self.obj_weight[:,i] = obj_weight    

        else:
            if cost_weight_type is not None and cost_weight_pn is None:
                raise ValueError
            for i, obj_spec in enumerate(objective_specs):
                offset, discount, obj_weight = obj_spec
                self.offset[:,i] = offset
                self.discount[:,i] = discount

                if cost_weight_type is not None and \
                    cost_weight_type[i]!='symm':
                    self.c_pos[i], self.c_neg[i] = cost_weight_pn[i]
                else:
                    self.obj_weight[:,i] = obj_weight
            
        # if cost_type=='quadratic':
        #     self.offset = self.offset**2
        # elif cost_type=='abs':
        #     self.offset = np.abs(self.offset)
        # else:
        #     pass
    @staticmethod
    def compute_lti_discount(A, B, x_idx, u_idx, seq_len):
        """
        Computes a discount matrix for an LTI environment

        Assuming the LTI environment is described by:
            X[k+1] = A X[k] + B U[k]
        The contribution of input u^ to x_i at time k+n is computed as:
            D x_i [k+n] / d u^ [k] = [A^n B_u^]_i
            
        Parameters
        ----------
        A: 2D numpy array of shape (n_state, n_state)
        B: 2D numpy array of shape (n_state, n_input)
        x_idx: int
            the index of the state variable to compute discount for
        u_idx: int
            the index of the input to compute discount for
        seq_len: int
            the length of the planning horizon
        
        Returns
        ----------
        discount_mat: numpy array of shape (seq_len)
        
        """
        discount_mat = np.zeros(seq_len)
        B_u = (B[:,u_idx]).reshape(-1,1)
        for n in range(seq_len):
            discount_n = np.matmul(
                np.linalg.matrix_power(A,n),
                B_u)[x_idx,0]
            discount_mat[n] = discount_n
        discount_mat = discount_mat/np.sum(discount_mat)
        return discount_mat
    
    @staticmethod
    def compute_no_discount(seq_len, no_discount_len=3):
        """
        
            
        Parameters
        ----------
        
        
        Returns
        ----------
        discount_vec: numpy array of shape (seq_len)
        
        """
        discount_vec = np.array([1 for j in range(no_discount_len)]+ \
                                                  [0. for j in range(seq_len-no_discount_len)])
        discount_vec = discount_vec/np.sum(discount_vec)
        return discount_vec
    
    
    def compute_asymm_weight_vec(self, obj_seq, obj_idx):
        return np.where(obj_seq[:,obj_idx]>0, self.c_pos[obj_idx], self.c_neg[obj_idx])
    
    
    def compute_adv(self, obj_seq, ret_terms=False):
        """
        The main function

        adv = W * discount * (obj_seq-offset)
        
        Parameters
        ----------
        obj_seq: 2D numpy array of shape (seq_len, n_objectives)
        
        Returns
        ----------
        advantage: float or sequence of size n_obj
        
        """
        if self.cost_weight_type is not None:
            # Determine the cost weight given the input sequence
            for i in range(obj_seq.shape[1]):
                if self.cost_weight_type[i] != 'symm':
                    self.obj_weight[:,i] = self.compute_asymm_weight_vec(obj_seq, i)
        if self.cost_type=='quadratic':
            obj_seq = obj_seq**2
        elif self.cost_type=='abs':
            obj_seq = np.abs(obj_seq)
        else:
            pass
        
        adv_terms = -np.sum(((obj_seq-self.offset)*self.discount)*self.obj_weight, 
                          axis=0)
        adv = np.sum(adv_terms)
        if ret_terms:
            return adv, adv_terms
        else:
            return adv
    
        
    
def _seq_to_sample(seq, inout_idx, obj_scalarizer=None):
    """
    Convenience function for turning a single sequence into a 
    dataloader-compatible sample.
    
    Parameters
    ----------
    seq: 2D numpy array of shape (seq_len, n_features)
    
    inout_idx: sequence of size 3
        Marks the indices of the following entities across the 2nd axis of seq.
        [features, action, objectives]
    
    scalarizer: function
        receives the time series of the objectives and turns them into a scalar
        as a metric for the goodness of the sample.

    Returns
    ----------
    sample: tuple of size 3, specifying feature_seq, action,  
    
    """
    pass

def get_samples_from_eps(epsides_ts, seq_len, split_point=None):
    
    """
    Converts a sequence of Episode time series into a sequence of 
    dataloader-compatible samples.
        
    Parameters
    ----------
    episodes_ts: 3D or 2D numpy array of shape (n_episodes, ts_len, n_feature)
    split_point: scalar or sequence of shape (n_episodes, n_samples)
        the starting index of the samples should be 
        the numbers should be less than ts_len-seq_len
        if None, split_idx = ts_len/seq_len
    
    Returns
    ----------
    samples: sequence of samples of shape (n_episodes, n_samples, sample_shape)
    """
    pass


dt_d = 0.02


ftr_seq_len = int(1./dt_d)
obj_seq_len = int(1./dt_d)
seq_len = ftr_seq_len + obj_seq_len


ftr_idx = slice(1,4)
act_idx = ts_dict['act1']
obj_idx = [ts_dict['e1'], ts_dict['e1dot'], ts_dict['f1']]
inout_idx = (ftr_idx, act_idx, obj_idx)

action_split_pt = ftr_seq_len




"""

Create the scalarizer function

    Here we assume we care about the rewards within the short planning horizon 
    equally. Hence, we only account for the diminishing contribution of the 
    current action to future rewards.
    
    Calculate gamma_e, gamma_ed as such:
    Find a_e, a_ed from the diagonal elements of the discrete system dynamics matrix: Ad
    Then plug a into the following formula: 
"""

# Determine offsets using the averages of the dataset
ep_trials_arr = np.array(ep_trials)
df_ts = pd.DataFrame(
    ep_trials_arr.reshape(-1,ep_trials_arr.shape[-1]),
    columns=list(ts_dict))
offset_vec_e = (df_ts['e1']**2).median() #0. #
offset_vec_ed = (df_ts['e1dot']**2).median() #0. #
offset_vec_f = (df_ts['f1']**2).median() #0. #

# Determine discounting using the lti discounting method
with open('Ad_Bd_single_Ts0.02.pickle', 'rb') as f:
    Ad_mat, Bd_mat = pickle.load(f)
discount_vec_e = Scalarizer.compute_lti_discount(Ad_mat, Bd_mat, 2, 0, obj_seq_len)
discount_vec_ed = Scalarizer.compute_lti_discount(Ad_mat, Bd_mat, 3, 0, obj_seq_len)
discount_vec_f = Scalarizer.compute_no_discount(obj_seq_len, no_discount_len=10)

# Determine objective weights using the numbers from optimal control
w_e = 2; w_ed = 46; w_fp = 1.; w_fn = 1.

obj_e_spec = (offset_vec_e, discount_vec_e, w_e)
obj_ed_spec = (offset_vec_ed, discount_vec_ed, w_ed)
obj_f_spec = (offset_vec_f, discount_vec_f, w_fp)

cost_weight_type = ['symm', 'symm', 'asymm']
cost_weight_pn = [0., 0., [w_fp, w_fn]]

scalarize_fn = Scalarizer('custom', [obj_e_spec, obj_ed_spec, obj_f_spec], 
                          obj_seq_len, cost_type='quadratic', 
                          cost_weight_type=cost_weight_type, 
                          cost_weight_pn=cost_weight_pn)


ep_len = len(ep_trials[0])
adv_arr = []
n_repeat = 10
for xep in ep_trials:
    for rep in range(n_repeat):
        split_pt = np.random.randint(ep_len-seq_len)
        seq = xep[split_pt:split_pt+seq_len]    
        obj_seq = seq[ftr_seq_len:,obj_idx]
        _, x_adv = scalarize_fn.compute_adv(obj_seq, ret_terms=True)
        adv_arr.append(x_adv)
adv_arr = np.array(adv_arr)

nbins=500; alpha=0.7
fig,ax = plt.subplots(4,1, figsize=(8,8), sharex=True)
_ = ax[0].hist(np.sum(adv_arr, axis=1), bins=nbins,label='combined')
ax[0].axvline(x=-2*np.sum(adv_arr, axis=1).std(), ls='--', color='black')
ax[1].hist(adv_arr[:,0], bins=nbins, alpha=alpha, label='adv_e')
ax[2].hist(adv_arr[:,1], bins=nbins, alpha=alpha, label='adv_ed')
_ = ax[3].hist(adv_arr[:,2], bins=nbins, alpha=alpha, label='adv_f')
# np.linspace(-1,1,100)
ax[-1].set_xlabel('scalarized advantage')
ax[-1].set_ylabel('frequency in the dataset')
ax[0].set_title('w_e = %d, w_ed = %d, w_fp = %.1f, w_fn = %.1f' %(w_e, w_ed, w_fp, w_fn))
for axij in ax:
    axij.legend()
    axij.axvline(x=0, ls='--', color='black')
    axij.set_xlim([-1.,0.5])

#Plot the distribution of e^2, e’^2, f^2 (used in computing the offset) in the random dataset
nbins=500; 
fig,ax = plt.subplots(3,1, figsize=(8,6), sharex=False)
_ = ax[0].hist(df_ts['e1']**2, bins=nbins, label='e^2')
ax[0].axvline(x=(df_ts['e1']**2).mean(), ls='--', color='black')
_ = ax[1].hist(df_ts['e1dot']**2, bins=nbins, label='ed ^2')
ax[1].axvline(x=(df_ts['e1dot']**2).mean(), ls='--', color='black')
_ = ax[2].hist(df_ts['f1']**2, bins=nbins, label='f ^2')
ax[2].axvline(x=(df_ts['f1']**2).mean(), ls='--', color='black')

ax[-1].set_xlabel('objective metric')
ax[-1].set_ylabel('frequency in the dataset')
# ax[0].set_title('w_e = %d, w_ed = %d, w_fp = %.1f, w_fn = %.1f' %(w_e, w_ed, w_fp, w_fn))
for axij in ax:
    axij.legend()
    # axij.axvline(x=0, ls='--', color='black')
    # axij.set_xlim([-1.,0.5])
    
    
# Plot the relationship between action and each advantage