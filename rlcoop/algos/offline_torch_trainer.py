from copy import deepcopy
import sys,os, inspect
# SCRIPT_DIR = os.path.realpath(os.path.dirname(inspect.getfile(inspect.currentframe())))
# PARENT_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, '..'))
# # sys.path.append(os.path.join(PARENT_PATH,'configs'))
# # sys.path.append(os.path.join(PARENT_PATH,'agents'))
# # sys.path.append(os.path.join(PARENT_PATH,'algos'))
# sys.path.append(os.path.join(PARENT_PATH,'util'))
# # sys.path.append(os.path.join(PARENT_PATH,'envs'))
import rlcoop
from rlcoop.util import helper_funcs, trajectory_tools

from rlcoop.util import buffers
from rlcoop import Transition

import numpy as np
import scipy
from numpy import asarray as narr
import torch
from torch import FloatTensor as tarr

# from collections import namedtuple
# Transition = namedtuple('Transition',
#                         ('state', 'action', 'next_state', 'reward'))

class OfflinePGTrainer():

    """
    Properties
        Linked objects:
            agent
            actor_net
            net = actor_net
            actor_opt
            optim = actor_opt
            critic_net=None
            critic_opt=None
            device
            logger
            scheduler=None
            target_vnet=None
            target_pnet

        Algorithm Parameters
            gamma => discounts #reward discounts
            lambda
            action_discounts
            nA
            nsteps

        Logging
            log_weights

            
    Methods
        Construction
            __init__
            set_agent

        Logging
            _log_metrics
            _log_params
            _log_weights

        Train step
            step
                _get_batch_experience
                    _unpack_nsteps
                _prepare_adv

        Update Target Networks
            update_target_nets
            update_target_vnet
    """

    
    def __init__(self, actor_net, actor_opt, 
                 n_actions, trust_region=0.5, 
                 nsteps=1, scheduler=None, agent=None, logger=None,
                 log_weights=(0,0), f_ftr_idx=None, device=None, gamma=1, loss_coef=100):
        
        # Linked objects:
        self.actor_net, self.actor_opt, self.agent = actor_net, actor_opt, agent
        self.net, self.optim = actor_net, actor_opt
        
        self.critic_net, self.critic_opt, self.scheduler = None, None, scheduler
        self.target_vnet = deepcopy(self.critic_net)
        self.target_pnet = deepcopy(actor_net)
        self.logger = logger
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = device
        
        # Algorithm Parameters
        self.nA = n_actions
        self.nsteps = nsteps
        # self.trust_region = trust_region
        self.loss_coef = loss_coef

        discounts = np.array([gamma**i for i in range(self.nsteps)])
        discounts = discounts/np.sum(discounts)
        self.discounts = torch.tensor(discounts, device=self.device, dtype=torch.float32)

        if self.nsteps>4:
            gamma = 0.8
            action_discounts = np.array([gamma**i for i in range(4)]+[0. for i in range(nsteps-4)])
            action_discounts = action_discounts/np.sum(action_discounts)
            self.action_discounts = torch.tensor(action_discounts, device=self.device, dtype=torch.float32)
        else:
            self.action_discounts = deepcopy(self.discounts)

        # Logging
        self.log_weights = log_weights 
        # helper variable
        self.f_ftr_idx = f_ftr_idx

    
    def set_agent(self, agent):
        self.agent = agent
        self.f_ftr_idx = agent.ftr_pos_dict['f_own']


# Train step
    def step(self, buffer, current_performance):
        
        
        """
        parameters
        ----------
        
        
        # rollout params
        ftrseq_len_c = 0.5
        ftrseq_len = seq_len_c/tstep
        
        
        n_epochs
        
        n_update_per_chunk: int
        How many updates before moving on from this set of episodes.
        default: int( episode duration/ rollout duration) = 5
        
        n_update_per_rollout: int
        default=3
        
        optim(lr)
        
        
        Algorithm:
        For epoch in range(n_epochs):
            - Shuffle the episode list
            - Separate the entire dataset into chunks of size 512, each chunk 
                 representing the batch size
            - for chunk in the chunks:
                for update_i in range(n_update_per_chunk):
                    Prepare a batch of feature sequences, curr_actions, and adv_total
                        - For each episode in the batch, randomly assign the starting 
                             point of a rollout. 
                             (operation: np.randint(ep_len-rollout_len, size=batch_size) )
                        For each rollout, prepare: 
                            (operation: select an array slice by index)
                            the feature sequence, 
                            the action sequence, 
                            the objectives sequence (e^2, e'^2, f^2)
                            
                            For update_j in range(n_update_per_rollout): 
                                (operation: make a matrix from the shifted version of the ts)
                                curr_action = action_sequence[update_j]
                                Prepare the Advantage for each objective:
                                    (operation: sum and dot)
                                    Compute the mean squared for each obj variable over the 
                                        entire batch. Alternatively, we may compute it once over 
                                        the entire dataset and only use that. 
                                    obj_ts = obj- obj_mean
                                    For e, e': Adv = np.dot(discount, obj_ts)
                                    For f: Adv = obj_ts[0] 
                                    
                                    Adv_total = w_e Adv_e + w_ed Adv_ed + w_f Adv_f
                        
                    Compute the network action to the feature sequence
                    loss = (output-curr_action)* Adv_total
                    loss.backward()
                    optim.step()
                    
                    
                    
        Implementation
        n_epochs=1
        batch_size = 512 (or 256)
        
        
        
        For epoch in range(n_epochs):
            #Shuffle the episode list.
            from random import shuffle; shuffle(ep_trials)
            #Separate the entire dataset into chunks of size 512, each chunk 
                 #representing the batch size
            # for chunk in the chunks:
            n_chunks = len(ep_trials)/batch_size
            for i in range(n_chunks)
                ep_range = slice(batch_size*i, batch_size*(i+1) )
                                 
                for update_i in range(n_update_per_chunk):
                    # Prepare a batch of feature sequences, curr_actions, and adv_total
                    
                    # For each episode in the batch, randomly assign the starting 
                    #      point of a rollout. 
                    rollout_t0s = np.random.randint(
                        ep_len-rollout_len, size=batch_size)
                    # ro_len, c_act_idx, act_idx, ftr_idx, f_idx
                    
                    ts_slices = torch.tensor([ep[t0:t0+ro_len] 
                                              for ep,t0 in zip(ep_list, rollout_t0s)])
                    te_features = ts_slices[:, 0:ftrseq_len, ftr_idx])
                    te_curr_actions = ts_slices[:, c_act_idx, act_idx]
                    
                    te_e2 = ts_slices[:, 0:ftrseq_len, e_idx]**2
                    te_ed2 = ts_slices[:, 0:ftrseq_len, ed_idx]**2
                    te_f = ts_slices[:, c_act_idx+1, f_idx]
                    te_adv_e = discount* te_e2-e2_mean)
                    te_adv_ed = discount* te_ed2-ed2_mean)
                    te_wfs = torch.where(te_f>0, self.c_positivef, self.c_negativef)#.to(self.nn_mod.device)
                    te_adv_total = w_e *te_adv_e + w_ed *te_adv_ed + te_wfs*(te_f**2 -f2_mean)
                    
                    
                    # Compute the network action to the feature sequence
                    te_output = network(te_features)
                    loss = (te_output-te_curr_actions)* te_adv_total
                    loss.backward()
                    optim.step()
                    
                    
                    
                    
        Implementation 2 with torch dataloader
            Process each episode into (feature, target, weight) samples.
            Load them into a dataloader.
            Train the policy using that.
                                            
        """
        
        if len(buffer)< self.nsteps+1:
            return
        
        self.actor_net.train();        
#         eps = 1- scipy.special.expit(-16*np.log10(-current_performance)-12)
        eps = 1.
        s_batch, a_batch, nexts_batch, r_batch, log_probs_old = self._get_batch_experience(buffer)
    
        # Calculate the Pytorch-related components of the loss function
        if self.agent.c_positivef!=0 and self.agent.c_negativef!=0 and \
                current_performance >-0.25:
            adv_rew, adv_eff, log_probs, entropy = \
                self._prepare_adv(s_batch, a_batch, nexts_batch, r_batch, eps, ret_eff=True)
            
        else:
            adv_rew, _, log_probs, entropy = \
            self._prepare_adv(s_batch, a_batch, nexts_batch, 
                                      r_batch, eps, ret_eff=False)
            
            adv_eff = torch.tensor([0.], dtype=torch.float32)
        
        advantage = adv_rew + adv_eff
        
        logprobs_diff = log_probs - log_probs_old 
        # Create a copy of the log_prob ratio, clamped between 
        logprobs_diff_c = torch.clamp(logprobs_diff, -self.trust_region, self.trust_region)
        
        surr1 = torch.exp(logprobs_diff)
        surr2 = torch.exp(logprobs_diff_c)
        ratio = torch.min(surr1, surr2)
        
        actor_loss = -self.loss_coef *(ratio*advantage).mean()#- 0.001 * entropy 
        # Alternative way of applying the trust region
#         surr1 = ratio * advantage
#         surr2 = torch.clamp(ratio, 1.0 - self.trust_region, 1.0 + self.trust_region) * advantage
#         actor_loss = -torch.min(surr1, surr2).mean()- 0.001 * entropy
        
        self.actor_opt.zero_grad();
        actor_loss.backward(); 
        torch.nn.utils.clip_grad_norm_(self.actor_net.parameters(), max_norm=0.1, norm_type='inf') #Clip gradients
        self.actor_opt.step(); 
        
        if self.scheduler is not None:
            self.scheduler.step()
            

        # Log the training stats
        self._log_weights(self.logger, self.log_weights)

        if np.random.rand()<0.005:
            # compute metrics
            actor_loss_m = actor_loss.item()
            entropy_m = entropy.item()
            adv_m = torch.norm(adv_rew.detach()).item()/16
            logprob_m = log_probs.detach().median().item()
            
#             eff_adv_ratio_m = torch.mean( (torch.abs(adv_eff)>torch.abs(adv_rew)).float() 
#                                         ).item()
            if adv_eff.shape[0]!=1:
                eff_rew_ratios = torch.log2(torch.abs(adv_eff/adv_rew)).numpy()
                eff_rew_ratios = np.nan_to_num(eff_rew_ratios, nan=0.0, posinf=100, neginf=-100)
                ea_skew, eff_adv_ratio_m, ea_std = scipy.stats.skewnorm.fit(eff_rew_ratios)
            else:
                ea_skew, eff_adv_ratio_m, ea_std = 0., 0., 0.
    
            print('# Agent ', self.agent.perspective+1, 
                  'Actor loss = %.3f'%actor_loss_m,
                  'Error Advantage Norm = %.3f'% adv_m,
                  'Log Effort / Reward mean, std, skew = %.2f'% eff_adv_ratio_m, '%.2f'%ea_std,'%.2f'%ea_skew, 
                 'Log probs Median =  %.3f'% logprob_m,
                 'Entropy = %.3f'% entropy_m)
            self._log_metrics(self.logger, actor_loss_m, adv_m, entropy_m, logprob_m, eff_adv_ratio_m)


    def get_batch_experience(self, buffer):
        return self._get_batch_experience(buffer)
    
    def _get_batch_experience(self, buffer):
        batch_size_ = min(len(buffer), self.agent.hp.batch_size)
        transitions = buffer.sample_nstep(batch_size_, self.nsteps)
        # returns a list of lists
        
        s_batch, a_batch, nexts_batch, r_batch, log_probs_old = tuple(
            zip(*map(self._unpack_nsteps, transitions)))
        
        s_batch = torch.cat(s_batch, dim=0)
        a_batch = torch.cat(a_batch, dim=0)
        a_batch = torch.matmul(a_batch, self.action_discounts.view(-1,1)).squeeze()
        
        nexts_batch = torch.cat(nexts_batch, dim=0)
        r_batch = torch.cat(r_batch, dim=0)
        log_probs_old = torch.cat(log_probs_old, dim=0)
        
        return s_batch, a_batch, nexts_batch, r_batch, log_probs_old
    
  
    def _unpack_nsteps(self, nsteps_trans): 
        
        tran_nsteps = Transition(*zip(*nsteps_trans))                   
        
        reward_n = torch.cat(tran_nsteps.reward, dim=0).unsqueeze(0)
        state_n = torch.cat(tran_nsteps.state, dim=0).unsqueeze(0)
        nexts_n = torch.cat(tran_nsteps.next_state, dim=0).unsqueeze(0)
        
        get_a = lambda lst: lst[0]
        action_n = torch.cat(tuple(map(get_a, tran_nsteps.action))).T.unsqueeze(0)
    
        return (state_n,
                action_n,
                nexts_n,
                reward_n,
                tran_nsteps.action[0][1].view(1,-1))

    
    def _prepare_adv(self, sn_batch, a_batch, nextsn_batch, rn_batch, eps, ret_eff=True):
        
        if self.agent is None:
            raise ValueError('model.agent must be set.')
        
        # rn_batch is [nsamples, nsteps, 2]
        # sn_batch is [nsamples, nsteps, 9]
        
        rdiff_batch = rn_batch[:,:,1]-rn_batch[:,:,0]
        adv_rew = torch.mm(rdiff_batch, self.discounts.view(-1,1))#torch.dot(self.discounts, rdiff_batch) #should be [nsamples, 1]  #@@@shape mismatch
        
        dists = self.actor_net(sn_batch[:,0,:], eps=eps)
        entropy = dists.entropy().mean()
        log_probs = dists.log_prob(a_batch) 
        log_probs = torch.clamp(log_probs, min=-100, max=100.)
        log_probs[log_probs != log_probs] = -100.
        
        if ret_eff is False:
            return adv_rew, None, log_probs, entropy
        else:
            eff1_batch = self.agent.compute_effort_batch(sn_batch[:, :, self.f_ftr_idx])#@@@shape mismatch
            
            eff2_batch = torch.cat((eff1_batch[:,1:], 
                 self.agent.compute_effort_batch(nextsn_batch[:, -1, self.f_ftr_idx]).unsqueeze(1)), dim=1)
                
            effdiff_batch = eff2_batch-eff1_batch
            adv_eff = -torch.mm(effdiff_batch, self.discounts.view(-1,1))
        
            return adv_rew, adv_eff, log_probs, entropy



    # Logging
    def _log_metrics(self, logger, actor_loss, adv, entropy, logprob, eff_adv_ratio):
        logger.actorloss_ts.append(actor_loss)
        logger.entropy_ts.append(entropy)
        logger.adv_ts.append(adv)
        logger.logprobs_ts.append(logprob)
        logger.eff_adv_ratio_ts.append(eff_adv_ratio)

    def _log_weights(self, logger, weight_specs):
        # self: the ActorCritic Network module; has actor_net and critic_net as attributes
        # logger: the object that is written on.
        # weight_specs: a tuple of size 2, specifying the number of layers from the actor 
        #               and the critic to record, respectively.
        ac_nweights = weight_specs[0]; 
        if ac_nweights>0: 
            ac_layers = [module[1] for module in self.actor_net.named_modules()]
            for l in range(ac_nweights):
                 logger.actor_weight_ts[l].append(
                     deepcopy(ac_layers[-1-l].weight.detach().numpy()))
        
        cr_nweights = weight_specs[1]
        if cr_nweights>0: 
            cr_layers = [module[1] for module in self.critic_net.named_modules()]
            for l in range(cr_nweights):
                 logger.critic_weight_ts[l].append(
                     deepcopy(cr_layers[-1-l].weight.detach().numpy()))
        return
    
    def _log_params(self, logger, param_specs):
        # quick adaptation of _log_weights for networks that only have free parameters
        ac_nparams = param_specs[0]; 
        if ac_nparams>0: 
            ac_params = [param[1] for param in self.actor_net.named_parameters()]
            for l in range(ac_nparams):
                logger.actor_weight_ts[l].append(
                     deepcopy(ac_params[l].detach().numpy()))
        return
        

    def update_target_nets(self):
#         self.target_vnet.load_state_dict(self.critic_net.state_dict())
#         self.target_vnet.eval()
        
        self.target_pnet.load_state_dict(self.actor_net.state_dict())
        self.target_pnet.eval()
