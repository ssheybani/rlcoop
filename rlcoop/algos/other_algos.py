from copy import deepcopy
import buffers
import numpy as np
import scipy
from numpy import asarray as narr
import torch
from torch import FloatTensor as tarr

from collections import namedtuple
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

#__________________________________________
# Old versions
class TorchTrainer():
    # Base class for agent.rl, agent.predictive
    # Attributes: net, optim, criterion
    # imports: none
            
    def __init__(self, net, optim, criterion, agent):
        self.net = net # irrelevant for ACTrainer
        self.optim = optim
        self.criterion = criterion # irrelevant for ACTrainer
        self.agent = agent
    
    def set_agent(self, agent):
        self.agent = agent
        
    def _prepare_pred_target(self, buffer):
        # Prepare a batch of predictions and a batch of targets, using the buffer and the actions in the agent.
        # For RL: input: agent.buffer, agent.get_f_for_all_actions(state)
        # For supervised: input: agent.buffer, 
        # pred, target should be returned as torch variables with gradients
        #return pred, target
        raise NotImplementedError
        
    def step(self, buffer):
        prediction, target = self._prepare_pred_target(buffer)
        self.optim.zero_grad() #zero all of the gradients
        loss = self.criterion(prediction, target)
        loss.backward()# Backward pass: compute gradient of the loss with respect to model parameters.
        self.optim.step()# Update model parameters

        
class DQNTrainer(TorchTrainer):
    # imports: Transition, 
    # deepcopy from copy, numpy as np, np.asarray as narr
    # torch.FloatTensor as tarr
    
    def __init__(self, net, optim, criterion, agent=None):
        super().__init__(net, optim, criterion, agent)
        self.target_qnet = deepcopy(net)
    
    
    def _prepare_pred_target(self, buffer):
        if self.agent is None:
            raise ValueError('model.agent must be set.')
        batch_size_ = min(len(buffer), self.agent.hp.batch_size)
        transitions = buffer.sample(batch_size_)
        batch = buffers.Transition(*zip(*transitions))
        # batch.state is a tuple. each entry is one sample.
        # each sample is a list of the feature vars.
        # For batch.action, batch.reward, the sample is a float.
        sa_batch = np.concatenate( (narr(batch.state), 
                                    narr(batch.action)[:,np.newaxis]),axis=1)
        sa_batch = tarr(sa_batch)
        q_pred = self.net(sa_batch).view(-1)
        
        #calculate target qvals
        reward_batch = tarr(narr(batch.reward))
        next_s_batch_narr = narr(batch.next_state)
        force0_batch_narr, force1_batch_narr = \
        self.agent.get_force_batch(next_s_batch_narr)
        next_sa0_batch_narr = np.concatenate( (next_s_batch_narr, force0_batch_narr),axis=1)
        next_sa1_batch_narr = np.concatenate( (next_s_batch_narr, force1_batch_narr),axis=1)
        q0 = self.net(tarr(next_sa0_batch_narr))
        q1 = self.net(tarr(next_sa1_batch_narr))
        target_qvals = reward_batch+ torch.cat((q0, q1)).max()
            
        return q_pred, target_qvals
    
    
    def update_target_qnet(self):
        self.target_qnet.load_state_dict(self.net.state_dict())
        self.target_qnet.eval()
    
class SupervisedTrainer(TorchTrainer):
    def _prepare_pred_target(self, buffer):
        pass
    
    
class ACTrainer(TorchTrainer):
    # imports: Transition, 
    # deepcopy from copy, numpy as np, np.asarray as narr
    # torch.FloatTensor as tarr
    
    def __init__(self, actor_net, critic_net, actor_opt, critic_opt, criterion, scheduler=None, agent=None, logger=None):
        super().__init__(actor_net, actor_opt, criterion, agent)
        self.actor_net = actor_net
        self.critic_net = critic_net
        self.actor_opt = actor_opt
        self.critic_opt = critic_opt
        self.scheduler = scheduler
        self.target_vnet = deepcopy(critic_net)
        self.logger = logger
    
    
    def _prepare_pred_target(self, buffer):
        # This function 
        if self.agent is None:
            raise ValueError('model.agent must be set.')
        batch_size_ = min(len(buffer), self.agent.hp.batch_size)
        transitions = buffer.sample(batch_size_)
        
        s_batch = torch.zeros(batch_size_, len(transitions[0][0]))#, dtype=torch.float, device=device)
        a_batch = torch.zeros(batch_size_)
        nexts_batch = torch.zeros(batch_size_, len(transitions[0][2]))
        r_batch = torch.zeros(batch_size_)
        
        for i, item in enumerate(transitions):
            s_batch[i,:], a_batch[i], nexts_batch[i,:], r_batch[i] = \
            torch.tensor(item[0]), item[1], torch.tensor(item[2]), item[3]
        
        
        v_pred = self.critic_net(s_batch)#.view(-1)
        v_target = r_batch + self.agent.hp.gamma* self.target_vnet(nexts_batch).detach()

        # With the actor network
        dists = self.actor_net(s_batch)
        entropy = 0.; log_probs = torch.zeros(batch_size_)
        
        entropy = dists.entropy().mean()
        log_probs = dists.log_prob(a_batch) 
        log_probs = torch.clamp(log_probs, min=-100, max=0.)
        log_probs[log_probs != log_probs] = -100.
             
        return v_pred, v_target, log_probs, entropy
    
    
    def update_target_vnet(self):
        self.target_vnet.load_state_dict(self.critic_net.state_dict())
        self.target_vnet.eval()
        
        
    def step(self, buffer):
        # Set the torch models to training mode
        self.actor_net.train(); self.critic_net.train();
        # Calculate v_pred, v_target
        v_pred, v_target, log_probs, entropy = \
        self._prepare_pred_target(buffer)
        advantage = v_target - v_pred
        
        if np.random.rand()<0.001:
            if self.logger is not None:
                self.logger.entropy_ts.append(entropy.item())
                self.logger.adv_ts.append(advantage.detach().mean().item())
                self.logger.logprobs_ts.append(log_probs.detach().median().item())
            print('Advantage mean= ', advantage.detach().mean().item(), 
                 'Log probs Median = ', log_probs.detach().median().item(),
                 'Entropy = ', entropy.item()) #@@@@@@@
        
        critic_loss = advantage.pow(2).mean()
        actor_loss  = (-1)*-(log_probs * 20*advantage.detach()).mean() - 0.001 * entropy #*1*advantage.detach()).mean()
        
        self.actor_opt.zero_grad(); self.critic_opt.zero_grad()
        actor_loss.backward(); critic_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.actor_net.parameters(), max_norm=0.1, norm_type='inf') #Clip gradients #@@@
        self.actor_opt.step(); self.critic_opt.step();
        
        if self.scheduler is not None:
            self.scheduler.step()
            
        # Set the torch models back to evaluation mode
        self.actor_net.eval(); self.critic_net.eval();
        
        
class ACTrainer2(ACTrainer):
    
    # The actor uses the representations learned by the critic
    def _prepare_pred_target(self, buffer):
        # This function 
        if self.agent is None:
            raise ValueError('model.agent must be set.')
        batch_size_ = min(len(buffer), self.agent.hp.batch_size)
        transitions = buffer.sample(batch_size_)
        
        s_batch = torch.zeros(batch_size_, len(transitions[0][0]))#, dtype=torch.float, device=device)
        a_batch = torch.zeros(batch_size_)
        nexts_batch = torch.zeros(batch_size_, len(transitions[0][2]))
        r_batch = torch.zeros(batch_size_)
        
        for i, item in enumerate(transitions):
            s_batch[i,:], a_batch[i], nexts_batch[i,:], r_batch[i] = \
            torch.as_tensor(item[0]), item[1], torch.as_tensor(item[2]), item[3]
        
        v_pred = self.critic_net(s_batch)
        v_target = r_batch + self.agent.hp.gamma* self.target_vnet(nexts_batch).detach()

        # With the actor network
        _, critic_repr = self.target_vnet(s_batch, return_hidden=True)
        ftr_batch = torch.cat((s_batch, self.critic_ftr_scale*critic_repr.detach()), 1) 
        dists = self.actor_net(ftr_batch)
        entropy = 0.; log_probs = torch.zeros(batch_size_)
        
        entropy = dists.entropy().mean()
        log_probs = dists.log_prob(a_batch) #@@@@@ May be problematic
        log_probs = torch.clamp(log_probs, min=-100, max=0.)
        log_probs[log_probs != log_probs] = -100.
             
        return v_pred, v_target, log_probs, entropy        
        
        
class ACTrainer_nstep(ACTrainer):
    
    def __init__(self, actor_net, critic_net, actor_opt, critic_opt, criterion, nsteps=1, scheduler=None, agent=None, logger=None):
        super().__init__(actor_net, critic_net, actor_opt, critic_opt, criterion, scheduler, agent, logger)
        self.nsteps = nsteps
    
    def step(self, buffer):
        
        if len(buffer)< self.nsteps:
            return
        
        # Set the torch models to training mode
        self.actor_net.train(); self.critic_net.train();
        # Calculate v_pred, v_target
        v_pred, v_target, log_probs, entropy = \
        self._prepare_pred_target(buffer)
        advantage = v_target - v_pred
        
        if np.random.rand()<0.001:
            if self.logger is not None:
                self.logger.entropy_ts.append(entropy.item())
                self.logger.adv_ts.append(advantage.detach().mean().item())
                self.logger.logprobs_ts.append(log_probs.detach().median().item())
            print('Advantage mean= ', advantage.detach().mean().item(), 
                 'Log probs Median = ', log_probs.detach().median().item(),
                 'Entropy = ', entropy.item()) #@@@@@@@
        
        critic_loss = advantage.pow(2).mean()
        actor_loss  = (-1)*-(log_probs * (20./self.nsteps)*advantage.detach()).mean() - 0.001 * entropy 
        
        self.actor_opt.zero_grad(); self.critic_opt.zero_grad()
        actor_loss.backward(); critic_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.actor_net.parameters(), max_norm=0.1, norm_type='inf') #Clip gradients #@@@
        self.actor_opt.step(); self.critic_opt.step();
        
        if self.scheduler is not None:
            self.scheduler.step()
            
        # Set the torch models back to evaluation mode
        self.actor_net.eval(); self.critic_net.eval();
        
        
    def _prepare_pred_target(self, buffer):
        
        if self.agent is None:
            raise ValueError('model.agent must be set.')
        batch_size_ = min(len(buffer), self.agent.hp.batch_size)
        samples = buffer.sample_nstep(batch_size_, self.nsteps)
        
        s_batch = torch.zeros(batch_size_, len(samples[0][0][0]))#, dtype=torch.float, device=device)
        a_batch = torch.zeros(batch_size_)
        nexts_batch = torch.zeros(batch_size_, len(samples[0][2][0]))
        r_batch = torch.zeros(batch_size_)
        
        discounts = np.asarray([self.agent.hp.gamma**i for i in range(self.nsteps)])
        
        
        for i, item in enumerate(samples):
            f_item = item[0]
            l_item = item[-1]
            s_batch[i,:], a_batch[i], nexts_batch[i,:] = \
            torch.as_tensor(f_item[0]), f_item[1], torch.as_tensor(l_item[2])
            rewards = [item[j][3] for j in range(self.nsteps)]
            r_batch[i] = np.dot(discounts, rewards)
            
        v_pred = self.critic_net(s_batch)#.view(-1)
        v_target = r_batch + (self.agent.hp.gamma**self.nsteps)* self.target_vnet(nexts_batch).detach()

        # With the actor network
        dists = self.actor_net(s_batch)
        entropy = 0.; log_probs = torch.zeros(batch_size_)
        
        entropy = dists.entropy().mean()
        log_probs = dists.log_prob(a_batch) 
        log_probs = torch.clamp(log_probs, min=-100, max=0.)
        log_probs[log_probs != log_probs] = -100.
             
        return v_pred, v_target, log_probs, entropy
        
        
class ACTrainer3(ACTrainer):
    # imports: Transition, 
    # deepcopy from copy, numpy as np, np.asarray as narr
    # torch.FloatTensor as tarr
    
    # target policy network
    
    def __init__(self, actor_net, critic_net, actor_opt, critic_opt, criterion, nsteps=1, scheduler=None, agent=None, logger=None):
        super().__init__(actor_net, critic_net, actor_opt, critic_opt, criterion, scheduler, agent, logger)
        self.nsteps = nsteps
        self.target_pnet = deepcopy(actor_net)
    
    def update_target_nets(self):
        self.target_vnet.load_state_dict(self.critic_net.state_dict())
        self.target_vnet.eval()
        
        self.target_pnet.load_state_dict(self.actor_net.state_dict())
        self.target_pnet.eval()
        
        
        
class PPOTrainer(ACTrainer3):
    
    def __init__(self, actor_net, critic_net, actor_opt, critic_opt, criterion, trust_region=0.2, nsteps=1, scheduler=None, agent=None, logger=None, log_weights=(0,0)):
        super().__init__(actor_net, critic_net, actor_opt, critic_opt, criterion, nsteps, scheduler, agent, logger)
        self.trust_region = trust_region
        self.log_weights = log_weights # log_weights is a tuple of size 2, determinig the number of layers from the actor and from the critic to be logged.
        
    def _prepare_pred_target(self, buffer):
        # This function 
        if self.agent is None:
            raise ValueError('model.agent must be set.')
        batch_size_ = min(len(buffer), self.agent.hp.batch_size)
        transitions = buffer.sample(batch_size_)
        
        s_batch = torch.zeros(batch_size_, len(transitions[0][0]))#, dtype=torch.float, device=device)
        a_batch = torch.zeros(batch_size_)
        nexts_batch = torch.zeros(batch_size_, len(transitions[0][2]))
        r_batch = torch.zeros(batch_size_)
        log_probs_old = torch.zeros(batch_size_)
        
        for i, item in enumerate(transitions):
            s_batch[i,:], a_batch[i], nexts_batch[i,:], r_batch[i], log_probs_old[i] = \
            torch.tensor(item[0]), item[1][0], torch.tensor(item[2]), item[3], item[1][1]
        
        
        v_pred = self.critic_net(s_batch)#.view(-1)
        v_target = r_batch + self.agent.hp.gamma* self.target_vnet(nexts_batch).detach()

        # With the actor network
        dists = self.actor_net(s_batch)
        entropy = 0.; log_probs = torch.zeros(batch_size_)
        
        entropy = dists.entropy().mean()
        log_probs = dists.log_prob(a_batch) 
        log_probs = torch.clamp(log_probs, min=-100, max=0.)
        log_probs[log_probs != log_probs] = -100.
        
#         print('entropy, log_probs', entropy, log_probs)
        
#         for i in range(batch_size_):
#             entropy += dists[i].entropy()#.mean()
#             log_probs[i] = dists[i].log_prob(a_batch[i]) #@@@@@ May be problematic
             
        return v_pred, v_target, log_probs, log_probs_old, entropy 
    
    
    def step(self, buffer):
        # Set the torch models to training mode
        self.actor_net.train(); self.critic_net.train();
        # Calculate v_pred, v_target
        v_pred, v_target, log_probs, log_probs_old, entropy = \
        self._prepare_pred_target(buffer)
        advantage = v_target - v_pred
        
        critic_loss = advantage.pow(2).mean()
        
        ratio = torch.exp(log_probs - log_probs_old)
        surr1 = ratio * advantage.detach()
        surr2 = torch.clamp(ratio, 1.0 - self.trust_region, 1.0 + self.trust_region) * advantage.detach()
        
        actor_loss = (-1.)*-torch.min(surr1, surr2).mean()- 0.001 * entropy
        
        self.actor_opt.zero_grad(); self.critic_opt.zero_grad()
        actor_loss.backward(); 
        critic_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.actor_net.parameters(), max_norm=0.1, norm_type='inf') #Clip gradients #@@@
        torch.nn.utils.clip_grad_norm_(self.critic_net.parameters(), max_norm=0.1, norm_type='inf') #Clip gradients #@@@
        self.actor_opt.step(); self.critic_opt.step();
        
        if self.scheduler is not None:
            self.scheduler.step()
                    
        if self.log_weights[0]>0: 
            ac_layers = [module[1] for module in self.actor_net.named_modules()]
            for l in range(self.log_weights[0]):
                 self.logger.actor_weight_ts[l].append(
                     deepcopy(ac_layers[-1-l].weight.detach().numpy()))

        if self.log_weights[1]>0: 
            cr_layers = [module[1] for module in self.critic_net.named_modules()]
            for l in range(self.log_weights[1]):
                 self.logger.critic_weight_ts[l].append(
                     deepcopy(cr_layers[-1-l].weight.detach().numpy()))
                    
        if np.random.rand()<0.003:
            if self.logger is not None:
                self.logger.entropy_ts.append(entropy.item())
                self.logger.adv_ts.append(advantage.detach().mean().item())
                self.logger.logprobs_ts.append(log_probs.detach().median().item())
            print('Advantage mean= ', advantage.detach().mean().item(), 
                 'Log probs Median = ', log_probs.detach().median().item(),
                 'Entropy = ', entropy.item())
            
            
class PPO2Trainer(PPOTrainer):

# The actor loss function accounts for the "effort" advantage, as well as the advantage returned by the critic.
    
    def __init__(self, actor_net, critic_net, actor_opt, 
                 critic_opt, criterion, trust_region=0.2, 
                 nsteps=1, scheduler=None, agent=None, logger=None, 
                 log_weights=(0,0), f_ftr_idx=-1):
        
        super().__init__(actor_net, critic_net, actor_opt, 
                         critic_opt, criterion, nsteps=nsteps, 
                         scheduler=scheduler, agent=agent, logger=logger,
                         log_weights=log_weights, trust_region=trust_region)
        self.f_ftr_idx = f_ftr_idx
        
    def _prepare_pred_target(self, buffer):
        # This function 
        if self.agent is None:
            raise ValueError('model.agent must be set.')
        batch_size_ = min(len(buffer), self.agent.hp.batch_size)
        transitions = buffer.sample(batch_size_)
        
        s_batch = torch.zeros(batch_size_, len(transitions[0][0]))#, dtype=torch.float, device=device)
        a_batch = torch.zeros(batch_size_)
        nexts_batch = torch.zeros(batch_size_, len(transitions[0][2]))
        r_batch = torch.zeros(batch_size_)
        log_probs_old = torch.zeros(batch_size_)
        
        for i, item in enumerate(transitions):
            s_batch[i,:], a_batch[i], nexts_batch[i,:], r_batch[i], log_probs_old[i] = \
            torch.tensor(item[0]), item[1][0], torch.tensor(item[2]), item[3], item[1][1]
        
        
        v_pred = self.critic_net(s_batch)#.view(-1)
        v_target = r_batch + self.agent.hp.gamma* self.target_vnet(nexts_batch).detach()

        # With the actor network
        dists = self.actor_net(s_batch)
        entropy = 0.; log_probs = torch.zeros(batch_size_)
        
        entropy = dists.entropy().mean()
        log_probs = dists.log_prob(a_batch) 
        log_probs = torch.clamp(log_probs, min=-100, max=0.)
        log_probs[log_probs != log_probs] = -100.
        
        # Compute the terms for effort advantage
        curr_force_batch = s_batch[:, self.f_ftr_idx].detach().numpy()
        next_force_batch = nexts_batch[:, self.f_ftr_idx].detach().numpy()
        eff_target = self.agent.compute_effort_batch(next_force_batch)
        
        force_avg = self.agent.muscle.get_force_batch(curr_force_batch,
                                                      dists.loc.detach().numpy())
        
        eff_pred = self.agent.compute_effort_batch(force_avg)
        
        # bias = self.agent.muscle.alpha * dists.scale.detach().numpy()
        #+ self.agent.compute_effort_batch(bias)
             
        return v_pred, v_target, log_probs, log_probs_old, entropy, eff_pred, eff_target
    
    
    def _log_params(self, logger, param_specs):
        # self: the ActorCritic Network module; has actor_net and critic_net as attributes
        # logger: the object that is written on.
        # weight_specs: a tuple of size 2, specifying the number of layers from the actor 
        #               and the critic to record, respectively.
        ac_nparams = param_specs[0]; 
        if ac_nparams>0: 
            ac_params = [module[1] for module in self.actor_net.named_parameters()]
            for l in range(ac_nparams):
                logger.actor_weight_ts[l].append(
                     deepcopy(ac_params[l].detach().numpy()))
        return

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
    
    def _log_metrics(self, logger, actor_loss, adv, entropy, logprob, eff_adv_ratio):
        logger.actorloss_ts.append(actor_loss)
        logger.entropy_ts.append(entropy)
        logger.adv_ts.append(adv)
        logger.logprobs_ts.append(logprob)
        logger.eff_adv_ratio_ts.append(eff_adv_ratio)
                    
    
    def step(self, buffer, current_performance):
        # Set the torch models to training mode
        self.actor_net.train(); self.critic_net.train();
        # Calculate v_pred, v_target
        v_pred, v_target, log_probs, log_probs_old, entropy, eff_pred, eff_target = \
        self._prepare_pred_target(buffer)
        critic_adv = -1*(v_target - v_pred)
        
        critic_loss = critic_adv.pow(2).mean()
        
        training_scaler = scipy.special.expit(-16*np.log10(-current_performance)-12)
        effort_adv = -1* (eff_target - eff_pred)
    
#         training_scaler = 1.
        # Make sure the relative scale of critic_adv and eff_adv is appropriate.
        advantage = critic_adv.detach() + (1.)* training_scaler * effort_adv
        
        
        ratio_ = torch.exp(log_probs - log_probs_old)
        
        ratio = torch.clamp(ratio_, 1.0 - self.trust_region, 1.0 + self.trust_region)
        
#         surr1 = ratio * advantage
#         surr2 = torch.clamp(ratio, 1.0 - self.trust_region, 1.0 + self.trust_region) * advantage
#         actor_loss = -torch.min(surr1, surr2).mean()- 0.001 * entropy
        
        actor_loss = -(ratio*advantage).mean()- 0.001 * entropy 
        
        self.actor_opt.zero_grad(); self.critic_opt.zero_grad()
        actor_loss.backward(); 
        critic_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.actor_net.parameters(), max_norm=0.1, norm_type='inf') #Clip gradients #@@@
        torch.nn.utils.clip_grad_norm_(self.critic_net.parameters(), max_norm=0.1, norm_type='inf') #Clip gradients #@@@
        self.actor_opt.step(); self.critic_opt.step();
        
        if self.scheduler is not None:
            self.scheduler.step()
            

        # Log the training stats
        self._log_weights(self.logger, self.log_weights)

        if np.random.rand()<0.003:
            # compute metrics
            actor_loss_m = actor_loss.item()
            entropy_m = entropy.item()
            adv_m = torch.norm(critic_adv.detach()).item()/16
            logprob_m = log_probs.detach().median().item()
            eff_adv_ratio_m = torch.median(torch.abs(effort_adv/critic_adv.detach())).item()
    
            print('# Agent ', self.agent.perspective+1, 
                  'Actor loss = %.3f'%actor_loss_m,
                  'Advantage norm= %.3f'% adv_m,
                  'Effort / Error Advantage = %.3f'% eff_adv_ratio_m, 
                  'Effort loss scaler= %.3f'% training_scaler,
                 'Log probs Median =  %.3f'% logprob_m,
                 'Entropy = %.3f'% entropy_m)
            self._log_metrics(self.logger, actor_loss_m, adv_m, entropy_m, logprob_m, eff_adv_ratio_m)
            

            
class PPO3Trainer(PPO2Trainer):

# The trainer accounts for eps, which is a measure of exploration. Works with train_single3, PPO3Agent
        
    def _prepare_pred_target(self, buffer, eps):
        # This function 
        if self.agent is None:
            raise ValueError('model.agent must be set.')
        batch_size_ = min(len(buffer), self.agent.hp.batch_size)
        transitions = buffer.sample(batch_size_)
        
        s_batch = torch.zeros(batch_size_, len(transitions[0][0]))#, dtype=torch.float, device=device)
        a_batch = torch.zeros(batch_size_)
        nexts_batch = torch.zeros(batch_size_, len(transitions[0][2]))
        r_batch = torch.zeros(batch_size_)
        log_probs_old = torch.zeros(batch_size_)
        
        for i, item in enumerate(transitions):

#             try:
            s_batch[i,:], a_batch[i], nexts_batch[i,:], r_batch[i], log_probs_old[i] = \
            torch.tensor(item[0]), item[1][0], torch.tensor(item[2]), item[3], item[1][1]
#             except TypeError:
#                 print('PPO3Trainer: item = ', item)
        
        v_pred = self.critic_net(s_batch)#.view(-1)
        v_target = r_batch + self.agent.hp.gamma* self.target_vnet(nexts_batch).detach()

        # With the actor network
        dists = self.actor_net(s_batch, eps=eps)
        entropy = 0.; log_probs = torch.zeros(batch_size_)
        
        entropy = dists.entropy().mean()
        log_probs = dists.log_prob(a_batch) 
        log_probs = torch.clamp(log_probs, min=-100, max=0.)
        log_probs[log_probs != log_probs] = -100.
        
        # Compute the terms for effort advantage
        curr_force_batch = s_batch[:, self.f_ftr_idx].detach().numpy()
        next_force_batch = nexts_batch[:, self.f_ftr_idx].detach().numpy()
        eff_target = self.agent.compute_effort_batch(next_force_batch)
        
        force_avg = self.agent.muscle.get_force_batch(curr_force_batch,
                                                      dists.loc.detach().numpy())
        
        eff_pred = self.agent.compute_effort_batch(force_avg)
        
        # bias = self.agent.muscle.alpha * dists.scale.detach().numpy()
        #+ self.agent.compute_effort_batch(bias)
             
        return v_pred, v_target, log_probs, log_probs_old, entropy, eff_pred, eff_target
                    
    
    def step(self, buffer, current_performance):
        # Set the torch models to training mode
        self.actor_net.train(); self.critic_net.train();
        
        eps = 1- scipy.special.expit(-16*np.log10(-current_performance)-12)
        # Calculate the Pytorch-related components of the loss function
        v_pred, v_target, log_probs, log_probs_old, entropy, eff_pred, eff_target = \
        self._prepare_pred_target(buffer, eps)
        critic_adv = -1*(v_target - v_pred)
        
        critic_loss = 0.5*critic_adv.pow(2).mean()
        
        training_scaler = 1-eps
        effort_adv = -1* (eff_target - eff_pred)
    
#         training_scaler = 1.
        # Make sure the relative scale of critic_adv and eff_adv is appropriate.
    
#         v_pred_np = v_pred.detach().numpy()
#         term_sum = np.abs(v_pred_np)+ np.abs(eff_pred)
        
        adv_rew_term = critic_adv.detach() #* (np.abs(v_pred_np)/term_sum)
        adv_eff_term = 0.001* effort_adv * training_scaler #* (np.abs(eff_pred) / term_sum)
        
        advantage = adv_rew_term + adv_eff_term
#         advantage = critic_adv.detach()  + (1.)* training_scaler * effort_adv
        
        
        ratio_ = torch.exp(log_probs - log_probs_old)
        
        ratio = torch.clamp(ratio_, 1.0 - self.trust_region, 1.0 + self.trust_region)
        
#         surr1 = ratio * advantage
#         surr2 = torch.clamp(ratio, 1.0 - self.trust_region, 1.0 + self.trust_region) * advantage
#         actor_loss = -torch.min(surr1, surr2).mean()- 0.001 * entropy
        
        actor_loss = -(ratio*advantage).mean()- 0.001 * entropy 
        
        self.actor_opt.zero_grad(); self.critic_opt.zero_grad()
        actor_loss.backward(); 
        critic_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.actor_net.parameters(), max_norm=0.1, norm_type='inf') #Clip gradients #@@@
        torch.nn.utils.clip_grad_norm_(self.critic_net.parameters(), max_norm=0.1, norm_type='inf') #Clip gradients #@@@
        self.actor_opt.step(); self.critic_opt.step();
        
        if self.scheduler is not None:
            self.scheduler.step()
            

        # Log the training stats
        self._log_weights(self.logger, self.log_weights)

        if np.random.rand()<0.009:
            # compute metrics
            actor_loss_m = actor_loss.item()
            entropy_m = entropy.item()
            adv_m = torch.norm(critic_adv.detach()).item()/16
            logprob_m = log_probs.detach().median().item()
            eff_adv_ratio_m = torch.max(torch.abs(adv_eff_term/adv_rew_term)).item()
    
            print('# Agent ', self.agent.perspective+1, 
                  'Actor loss = %.3f'%actor_loss_m,
                  'Critic error norm= %.3f'% adv_m,
                  'Effort / Reward Advantage = %.3f'% eff_adv_ratio_m, 
                  'Effort loss scaler= %.3f'% training_scaler,
                 'Log probs Median =  %.3f'% logprob_m,
                 'Entropy = %.3f'% entropy_m)
            self._log_metrics(self.logger, actor_loss_m, adv_m, entropy_m, logprob_m, eff_adv_ratio_m)
            
            
            
class PPO4Trainer(PPO3Trainer):
    # modified the way eff_target, eff_pred is calculated
    # making adv_eff_term effective only some of the time

    def _prepare_pred_target(self, buffer, eps, ret_eff=True):
        # This function 
        if self.agent is None:
            raise ValueError('model.agent must be set.')
        batch_size_ = min(len(buffer), self.agent.hp.batch_size)
        transitions = buffer.sample(batch_size_)
        
        s_batch = torch.zeros(batch_size_, len(transitions[0][0]))#, dtype=torch.float, device=device)
        a_batch = torch.zeros(batch_size_)
        nexts_batch = torch.zeros(batch_size_, len(transitions[0][2]))
        r_batch = torch.zeros(batch_size_)
        log_probs_old = torch.zeros(batch_size_)
        
        for i, item in enumerate(transitions):

#             try:
            s_batch[i,:], a_batch[i], nexts_batch[i,:], r_batch[i], log_probs_old[i] = \
            torch.tensor(item[0]), item[1][0], torch.tensor(item[2]), item[3], item[1][1]
#             except TypeError:
#                 print('PPO3Trainer: item = ', item)
        
        v_pred = self.critic_net(s_batch)#.view(-1)
        v_target = r_batch + self.agent.hp.gamma* self.target_vnet(nexts_batch).detach()

        # With the actor network
        dists = self.actor_net(s_batch, eps=eps)
        entropy = 0.; log_probs = torch.zeros(batch_size_)
        
        entropy = dists.entropy().mean()
        log_probs = dists.log_prob(a_batch) 
        log_probs = torch.clamp(log_probs, min=-100, max=100.)
        log_probs[log_probs != log_probs] = -100.
        
        if ret_eff is False:
            return v_pred, v_target, log_probs, log_probs_old, entropy
        else:
        # Compute the terms for effort advantage
            curr_force_batch = s_batch[:, self.f_ftr_idx].detach().numpy()
            next_force_batch = nexts_batch[:, self.f_ftr_idx].detach().numpy()
            # What is the average effort in this state? => what would the effort be given our current policy?
            eff_target = self.agent.compute_effort_batch(
                next_force_batch)

            force_avg = self.agent.muscle.get_force_batch(
                curr_force_batch, dists.loc.detach().numpy())
            eff_pred = self.agent.compute_effort_batch(force_avg)

            return v_pred, v_target, log_probs, log_probs_old, entropy, eff_pred, eff_target
    
    
    def step(self, buffer, current_performance):
        # Set the torch models to training mode
        self.actor_net.train(); self.critic_net.train();
        
        eps = 1- scipy.special.expit(-16*np.log10(-current_performance)-12)
        # Calculate the Pytorch-related components of the loss function
        if self.agent.c_positivef!=0 and self.agent.c_negativef!=0 and \
        current_performance >-0.1: #and np.random.rand()<0.3: 
            
            v_pred, v_target, log_probs, log_probs_old, entropy, eff_pred, eff_target = \
            self._prepare_pred_target(buffer, eps, ret_eff=True)
            
            effort_adv = -1* (eff_target - eff_pred) #in a good update, eff_target < eff_pred
            adv_eff_term = effort_adv #* training_scaler #* (np.abs(eff_pred) / term_sum)
            
        else:
            v_pred, v_target, log_probs, log_probs_old, entropy = \
            self._prepare_pred_target(buffer, eps, ret_eff=False)
            
            adv_eff_term = 0.
        
        critic_adv = -1*(v_target - v_pred)
        critic_loss = 0.5*critic_adv.pow(2).mean() #@@@@@@@
        adv_rew_term = critic_adv.detach() #* (np.abs(v_pred_np)/term_sum)
        
        ceils = 2* np.abs(adv_rew_term)
        adv_eff_term = np.clip(adv_eff_term, -ceils, ceils)
            
        advantage = adv_rew_term + adv_eff_term
        
        ratio_ = torch.exp(log_probs - log_probs_old)
        
        ratio = torch.clamp(ratio_, 1.0 - self.trust_region, 1.0 + self.trust_region)
        
#         surr1 = ratio * advantage
#         surr2 = torch.clamp(ratio, 1.0 - self.trust_region, 1.0 + self.trust_region) * advantage
#         actor_loss = -torch.min(surr1, surr2).mean()- 0.001 * entropy
        
        actor_loss = -(ratio*advantage).mean()- 0.001 * entropy 
        
        self.actor_opt.zero_grad(); self.critic_opt.zero_grad()
        actor_loss.backward(); 
        critic_loss.backward() #@@@@@@@@@@@@@@@
        
        torch.nn.utils.clip_grad_norm_(self.actor_net.parameters(), max_norm=0.1, norm_type='inf') #Clip gradients #@@@
        torch.nn.utils.clip_grad_norm_(self.critic_net.parameters(), max_norm=0.1, norm_type='inf') #Clip gradients #@@@@@@@@@@
        self.actor_opt.step(); 
        self.critic_opt.step(); #@@@@@@@@@@@@@@@@
        
        if self.scheduler is not None:
            self.scheduler.step()
            

        # Log the training stats
        self._log_weights(self.logger, self.log_weights)

        if np.random.rand()<0.003:
            # compute metrics
            actor_loss_m = actor_loss.item()
            entropy_m = entropy.item()
            adv_m = torch.norm(critic_adv.detach()).item()/16
            logprob_m = log_probs.detach().median().item()
            eff_adv_ratio_m = torch.median(torch.abs(adv_eff_term/adv_rew_term)).item()
    
            print('# Agent ', self.agent.perspective+1, 
                  'Actor loss = %.3f'%actor_loss_m,
                  'Critic error norm= %.3f'% adv_m,
                  'Effort / Reward Advantage = %.3f'% eff_adv_ratio_m,
                 'Log probs Median =  %.3f'% logprob_m,
                 'Entropy = %.3f'% entropy_m)
            self._log_metrics(self.logger, actor_loss_m, adv_m, entropy_m, logprob_m, eff_adv_ratio_m)
            
            
class PPOnstepTrainer(PPO4Trainer):
    
    def _prepare_pred_target(self, buffer, eps, ret_eff=True):
        # This function 
        if self.agent is None:
            raise ValueError('model.agent must be set.')
        batch_size_ = min(len(buffer), self.agent.hp.batch_size)
        transitions = buffer.sample_nstep(batch_size_, self.nsteps)
        
        s_batch = torch.zeros(batch_size_, len(transitions[0][0][0]))#, dtype=torch.float, device=device)
        a_batch = torch.zeros(batch_size_)
        nexts_batch = torch.zeros(batch_size_, len(transitions[0][2][0]))
        r_batch = torch.zeros(batch_size_)
        log_probs_old = torch.zeros(batch_size_)
        
        discounts = np.asarray([self.agent.hp.gamma**i 
                                for i in range(self.nsteps)])
        discounts = discounts/np.sum(discounts)
        
        for i, item in enumerate(transitions):
            f_item = item[0]
            l_item = item[-1]
#             try:
            s_batch[i,:], a_batch[i], nexts_batch[i,:], log_probs_old[i] = \
            torch.as_tensor(f_item[0]), f_item[1][0], torch.as_tensor(l_item[2]), f_item[1][1]
            rewards = [item[j][3] for j in range(self.nsteps)]
            r_batch[i] = np.dot(discounts, rewards)
#             except TypeError:
#                 print('PPO3Trainer: item = ', item)
        
        v_pred = self.critic_net(s_batch)#.view(-1)
        v_target = r_batch + (self.agent.hp.gamma**self.nsteps)* self.target_vnet(nexts_batch).detach()

        # With the actor network
        dists = self.actor_net(s_batch, eps=eps)
        entropy = 0.; log_probs = torch.zeros(batch_size_)
        
        entropy = dists.entropy().mean()
        log_probs = dists.log_prob(a_batch) 
        log_probs = torch.clamp(log_probs, min=-100, max=100.)
        log_probs[log_probs != log_probs] = -100.
        
        if ret_eff is False:
            return v_pred, v_target, log_probs, log_probs_old, entropy
        else:
        # How is the action in this state doing in terms of effort?
        # Normally we'd get x amounts of effort in this state. This policy gets y amount of effort in this state.
        # Compute the terms for the effort advantage
            curr_force_batch = s_batch[:, self.f_ftr_idx].detach().numpy()
            next_force_batch = nexts_batch[:, self.f_ftr_idx].detach().numpy()
            effective_f_batch = 0.5*(curr_force_batch+next_force_batch)
            # How much was the effort given the old policy?
            eff_target = self.agent.compute_effort_batch(effective_f_batch)

            # What is the average effort in this state? => assuming the average of all possible actions
            # is 0 and hence no change in force is caused.
            eff_pred = self.agent.compute_effort_batch(curr_force_batch)
            #in a good update, eff_target < eff_pred
            
            return v_pred, v_target, log_probs, log_probs_old, entropy, eff_pred, eff_target
    
    
    def step(self, buffer, current_performance):
        
        if len(buffer)< self.nsteps+1:
            return
        super().step(buffer, current_performance)
        

        
class PPOMultiActionNstepTrainer(PPO2Trainer):
    # works with multiple actions.
    # In addition to other modifications to trust_region and logprobs.
    # Works with PPO4PDAgent 
    def __init__(self, actor_net, critic_net, actor_opt, 
                 critic_opt, criterion, n_actions, trust_region=1., 
                 nsteps=1, scheduler=None, agent=None, logger=None, 
                 log_weights=(0,0), f_ftr_idx=-1):
        
        super().__init__(actor_net, critic_net, actor_opt, 
                         critic_opt, criterion, nsteps=nsteps, 
                         scheduler=scheduler, agent=agent, logger=logger,
                         log_weights=log_weights, trust_region=trust_region, 
                         f_ftr_idx=f_ftr_idx)
        self.nA = n_actions
    
    
    def _prepare_pred_target(self, buffer, eps, ret_eff=True):
        # This function 
        if self.agent is None:
            raise ValueError('model.agent must be set.')
        batch_size_ = min(len(buffer), self.agent.hp.batch_size)
        transitions = buffer.sample_nstep(batch_size_, self.nsteps)
        
        s_batch = torch.zeros(batch_size_, len(transitions[0][0][0]))#, dtype=torch.float, device=device)
        a_batch = torch.zeros(batch_size_, self.nA)
        nexts_batch = torch.zeros(batch_size_, len(transitions[0][2][0]))
        r_batch = torch.zeros(batch_size_)
        log_probs_old = torch.zeros(batch_size_)
        
        discounts = np.asarray([self.agent.hp.gamma**i 
                                for i in range(self.nsteps)])
        discounts = discounts/np.sum(discounts)
        
        for i, item in enumerate(transitions):
            # Each item has n tuples, corresponding to n consecutive time steps
            f_item = item[0] # very next step
            l_item = item[-1] # nth step from now
#             try:
            s_batch[i,:], a_batch[i], nexts_batch[i,:], log_probs_old[i] = \
                torch.as_tensor(f_item[0]), torch.as_tensor(f_item[1][0]), torch.as_tensor(l_item[2]), f_item[1][1]
            rewards = [item[j][3] for j in range(self.nsteps)]
            r_batch[i] = np.dot(discounts, rewards)
#             except TypeError:
#                 print('PPO3Trainer: item = ', item)
        
        v_pred = self.critic_net(s_batch)#.view(-1)
        v_target = r_batch + (self.agent.hp.gamma**self.nsteps)* self.target_vnet(nexts_batch).detach()

        # With the actor network
        dists = self.actor_net(s_batch, eps=eps)
#         entropy = 0.; log_probs = torch.zeros(batch_size_)
        
        entropy = dists.entropy().mean()
        log_probs = dists.log_prob(a_batch) 
        log_probs = torch.clamp(log_probs, min=-100, max=100.)
        log_probs[log_probs != log_probs] = -100.
        
        if ret_eff is False:
            return v_pred, v_target, log_probs, log_probs_old, entropy
        else:
        # How is the action in this state doing in terms of effort?
        # Normally we'd get x amounts of effort in this state. This policy gets y amount of effort in this state.
        # Compute the terms for the effort advantage
            curr_force_batch = s_batch[:, self.f_ftr_idx].detach().numpy()
            next_force_batch = nexts_batch[:, self.f_ftr_idx].detach().numpy()
            effective_f_batch = 0.5*(curr_force_batch+next_force_batch)
            # How much was the effort given the old policy?
            eff_target = self.agent.compute_effort_batch(effective_f_batch)

            # What is the average effort in this state? => assuming the average of all possible actions
            # is 0 and hence no change in force is caused.
            eff_pred = self.agent.compute_effort_batch(curr_force_batch)
            #in a good update, eff_target < eff_pred
            
            return v_pred, v_target, log_probs, log_probs_old, entropy, eff_pred, eff_target
        
        
    def step(self, buffer, current_performance):
        
        if len(buffer)< self.nsteps+1:
            return
        
        self.actor_net.train(); self.critic_net.train();
        
        eps = 1- scipy.special.expit(-16*np.log10(-current_performance)-12)
        # Calculate the Pytorch-related components of the loss function
        if self.agent.c_positivef!=0 and self.agent.c_negativef!=0 and \
                current_performance >-0.1: #and np.random.rand()<0.3: 
            
            v_pred, v_target, log_probs, log_probs_old, entropy, eff_pred, eff_target = \
            self._prepare_pred_target(buffer, eps, ret_eff=True)
            
            effort_adv = -1* (eff_target - eff_pred) #in a good update, eff_target < eff_pred
            adv_eff_term = effort_adv #* training_scaler #* (np.abs(eff_pred) / term_sum)
            
        else:
            v_pred, v_target, log_probs, log_probs_old, entropy = \
            self._prepare_pred_target(buffer, eps, ret_eff=False)
            
            adv_eff_term = 0.
        
        critic_adv = -1*(v_target - v_pred)
        critic_loss = 0.5*critic_adv.pow(2).mean()
        adv_rew_term = critic_adv.detach() #* (np.abs(v_pred_np)/term_sum)
        
        ceils = 2* np.abs(adv_rew_term)
        adv_eff_term = np.clip(adv_eff_term, -ceils, ceils)
        
        advantage = adv_rew_term + adv_eff_term
        
#         logprobs_diff_ = log_probs - log_probs_old 
#         logprobs_diff = torch.clamp(logprobs_diff_, -self.trust_region, self.trust_region)
#         ratio = torch.exp(logprobs_diff)
        
        logprobs_diff = log_probs - log_probs_old 
        logprobs_diff_c = torch.clamp(logprobs_diff, -self.trust_region, self.trust_region)
        
        surr1 = torch.exp(logprobs_diff)
        surr2 = torch.exp(logprobs_diff_c)
        ratio = torch.min(surr1, surr2)
        
        actor_loss = -(ratio*advantage).mean()- 0.001 * entropy 
#         surr1 = ratio * advantage
#         surr2 = torch.clamp(ratio, 1.0 - self.trust_region, 1.0 + self.trust_region) * advantage
#         actor_loss = -torch.min(surr1, surr2).mean()- 0.001 * entropy
        
        self.actor_opt.zero_grad(); self.critic_opt.zero_grad()
        actor_loss.backward(); 
        critic_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.actor_net.parameters(), max_norm=0.1, norm_type='inf') #Clip gradients #@@@
        torch.nn.utils.clip_grad_norm_(self.critic_net.parameters(), max_norm=0.1, norm_type='inf') #Clip gradients #@@@@@@@@@@
        self.actor_opt.step(); 
        self.critic_opt.step();
        
        if self.scheduler is not None:
            self.scheduler.step()
            

        # Log the training stats
        self._log_weights(self.logger, self.log_weights)

        if np.random.rand()<0.003:
            # compute metrics
            actor_loss_m = actor_loss.item()
            entropy_m = entropy.item()
            adv_m = torch.norm(critic_adv.detach()).item()/16
            logprob_m = log_probs.detach().median().item()
            eff_adv_ratio_m = torch.median(torch.abs(adv_eff_term/adv_rew_term)).item()
    
            print('# Agent ', self.agent.perspective+1, 
                  'Actor loss = %.3f'%actor_loss_m,
                  'Critic error norm= %.3f'% adv_m,
                  'Effort / Reward Advantage = %.3f'% eff_adv_ratio_m,
                 'Log probs Median =  %.3f'% logprob_m,
                 'Entropy = %.3f'% entropy_m)
            self._log_metrics(self.logger, actor_loss_m, adv_m, entropy_m, logprob_m, eff_adv_ratio_m)
        
            
            
class PPOMultiActionNstepGPUTrainer(PPO2Trainer):
    # works with multiple actions.
    # In addition to other modifications to trust_region and logprobs.
    # Works with PPO4PDAgent 
    def __init__(self, actor_net, critic_net, actor_opt, 
                 critic_opt, criterion, n_actions, trust_region=1., 
                 nsteps=1, scheduler=None, agent=None, logger=None,
                 log_weights=(0,0), f_ftr_idx=-1, device=None, gamma=1):
        
        super().__init__(actor_net, critic_net, actor_opt, 
                         critic_opt, criterion, nsteps=nsteps, 
                         scheduler=scheduler, agent=agent, logger=logger,
                         log_weights=log_weights, trust_region=trust_region, 
                         f_ftr_idx=f_ftr_idx)
        self.nA = n_actions
        
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = device
            
        discounts = np.array([gamma**i for i in range(self.nsteps)])
        discounts = discounts/np.sum(discounts)
        self.discounts = torch.tensor(discounts, device=self.device, dtype=torch.float32)
        
#         if nsteps>1:
#             self._get_batch_experience = _get_batch_experience_nstep

    def _unpack_nsteps(self, nsteps_trans): 
        reward_n = torch.as_tensor(tuple(item.reward for item in nsteps_trans), device=self.device)
        reward = torch.dot(self.discounts, reward_n)
        return (nsteps_trans[0].state.view(1,-1),
                nsteps_trans[0].action[0],
                nsteps_trans[-1].next_state.view(1,-1),
                reward.view(1,-1),
                nsteps_trans[0].action[1].view(1,-1))
                
    def _get_batch_experience(self, buffer):
        batch_size_ = min(len(buffer), self.agent.hp.batch_size)
        transitions = buffer.sample_nstep(batch_size_, self.nsteps)
        # returns a list of lists
        
        s_batch, a_batch, nexts_batch, r_batch, log_probs_old = tuple(
            zip(*map(self._unpack_nsteps, transitions)))
#         s_batch = torch.as_tensor(s_batch)

#         print('s_batch = ', type(s_batch))
        
        s_batch = torch.cat(s_batch, dim=0)
        a_batch = torch.cat(a_batch, dim=0)
        nexts_batch = torch.cat(nexts_batch, dim=0)
        r_batch = torch.cat(r_batch, dim=0)
        log_probs_old = torch.cat(log_probs_old, dim=0)
#         print('r_batch = ', r_batch)
#         print('log_probs_old = ', log_probs_old)
        #, device=self.device)
#         r_batch = torch.zeros(batch_size_, device=self.device)
        
#         for i, item in enumerate(transitions):
#             # Each item has n tuples, corresponding to n consecutive time steps
#             rewards = [item[j][3] for j in range(self.nsteps)]
#             r_batch[i] = torch.dot(self.discounts, rewards)
        
        return s_batch, a_batch, nexts_batch, r_batch, log_probs_old
    
    
    def _prepare_pred_target(self, s_batch, a_batch, nexts_batch, r_batch, log_probs_old, eps, ret_eff=True):
        # This function 
        if self.agent is None:
            raise ValueError('model.agent must be set.')
        
        v_pred = self.critic_net(s_batch)#.view(-1)
        v_target = r_batch + (self.agent.hp.gamma**self.nsteps)* self.target_vnet(nexts_batch).detach()

        # With the actor network
        dists = self.actor_net(s_batch, eps=eps)
#         entropy = 0.; log_probs = torch.zeros(batch_size_)
        
        entropy = dists.entropy().mean()
        log_probs = dists.log_prob(a_batch) 
        log_probs = torch.clamp(log_probs, min=-100, max=100.)
        log_probs[log_probs != log_probs] = -100.
        
        if ret_eff is False:
            return v_pred, v_target, log_probs, log_probs_old, entropy
        else:
        # How is the action in this state doing in terms of effort?
        # Normally we'd get x amounts of effort in this state. This policy gets y amount of effort in this state.
        # Compute the terms for the effort advantage
            curr_force_batch = s_batch[:, self.f_ftr_idx].detach()#.numpy()
            next_force_batch = nexts_batch[:, self.f_ftr_idx].detach()#.numpy()
            effective_f_batch = 0.5*(curr_force_batch+next_force_batch)
            # How much was the effort given the old policy?
            eff_target = self.agent.compute_effort_batch(effective_f_batch)

            # What is the average effort in this state? => assuming the average of all possible actions
            # is 0 and hence no change in force is caused.
            eff_pred = self.agent.compute_effort_batch(curr_force_batch)
            #in a good update, eff_target < eff_pred
            
            return v_pred, v_target, log_probs, log_probs_old, entropy, eff_pred, eff_target
        
        
    def step(self, buffer, current_performance):
        if len(buffer)< self.nsteps+1:
            return
        
        self.actor_net.train(); self.critic_net.train();
        
#         eps = 1- scipy.special.expit(-16*np.log10(-current_performance)-12)
        eps = 1.

        s_batch, a_batch, nexts_batch, r_batch, log_probs_old = self._get_batch_experience(buffer)
    
    
        # Calculate the Pytorch-related components of the loss function
        if self.agent.c_positivef!=0 and self.agent.c_negativef!=0 and \
                current_performance >-0.1: #and np.random.rand()<0.3: 
            
            v_pred, v_target, log_probs, log_probs_old, entropy, eff_pred, eff_target = \
            self._prepare_pred_target(s_batch, a_batch, nexts_batch, 
                                      r_batch, log_probs_old, eps, ret_eff=True)
            
            effort_adv = -1* (eff_target - eff_pred) #in a good update, eff_target < eff_pred
            adv_eff_term = effort_adv #* training_scaler #* (np.abs(eff_pred) / term_sum)
            
        else:
            v_pred, v_target, log_probs, log_probs_old, entropy = \
            self._prepare_pred_target(s_batch, a_batch, nexts_batch, 
                                      r_batch, log_probs_old, eps, ret_eff=False)
            
            adv_eff_term = torch.tensor([0.], dtype=torch.float32)
        
        critic_adv = -1*(v_target - v_pred)
        critic_loss = 0.5*critic_adv.pow(2).mean()
        adv_rew_term = critic_adv.detach() #* (np.abs(v_pred_np)/term_sum)
        
        if adv_eff_term.view(-1).shape[0] != 1:
            ceils = 2* torch.abs(adv_rew_term)
            adv_eff_term = torch.where(adv_eff_term>ceils, ceils, adv_eff_term)
            adv_eff_term = torch.where(adv_eff_term<-ceils, -ceils, adv_eff_term)
#         adv_eff_term = np.clip(adv_eff_term, -ceils, ceils)
        
        advantage = adv_rew_term + adv_eff_term
        
#         logprobs_diff_ = log_probs - log_probs_old 
#         logprobs_diff = torch.clamp(logprobs_diff_, -self.trust_region, self.trust_region)
#         ratio = torch.exp(logprobs_diff)
        
        logprobs_diff = log_probs - log_probs_old 
        logprobs_diff_c = torch.clamp(logprobs_diff, -self.trust_region, self.trust_region)
        
        surr1 = torch.exp(logprobs_diff)
        surr2 = torch.exp(logprobs_diff_c)
        ratio = torch.min(surr1, surr2)
        
        actor_loss = -(ratio*advantage).mean()- 0.001 * entropy 
#         surr1 = ratio * advantage
#         surr2 = torch.clamp(ratio, 1.0 - self.trust_region, 1.0 + self.trust_region) * advantage
#         actor_loss = -torch.min(surr1, surr2).mean()- 0.001 * entropy
        
        self.actor_opt.zero_grad(); self.critic_opt.zero_grad()
        actor_loss.backward(); 
        critic_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.actor_net.parameters(), max_norm=0.1, norm_type='inf') #Clip gradients #@@@
        torch.nn.utils.clip_grad_norm_(self.critic_net.parameters(), max_norm=0.1, norm_type='inf') #Clip gradients #@@@@@@@@@@
        self.actor_opt.step(); 
        self.critic_opt.step();
        
        if self.scheduler is not None:
            self.scheduler.step()
            

        # Log the training stats
        self._log_weights(self.logger, self.log_weights)

        if np.random.rand()<0.003:
            # compute metrics
            actor_loss_m = actor_loss.item()
            entropy_m = entropy.item()
            adv_m = torch.norm(critic_adv.detach()).item()/16
            logprob_m = log_probs.detach().median().item()
#             eff_adv_ratio_m = (torch.sum(torch.abs(adv_eff_term)>torch.abs(adv_rew_term))/adv_eff_term.shape[1]).item()
            eff_adv_ratio_m = torch.mean( (torch.abs(adv_eff_term)>torch.abs(adv_rew_term)).float() 
                                        ).item()
#             eff_adv_ratio_m = torch.median(torch.abs(adv_eff_term/adv_rew_term)).item()
    
            print('# Agent ', self.agent.perspective+1, 
                  'Actor loss = %.3f'%actor_loss_m,
                  'Critic error norm= %.3f'% adv_m,
                  'Effort / Reward Advantage = %.3f'% eff_adv_ratio_m,
                 'Log probs Median =  %.3f'% logprob_m,
                 'Entropy = %.3f'% entropy_m)
            self._log_metrics(self.logger, actor_loss_m, adv_m, entropy_m, logprob_m, eff_adv_ratio_m)
        
        
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
                     deepcopy(ac_layers[-1-l].weight.detach().cpu().numpy()))
        
        cr_nweights = weight_specs[1]
        if cr_nweights>0: 
            cr_layers = [module[1] for module in self.critic_net.named_modules()]
            for l in range(cr_nweights):
                 logger.critic_weight_ts[l].append(
                     deepcopy(cr_layers[-1-l].weight.detach().cpu().numpy()))
        return
    

