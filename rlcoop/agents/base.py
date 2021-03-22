from collections import deque

import numpy as np
from scipy.special import expit as sigmoid
from copy import deepcopy


class MuscleModel(object):
    # Muscle with first order dynamics: 
    # y_i = \alpha x_i + (1-\alpha) y_{i-1}
    
    def __init__(self, sigma, force_max, ts=0.025, tau=0.1):
#         self.ts = ts
#         self.tau = tau
        self.sigma = sigma
        self.force_max = force_max
        
        self.alpha = ts/(ts+tau)
        self.muscle_force = 0.
        
    def _addnoise(self, action):
        mlt_noise = action*np.random.normal(0, self.sigma)
        add_noise = np.random.normal(0, self.sigma)
        noisy_cmd = action+ mlt_noise+add_noise
#         force_capped = self.force_max*np.tanh((2./self.force_max) *noisy_cmd)
        return noisy_cmd
    
    def _apply_dynamics(self, motor_command):
        self.muscle_force = self.alpha*motor_command +\
                            (1-self.alpha)*self.muscle_force
        return self.muscle_force
        
    def get_force(self, action):
        # any transformation between action and the output force, 
        # e.g. converting role or df or motor command to force.
        # Especially intended for introducing signal-dependent noise, 
        # muscle dynamics and force saturation.
        
        if self.sigma !=0:
            action = self._addnoise(action)
        
        cmd_capped = self.force_max*np.tanh(action/10.) 
#         muscle_force = self._apply_dynamics(cmd_capped)
        self.muscle_force = self.alpha*cmd_capped +\
                            (1-self.alpha)*self.muscle_force    
        return self.muscle_force 
    
    def reset(self):
        self.muscle_force = 0.
        
    def get_force_batch(self, f0_batch, command_batch):
        # For the agent trainers that need to work on batch data.
        
        if self.sigma != 0:
            raise NotImplementedError
        
        command_batch_capped = self.force_max*torch.tanh(command_batch/10.)
        
        muscle_force_batch = self.alpha*command_batch_capped +\
                            (1-self.alpha)*f0_batch
        return muscle_force_batch
    
#     def get_force_batch_nstep(self, f0_batch, command_batch, nsteps):
#         # For n-step return trainers
#         # Assumes that the motor command remains constant throughout the 
#         # nsteps.
        
#         if self.sigma != 0:
#             raise NotImplementedError
#         muscle_force_batch = deepcopy(f0_batch)
#         for i in range(nsteps):
#             muscle_force_batch = self.alpha*command_batch +\
#                                 (1-self.alpha)*muscle_force_batch
#         return muscle_force_batch
    

class Logger(): 
    def __init__(self): 
        self.entropy_ts = []
        self.adv_ts = []
        self.logprobs_ts = []
        self.actor_weight_ts = [[],[],[]]
        self.critic_weight_ts = [[],[],[]]
        self.eff_adv_ratio_ts = []
        self.actorloss_ts = []


class DyadSliderAgent(object):

    def __init__(self,
                 force_max = 1.0,
                 force_min = -1.0,
                 c_error = 1.0,
                 c_effort = 1.0,
                 perspective = 0,
                 history_length = 2):

        self.perspective = perspective

        self.force = 0.0
        self.force_max = force_max
        self.force_min = force_min

        self.observation_history = deque(maxlen = history_length)
        self.action_history = deque(maxlen = history_length)
        self.reward_history = deque(maxlen = history_length)

        self.c_error = c_error
        self.c_effort = c_effort


    def get_force(self, environment_state, record_history = True):
        observation = self.observe(environment_state)

        action = self.action_policy(observation)

        if record_history:
            self.observation_history.append(observation)
            self.action_history.append(action)

        self.force = self.action_to_force(action)

        return self.force


    def give_reward(self, reward, is_terminal, next_environment_state = None):
        self.reward_history.append(reward)

        subj_reward = self.subjective_reward(reward)


        if is_terminal:
            pass
        else:
            pass


    def observe(self, environment_state):
        x, x_dot, r, r_dot, \
        force_interaction, force_interaction_dot = environment_state

        if self.perspective % 2 == 0:
            error = x - r
            error_dot = x_dot - r_dot

        else:
            error = r - x
            error_dot = r_dot - x_dot

        observation = np.array([error, error_dot,
                                force_interaction, force_interaction_dot])

        return observation


    def action_policy(self, observation):
        return 0


    def action_to_force(self, action):
        force = action
        return force


    def effort(self, action):
        return np.sum(action)


    def subjective_reward(self, environment_reward):
        last_action = self.action_history[-1]

        reward = ((self.c_error * environment_reward)
                  + (self.c_effort * self.effort(last_action)))

        return reward

    def reset(self):
        pass



class RandomAgent(DyadSliderAgent):

    def __init__(self, action_space):
        self.action_space = action_space

    def action_policy(self, observation):
        return self.action_space.sample()




class FixedAgent(DyadSliderAgent):

    def action_policy(self, observation):
        error, error_dot, force_interaction, force_interaction_dot = observation

        if error > 0:
            action = -1
        elif error < 0:
            action = 1
        else:
            action = 0

        return action

    def action_to_force(self, action):
        if action > 0:
            force = self.force_max
        elif action < 0:
            force = self.force_min
        else:
            force = 0

        return force


class PIDAgent(DyadSliderAgent):

    def __init__(self, kP, kI, kD, **kwargs):

        self.kP = kP
        self.kI = kI
        self.kD = kD

        self.reset()

        super().__init__(**kwargs)

    def action_policy(self, observation):
        error, error_dot, force_interaction, force_interaction_dot = observation

        self.error_sum += error
        action = (self.kP * error) + (self.kI * self.error_sum) + (self.kD * error_dot)

        return action


    def reset(self):
        self.error_sum = 0.0
