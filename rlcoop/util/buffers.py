# Format of saving experiences:
# Since memory is not an issue here, let's allow redundancy and have the input and output of all.
#state, action, reward, next_state


import random
from collections import namedtuple, deque
import numpy as np

# Transition = namedtuple('Transition',
#                         ('state', 'action', 'next_state', 'reward'))

import rlcoop
from rlcoop import Transition
# from .. import Transition


class CyclicBuffer(object):
    # Adopted from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    
    # For using as batch, follow this recipe:
        #Make sure Transition is defined:
#         Transition = namedtuple('Transition',
#                                 ('state', 'action', 'next_state', 'reward'))
#         In training loop
#         transitions = memory.sample(BATCH_SIZE)
#         # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
#         # detailed explanation). This converts batch-array of Transitions
#         # to Transition of batch-arrays.
#         batch = Transition(*zip(*transitions))
    
    def __init__(self, capacity, tag=None):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.tag = tag # The tag is not used within this class. The main purpose is keeping an attribute about the data that is allowed in memory. 

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        self_size = len(self.memory)
        
        if batch_size>self_size:
            sample_size = self_size
        else:
            sample_size = batch_size
        
        return random.sample(self.memory, sample_size)
    
    
    def sample_nstep_depr(self, batch_size, nsteps):
        self_size = len(self.memory)
        
        if batch_size>self_size:
            sample_size = self_size
        else:
            sample_size = batch_size
            
        idxs = np.random.randint(len(self.memory)-nsteps, size=sample_size)
        
        samples = [[self.memory[idx+i] for i in range(nsteps)] for idx in idxs]
        return samples
    
    def sample_nstep(self, batch_size, nsteps):
        self_size = len(self.memory)
        
        if batch_size>self_size:
            sample_size = self_size
        else:
            sample_size = batch_size
            
        idxs = np.random.randint(len(self.memory)-nsteps, size=sample_size)
        
#         samples = [[self.memory[idx+i] for i in range(nsteps)] for idx in idxs]
        samples = [ self.memory[idx: idx+nsteps] for idx in idxs]
#         samples = [ Transition(*zip(*self.memory[idx: idx+nsteps-1])) for idx in idxs]
        return samples
    
    
    def __len__(self):
        return len(self.memory)


class ReplayMemory(object):
    # Adopted from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    
    # For using as batch, follow this recipe:
        #Make sure Transition is defined:
#         Transition = namedtuple('Transition',
#                                 ('state', 'action', 'next_state', 'reward'))
#         In training loop
#         transitions = memory.sample(BATCH_SIZE)
#         # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
#         # detailed explanation). This converts batch-array of Transitions
#         # to Transition of batch-arrays.
#         batch = Transition(*zip(*transitions))
    
    def __init__(self, capacity, tag=None):
        self.capacity = capacity
        self.memory = deque([], maxlen=capacity)
        pdist = [np.exp(-3.*t/capacity) for t in range(capacity)];  pdist[-1]=0.
        self.pdist = np.asarray(pdist)/np.sum(pdist)
#         self.position = 0
        self.tag = tag # The tag is not used within this class. The main purpose is keeping an attribute about the data that is allowed in memory. 

    def push(self, *args):
        """Saves a transition."""
#         if len(self.memory) < self.capacity:
#             self.memory.appendleft(None)
        self.memory.appendleft(Transition(*args))
#         self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        self_size = len(self.memory)
        
        if batch_size>self_size:
            sample_size = self_size
        else:
            sample_size = batch_size
            
#         if len(self.memory)==self.capacity:
#             return np.random.choice(self.memory+deque([None]), size=sample_size-1, p=self.pdist, replace=False)
#         else:
#             return random.sample(self.memory, sample_size)
        return random.sample(self.memory, sample_size) # no prob dist
# random.choices # doesn't have the option replace=false


    def sample_nstep(self, batch_size, nsteps):
        self_size = len(self.memory)
        
        if batch_size>self_size:
            sample_size = self_size
        else:
            sample_size = batch_size
            
        idxs = np.random.randint(len(self.memory)-nsteps, size=sample_size)
        
        samples = [[self.memory[idx+i] for i in range(nsteps)] for idx in idxs]
        return samples


    def __len__(self):
        return len(self.memory)
    

    
class ReplayMemoryList(object):
    # Adopted from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    
    # For using as batch, follow this recipe:
        #Make sure Transition is defined:
#         Transition = namedtuple('Transition',
#                                 ('state', 'action', 'next_state', 'reward'))
#         In training loop
#         transitions = memory.sample(BATCH_SIZE)
#         # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
#         # detailed explanation). This converts batch-array of Transitions
#         # to Transition of batch-arrays.
#         batch = Transition(*zip(*transitions))
    
    def __init__(self, capacity, tag=None):
        self.capacity = capacity
        self.memory = [] #deque([], maxlen=capacity)
#         pdist = [np.exp(-3.*t/capacity) for t in range(capacity)];  pdist[-1]=0.
#         self.pdist = np.asarray(pdist)/np.sum(pdist)
#         self.position = 0
        self.tag = tag # The tag is not used within this class. The main purpose is keeping an attribute about the data that is allowed in memory. 

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(Transition(*args))
        else:
            i = np.random.randint(self.capacity)
            self.memory[i] = Transition(*args)
            
            
        
#         self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        self_size = len(self.memory)
        
        if batch_size>self_size:
            sample_size = self_size
        else:
            sample_size = batch_size
            
#         if len(self.memory)==self.capacity:
#             return np.random.choice(self.memory+deque([None]), size=sample_size-1, p=self.pdist, replace=False)
#         else:
#             return random.sample(self.memory, sample_size)
        return random.sample(self.memory, sample_size) # no prob dist
# random.choices # doesn't have the option replace=false


    def sample_nstep(self, batch_size, nsteps):
        # Can\'t have sample_nstep!
        raise NotImplementedError


    def __len__(self):
        return len(self.memory)

    
# class SupervisionBuffer():
#     def __init__(self, maxsize, tol=0.01):
#         self.maxsize = maxsize
#         self.data = deque([], maxlen=maxsize)
#         self.target = deque([], maxlen=maxsize)
#         self.tolerance = tol
# #         self.size = 0
            
#     def append(self, data, target):
#         self.data.appendleft(data)
#         self.target.appendleft(target)
# #         self.size =len(self.data)
        
#     def sample_batch(self, size):
#         self_size = len(self.data)
#         if size>self_size:
#             sample_size = self_size
#         else:
#             sample_size = size
#         mask = np.random.randint(self_size, size=sample_size)
#         return narr(self.data)[mask], narr(self.target)[mask]
    
#     def __len__(self):
#         return len(self.data)
