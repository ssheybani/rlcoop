from copy import deepcopy

import context
from context import rlcoop
from rlcoop.util import buffers
from rlcoop.agents import base

n_eval = 20 #In benchmarking: The number of episodes the performance is averaged over in policy evaluation.

# Buffer-related
buffer_max_size = 300000# 1000 episodes
buffer1 = buffers.CyclicBuffer(buffer_max_size)
buffer2 = deepcopy(buffer1)

logger1 = base.Logger()
logger2 = base.Logger()

force_max=20.; #force_min=-20.
sigma = 0. #0.3;
tau = 0.09 #0.05 #

muscle1 = base.MuscleModel(sigma, force_max, ts=0.025, tau=tau)
muscle2 = deepcopy(muscle1)