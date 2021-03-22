import sys, time
from copy import deepcopy

# Import custom scripts
import sys,os, inspect
SCRIPT_DIR = os.path.realpath(os.path.dirname(inspect.getfile(inspect.currentframe())))
PARENT_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, '..'))
# sys.path.append(os.path.join(PARENT_PATH,'configs'))
sys.path.append(os.path.join(PARENT_PATH,'agents'))
# sys.path.append(os.path.join(PARENT_PATH,'algos'))
sys.path.append(os.path.join(PARENT_PATH,'util'))
# sys.path.append(os.path.join(PARENT_PATH,'envs'))

from util import buffers
from agents import base

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