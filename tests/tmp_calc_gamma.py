# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 19:27:23 2021

@author: Saber
"""
import numpy as np

# Calculating gamma for RL
att = np.linspace(0,1,100)
sum_at_1s = 0.7
ytt = (1/att)*(1-np.exp(-att))-sum_at_1s
# ytt=0 at att=0.075
gamma = np.exp(-0.75)

sum_discount = np.sum([0.01*gamma**t for t in att])
# sum_discount should be the same as sum_at_1s
# For sum_at_1s = 0.7, we get gamma=0.47

#Note: gamma_effort=1 because the relationship between force-effort is immediate.
# or perhaps, adv_effort[t] = effort[t] 