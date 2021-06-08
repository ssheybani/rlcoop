import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import functools, itertools
from collections import namedtuple

def sumOfSines(t, f,a=None, p=None, ret_deriv=True):
    if a is None:
        a = np.ones_like(f)
    if p is None:
        p = np.zeros_like(f)

    sos = np.zeros_like(t)
    sosdot = np.zeros_like(t)
    for ff, aa, pp in zip(f,a,p):
        sos += aa * np.sin(2*np.pi*ff*t +pp)
        sosdot += aa *2*np.pi*ff* np.cos(2*np.pi*ff*t +pp)

    if ret_deriv:
        return sos, sosdot
    else:
        return sos

def RK4Step(x0, f, h, u, t=0):
    k1 = h * f(x0, u, t)
    k2 = h * f(x0 + k1/2, u, t + h / 2)
    k3 = h * f(x0 + k2/2, u, t + h / 2)
    k4 = h * f(x0 + k3, u, t + h)
    return x0 + (k1 + 2 * k2 + 2 * k3 + k4) / 6

def EulerStep(x0, f, h, u=0, t=0):
    return x0+h*f(x0, u, t)

mat_mul_series = lambda *mats: functools.reduce(np.matmul, mats)#, initializer=None)
# Run_Data = namedtuple('Run_Data', ['signals', 'controllers', 'e_costs'])
Run_Data = namedtuple('Run_Data', ['signals', 'controllers', 'e_costs', 'loss'])


def term_i(i, A, B, C):
    CA = C 
    j=0
    while j<i:
       CA = np.matmul(CA, A)
       j+=1
    return np.matmul(CA, B)

def A_power_sum(A, i):
    # returns I+A+A^2+...+A^i
    # A is a square matrix
    # A better approach may come from (I-A^i )/(I-A)
    sum_p = np.eye(len(A))
    for j in range(1,i+1):
        sum_p = sum_p + np.linalg.matrix_power(A, j)
    return sum_p

def A_power_sum2(A, i):
    # returns I+A+A^2+...+A^i
    # A is a square matrix
    # Using the geometric series sum rule: (I-A^i )/(I-A)
    # Error prone becuase of the error in the inversion of I-A
    eyeA = np.eye(len(A))
    # print(np.linalg.det(eyeA-A))
    # print(np.dot(np.linalg.pinv(eyeA-A), eyeA-A))
    
    return eyeA + np.matmul(np.linalg.inv(eyeA-A),
                     eyeA - np.linalg.matrix_power(A, i+1)
                     )