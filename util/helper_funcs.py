import numpy as np
import collections
from scipy.stats import truncnorm


def ndrms(x, axis=0):
    return np.sqrt(np.mean(x**2, axis=axis))

def roots2ipd(r1, r2, r3, m, b):
    coefs = np.poly([r1, r2, r3])
    Ki = coefs[3]*m
    Kp = coefs[2]*m
    Kd = coefs[1]*m-b
    
    return (Ki, Kp, Kd)

def ipd2roots(ki, kp, kd, m, b):
    a0 = m
    a1 = b+kd
    a2 = kp
    a3 = ki

    return np.roots([a0, a1, a2, a3])


def norm_sampler(mu, sigma, size):
    lower, upper = 0, 1
    # mu, sigma = 0.5, 0.1
    samples = truncnorm.rvs((lower - mu) / sigma, (upper - mu) / sigma, 
                        loc=mu, scale=sigma, size=size)
    return samples


# One step of Runge-Kutta fourth order
def rk4_step(f, x0, u=0, t=0, h=0.01):

    k1 = h*f(x0, u, t)
    k2 = h*f(x0+k1, u, t+0.5*h)
    k3 = h*f(x0+k2, u, t+0.5*h)
    k4 = h*f(x0+k3, u, t+h)
    
    x1 = x0 + (k1 + 2*k2 + 2*k3 + k4)/6
    
    return tuple(x1)


def trajectory_generator(tstep, duration, traj_specs=None, max_amp=None, normalize=True):
    # traj_specs is a list of traj_spec. 
    # Each traj_spec is a list of sinosoid descriptors. 
    # Each descriptor comprises 3 scalars: amplitude, frequency, phase.
    # Hence, traj_specs is a 3 dimensional list.
    n = np.arange(0, duration, tstep)
    trajs = []
    
    # trajectory is a sum of sinuosoids
    if traj_specs==None:
        traj_specs = [[[1, 0.09,2], [1, 0.18,0], [0.5,0.027,1], [0.2,0.045,0.5], [0.2, 0.063,1.3]]]
    elif len(traj_specs[0][0])!= 3: #Warning: this only checks for the first descriptor!
        raise ValueError 
    # traj_specs[i][0]: frequency of the ith sinusoid
    # traj_specs[i][1]: phase of the ith sinusoid
    
    for traj_spec in traj_specs:
        x =np.zeros_like(n)
        sum_amp =0
        for i in range(len(traj_spec)):
            x += traj_spec[i][0]* np.cos(2*np.pi*n *traj_spec[i][1] +traj_spec[i][2])
            sum_amp += traj_spec[i][0]
        if max_sum_amp is not None:
            traj = (max_sum_amp/sum_amp) *x   
        else:
            traj = x 
        trajs.append(traj)
    return n, trajs

def traj_max_acc(traj_specs, max_sum_amp, egg_mass, egg_fric):
    # Calculate how much force is required to follow the trajectory specified
    acc_max=[]
    for tr_sp in traj_specs:
        sum_acc =0; sum_amp =0; 
        for snsd in tr_sp:
            sum_acc += abs(snsd[0]*((2*np.pi*snsd[1])**2))*egg_mass + abs(snsd[0]*(2*np.pi*snsd[1]))*egg_fric 
            sum_amp += snsd[0]
        acc_max.append(sum_acc*max_sum_amp/sum_amp)
    fc_req_max = max(acc_max)#*egg_mass
    return fc_req_max

    
class ushaped_f():
    # For the penalty function of the egg
    def __init__(self, lb, ub, fb, ls=-10, rs=2):#fb, =[0.]):
        # lb, ub: lower bound and upper bound
        # fb: output of the function for the boundary force
        # ls, rs: left line slope, right line slope
        self.ls = ls; self.rs = rs;
        self.x0l = fb -ls*lb #left line intercept
        self.x0r = fb -rs*ub #right line intercept
        self.xbpl = lb-fb/ls #left breaking point 
        self.xbpr = ub-fb/rs #right breaking point
        
    def apply(self, x):
        if x<self.xbpl:
            return self.x0l + self.ls*x
        elif x>self.xbpr:
            return self.x0r + self.rs*x
        else:
            return 0.

def slide_f(x, x_settle):
    # returns the output of a slide shaped function, given its input
    # x is a vector, representing the input
    # x_settle is a scalar, indicating the index after which the output is 1
    y = np.ones_like(x)
    for i, item in enumerate(x):
        if item>x_settle:
            break
        else:
            y[i] = item/x_settle
    
    return y
   
    
def sampler(list1, n_samples):
    # Receives a 1D list and the number of samples.
    # Returns a 1D array containing n samples from that list, selected as such:
    # xtt = np.arange(10)
    # print(sampler(xtt, 3))
    # >> [9. 6. 3.] 
    interval = int(len(list1)/n_samples)
    arr2 = np.zeros(n_samples)
    for i in range(n_samples):
        arr2[i] = list1[-1-i*interval]
    return arr2

def bound(x, bnd, symm=True):
    # For keeping forces within bounds
    if symm is False:
        raise NotImplementedError
    if abs(x)>bnd:
        if x>0:
            return bnd
        else:
            return -bnd
    else:
        return x
    

def _get_iterable(x):
    if isinstance(x, collections.Iterable):
        return x
    else:
        return (x,)

