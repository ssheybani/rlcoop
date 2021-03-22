import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch import distributions
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal

class Myinit():
    def eye(tensor, coef):
        with torch.no_grad():
            tensor = coef * torch.eye(*tensor.shape, requires_grad=tensor.requires_grad)
        return torch.nn.Parameter(tensor)
    
    
class NetL1(nn.Module):
# Inputs:
# r, r', x, x', f_n, f_n', f1
# Outputs:
# dr, dr', dx, dx', df_n, df_n'
    def __init__(self, nin, nout):
        super().__init__()
        self.fc1 = nn.Linear(nin, nout, bias=False)
        torch.nn.init.normal_(self.fc1.weight, mean=0.0, std=0.03)
        
    def forward(self, x):
        out = self.fc1(x)
        return out
    

class NetL1b(nn.Module):
# Inputs:
# r, r', x, x', f_n, f_n', f1
# Outputs:
# dr, dr', dx, dx', df_n, df_n'
    def __init__(self, nin, nout):
        super().__init__()
        self.fc1 = nn.Linear(nin, nout, bias=True)
        
    def forward(self, x):
        out = self.fc1(x)
        return out
    
    
class NetRelu1L1(nn.Module):

    def __init__(self, nin, nout, n_hidden=32):
        super().__init__()
        self.fc1 = nn.Linear(nin, n_hidden)
#         self.bn1 = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(n_hidden, nout)
        torch.nn.init.normal_(self.fc2.weight, mean=0.0, std=0.03)
#         self.fc3 = nn.Linear(64, 64)
#         self.fc4 = nn.Linear(64, nout)
        
    def forward(self, x):
        z1 = F.relu(self.fc1(x))#self.bn1(F.relu(self.fc1(x)))
        out = self.fc2(z1)
#         z3 = F.relu(self.fc3(z2))
#         out = self.fc4(z3)
        return out


class NetRelu3L1(nn.Module):
# Inputs:
# r, r', x, x', f_n, f_n', f1
# Outputs:
# dr, dr', dx, dx', df_n, df_n'
#     def __init__(self, nin, nout):
#         super(PredictiveNet, self).__init__()
#         self.fc1 = nn.Linear(nin, 20)
#         self.fc2 = nn.Linear(20, nout)
        
#     def forward(self, x):
#         z1 = F.relu(self.fc1(x))
#         out = self.fc2(z1)
#         return out

    def __init__(self, nin, nout):
        super().__init__()
        self.fc1 = nn.Linear(nin, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, nout)
        
#         torch.nn.init.normal_(self.fc1.weight, mean=0.0, std=0.03)
#         torch.nn.init.normal_(self.fc2.weight, mean=0.0, std=0.03)
#         torch.nn.init.normal_(self.fc3.weight, mean=0.0, std=0.03)
        torch.nn.init.normal_(self.fc4.weight, mean=0.0, std=0.03)
        
    def forward(self, x):
        z1 = F.relu(self.fc1(x))
        z2 = F.relu(self.fc2(z1))
        z3 = F.relu(self.fc3(z2))
        out = self.fc4(z3)
        return out
    
    

class NetGlu2L1(nn.Module):
# Inputs:
# r, r', x, x', f_n, f_n', f1
# Outputs:
# Q
    def __init__(self, nin, nout):
        super().__init__()
        self.fc1 = nn.Linear(nin, 128)
        self.fc2 = nn.Linear(64, 128)
#         self.fc3 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(64, nout)
        torch.nn.init.normal_(self.fc3.weight, mean=0.0, std=0.03)
        
    def forward(self, x):
        z1 = F.glu(self.fc1(x))
        z2 = F.glu(self.fc2(z1))
        out = self.fc3(z2)
#         z3 = F.relu(self.fc3(z2))
#         out = self.fc4(z3)
        return out


class NetReGlReGlL(nn.Module):
# Inputs:
# r, r', x, x', f_n, f_n', f1
# Outputs:
# Q
    def __init__(self, nin, nout, n_hidden=64):
        super().__init__()
        self.fc1 = nn.Linear(nin, 128)
        self.fc2 = nn.Linear(128, 128)
#         self.fc3 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, n_hidden*2)
        self.fc5 = nn.Linear(n_hidden, nout)
        torch.nn.init.normal_(self.fc5.weight, mean=0.0, std=0.03)
        
    def forward(self, x, return_hidden=False):
        z1 = F.relu(self.fc1(x))
        z2 = F.glu(self.fc2(z1))
        z3 = F.relu(self.fc3(z2))
        z4 = F.glu(self.fc4(z3))
        out = self.fc5(z4)
#         z3 = F.relu(self.fc3(z2))
#         out = self.fc4(z3)
        if return_hidden:
            return out, z4
        else:
            return out

        
class NetGlL(nn.Module):
# Inputs:
# r, r', x, x', f_n, f_n', f1
# Outputs:
# Q

    def __init__(self, nin, nout, n_hidden=128, masked_ftrs=[], ftr_normalizer=None,
                device=None):
        super().__init__()
        
        if device is None:
            device = torch.device("cpu")
            
        self.masked_ftrs = masked_ftrs
        if len(masked_ftrs)>0:
            mask = torch.ones(nin, device=device)
            mask[masked_ftrs]=0
            self.mask = mask.ge(0.5) 
            nin = nin-len(masked_ftrs)
            
        self.ftr_normalizer = ftr_normalizer
        if self.ftr_normalizer is not None:
            self.ftr_normalizer = torch.tensor(ftr_normalizer, dtype=torch.float32, 
                                               device=device)
            
        self.fc1 = nn.Linear(nin, n_hidden)
        self.fc2 = nn.Linear(int(n_hidden/2), nout)
        
        torch.nn.init.normal_(self.fc1.weight, mean=0.0, std=5.)
        torch.nn.init.normal_(self.fc2.weight, mean=0.0, std=0.03)
        
    def forward(self, x, return_hidden=False):
        if len(self.masked_ftrs)>0:
            x_ = torch.masked_select(x, self.mask).view(x.shape[0],-1) #x[:,self.mask] #
        else:
            x_ = x
            
        if self.ftr_normalizer is not None:
            x_ = x_*self.ftr_normalizer
        z1 = F.glu(self.fc1(x_))
        out = self.fc2(z1)
        
        if return_hidden:
            return out, z1
        else:
            return out

        


class NetReGlReGlL_bn(nn.Module):
# Inputs:
# r, r', x, x', f_n, f_n', f1
# Outputs:
# Q
    def __init__(self, nin, nout, n_hidden=64, batch_n=True):
        super().__init__()
        self.fc1 = nn.Linear(nin, 128)
        self.bn1 = nn.BatchNorm1d(128)
        
        self.fc2 = nn.Linear(128, 128)
        self.bn2 = nn.BatchNorm1d(128)
        
        self.fc3 = nn.Linear(64, 128)
        self.bn3 = nn.BatchNorm1d(128)
        
        self.fc4 = nn.Linear(128, n_hidden*2)
        self.bn4 = nn.BatchNorm1d(n_hidden*2)
        
        self.fc5 = nn.Linear(n_hidden, nout)
        torch.nn.init.normal_(self.fc5.weight, mean=0.0, std=0.03)
        
    def forward(self, x, return_hidden=False):
        z1 = F.relu(self.fc1(x))
        z2 = F.glu(self.bn2(self.fc2(z1)))
        z3 = F.relu(self.fc3(z2))
        z4 = F.glu(self.bn4(self.fc4(z3)))
        out = self.fc5(z4)
#         z3 = F.relu(self.fc3(z2))
#         out = self.fc4(z3)
        if return_hidden:
            return out, z4
        else:
            return out
        
        
        
class NetL1G_old(nn.Module):
# Inputs:
# r, r', x, x', f_n, f_n', f1
# Outputs:
# dr, dr', dx, dx', df_n, df_n'
    def __init__(self, nin, log_std_min=-20, log_std_max=0.):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.fc1 = nn.Linear(nin, 2, bias=False)
#         torch.nn.init.normal_(self.fc1.weight, mean=0.0, std=0.03)
        torch.nn.init.normal_(self.fc1.weight, mean=0.0, std=0.1)
        
    def forward(self, x):
        out = self.fc1(x)
#         if out.dim()>1:
        mu, log_std = out[:,0], out[:,1]
#         else:
#             mu, log_std = out
#         sig = nn.Softplus()(sig_)
        
        log_std = log_std -1. #@@@@@@ to improve the range
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        sig = log_std.exp()
        
        dist = distributions.normal.Normal(mu, sig)
#         if out.dim()>1:
#             print('mu, sigma', mu, sig)
        return dist


class NetL1G(nn.Module):
    
    def __init__(self, nin, std_min=0.03, std_max=2., 
                 masked_ftrs=[], ftr_normalizer=None):
        super().__init__()
        self.std_min = std_min
        self.std_max = std_max
        
        self.masked_ftrs = masked_ftrs
        self.ftr_normalizer = ftr_normalizer
        if self.ftr_normalizer is not None:
            self.ftr_normalizer = np.array(ftr_normalizer, dtype=np.float32)
        
        if len(masked_ftrs)>0:
            mask = torch.ones(nin)
            mask[masked_ftrs]=0
            self.mask = mask.ge(0.5) 
            nin = nin-len(masked_ftrs)
        
        self.fc1 = nn.Linear(nin, 2, bias=False)
#         torch.nn.init.normal_(self.fc1.weight, mean=0.0, std=0.03)
        torch.nn.init.normal_(self.fc1.weight, mean=0.0, std=0.1)
        
    def forward(self, x, eps=0):
        if len(self.masked_ftrs)>0:
            x_ = x[:,self.mask] #torch.masked_select(x, self.mask)
        else:
            x_ = x
        if self.ftr_normalizer is not None:
            x_ = x_*self.ftr_normalizer
            
        out = self.fc1(x_)
        mu, sigma_adjust = out[:,0], out[:,1] 
        
        sigma_perf = np.exp(3.5*eps-4.) #between 0.03, 0.6
        sigma_adjust = 0.5*torch.tanh(sigma_adjust) #between -0.5, 0.5
        sig_ = sigma_perf +sigma_adjust
        sig = torch.clamp(sig_, self.std_min, self.std_max)
        
        dist = distributions.normal.Normal(mu, sig)
        return dist
        
class NetL1G_bn(nn.Module):
# Inputs:
# r, r', x, x', f_n, f_n', f1
# Outputs:
# dr, dr', dx, dx', df_n, df_n'
    def __init__(self, nin, log_std_min=-20, log_std_max=0.):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.bn1 = nn.BatchNorm1d(nin)
        self.fc1 = nn.Linear(nin, 2, bias=False)
#         torch.nn.init.normal_(self.fc1.weight, mean=0.0, std=0.03)
        torch.nn.init.normal_(self.fc1.weight, mean=0.0, std=0.1)
        
    def forward(self, x):
        out = self.fc1(self.bn1(x))
#         if out.dim()>1:
        mu, log_std = out[:,0], out[:,1]
#         else:
#             mu, log_std = out
#         sig = nn.Softplus()(sig_)
        
        log_std = log_std -1. #@@@@@@ to improve the range
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        sig = log_std.exp()
        
        dist = distributions.normal.Normal(mu, sig)
#         if out.dim()>1:
#             print('mu, sigma', mu, sig)
        return dist

    
class NetRelu1L1G(nn.Module):

    def __init__(self, nin, n_hidden=64, log_std_min=-20, log_std_max=0.):
        super().__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.fc1 = nn.Linear(nin, n_hidden)
        
#         self.bn1 = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(n_hidden, 2)
        # Initialization
#         torch.nn.init.normal_(self.fc1.weight, mean=0.0, std=0.1)
        torch.nn.init.normal_(self.fc2.weight, mean=0.0, std=0.03)
#         self.fc3 = nn.Linear(64, 64)
#         self.fc4 = nn.Linear(64, nout)
        
    def forward(self, x):
        z1 = F.leaky_relu(self.fc1(x))#self.bn1(F.relu(self.fc1(x)))
        out = self.fc2(z1)
        if out.dim()>1:
            mu, log_std = out[:,0], out[:,1]
        else:
            mu, log_std = out
#         sig = nn.Softplus()(sig_)
        log_std = log_std -1.
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        sig = log_std.exp()
        
        dist = distributions.normal.Normal(mu, sig)
#         if out.dim()>1:
#             print('mu, sigma', mu, sig)
        return dist

class NetRelu3L1G(nn.Module):
    
    def __init__(self, nin, n_hidden=32, log_std_min=-20, log_std_max=0.):
        super().__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.fc1 = nn.Linear(nin, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_hidden)
        self.fc4 = nn.Linear(n_hidden, 2)
        
#         torch.nn.init.normal_(self.fc1.weight, mean=0.0, std=0.03)
#         torch.nn.init.normal_(self.fc2.weight, mean=0.0, std=0.03)
#         torch.nn.init.normal_(self.fc3.weight, mean=0.0, std=0.03)
        torch.nn.init.kaiming_normal_(self.fc1.weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')
        torch.nn.init.kaiming_normal_(self.fc2.weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')
        torch.nn.init.kaiming_normal_(self.fc3.weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')
        torch.nn.init.normal_(self.fc4.weight, mean=0.0, std=0.1/np.sqrt(n_hidden))
    
        torch.nn.init.uniform_(self.fc1.bias, -0.2, 0.2)
        torch.nn.init.uniform_(self.fc2.bias, -0.2, 0.2)
        torch.nn.init.uniform_(self.fc2.bias, -0.2, 0.2)
#         self.bn1 = nn.BatchNorm1d(32)
        
    def forward(self, x):
        z1 = F.leaky_relu(self.fc1(x))#self.bn1(F.relu(self.fc1(x)))
        z2 = F.leaky_relu(self.fc2(z1))
        z3 = F.leaky_relu(self.fc3(z2))
        
        out = self.fc4(z3)

        mu, log_std = out[:,0], out[:,1]

#         sig = nn.Softplus()(sig_)
        log_std = log_std -1.
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        sig = log_std.exp()
        
        dist = distributions.normal.Normal(mu, sig)
#         if out.dim()>1:
#             print('mu, sigma', mu, sig)
        return dist


class NetReGlReGlLG(nn.Module):
# Inputs:
# r, r', x, x', f_n, f_n', f1
# Outputs:
# Q
    def __init__(self, nin, n_hidden=128, log_std_min=-20, log_std_max=0.):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.fc1 = nn.Linear(nin, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
#         self.fc3 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(int(n_hidden/2), n_hidden)
        self.fc4 = nn.Linear(n_hidden, n_hidden)
        self.fc5 = nn.Linear(int(n_hidden/2), 2)
        
        torch.nn.init.uniform_(self.fc1.bias, a=-1., b=1.) #@@@@@@ for piecewise linear
        torch.nn.init.normal_(self.fc4.weight, mean=0.0, std=0.5)
        torch.nn.init.normal_(self.fc5.weight, mean=0.0, std=0.01)
        
    def forward(self, x):
        z1 = F.relu(self.fc1(x))
        z2 = F.glu(self.fc2(z1))
        z3 = F.relu(self.fc3(z2))
        z4 = F.glu(self.fc4(z3))
        out = self.fc5(z4)
        
        mu, log_std = out[:,0], out[:,1]

#         sig = nn.Softplus()(sig_)
        log_std = log_std -1.
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        sig = log_std.exp()
        
        dist = distributions.normal.Normal(mu, sig)
#         if out.dim()>1:
#             print('mu, sigma', mu, sig)
        return dist


class NetGlLG(nn.Module):
# Inputs:
# r, r', x, x', f_n, f_n', f1
# Outputs:
# Q
    def __init__(self, nin, n_hidden=128, log_std_min=-20, log_std_max=0., masked_ftrs=[], ftr_normalizer=None):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.masked_ftrs = masked_ftrs
        self.ftr_normalizer = ftr_normalizer
        if self.ftr_normalizer is not None:
            self.ftr_normalizer = np.array(ftr_normalizer, dtype=np.float32)
        
        if len(masked_ftrs)>0:
            mask = torch.ones(nin)
            mask[masked_ftrs]=0
            self.mask = mask.ge(0.5) 
            nin = nin-len(masked_ftrs)
        self.fc1 = nn.Linear(nin, n_hidden)
        self.fc2 = nn.Linear(int(n_hidden/2), 2, bias=False)
        
        torch.nn.init.normal_(self.fc1.weight, mean=0.0, std=5.)
        torch.nn.init.normal_(self.fc2.weight, mean=0.0, std=0.01)
        
    def forward(self, x):
        if len(self.masked_ftrs)>0:
            x_ = x[:,self.mask] #torch.masked_select(x, self.mask)
        else:
            x_ = x
        if self.ftr_normalizer is not None:
            x_ = x_*self.ftr_normalizer
        z1 = F.glu(self.fc1(x_))
        out = self.fc2(z1)
        
        mu, log_std = out[:,0], out[:,1]

#         sig = nn.Softplus()(sig_)
        log_std = log_std -1.
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        sig = log_std.exp()
        
        dist = distributions.normal.Normal(mu, sig)
#         if out.dim()>1:
#             print('mu, sigma', mu, sig)
        return dist


class NetGlLMu(nn.Module):
# Inputs:
# r, r', x, x', f_n, f_n', f1
# Outputs:
# Q
    def __init__(self, nin, n_hidden=128, std_min=0.03, std_max=2., masked_ftrs=[], ftr_normalizer=None):
        super().__init__()
        
        self.std_min = std_min
        self.std_max = std_max
        
        self.masked_ftrs = masked_ftrs
        self.ftr_normalizer = ftr_normalizer
        if self.ftr_normalizer is not None:
            self.ftr_normalizer = np.array(ftr_normalizer, dtype=np.float32)
        
        if len(masked_ftrs)>0:
            mask = torch.ones(nin)
            mask[masked_ftrs]=0
            self.mask = mask.ge(0.5) 
            nin = nin-len(masked_ftrs)
        self.fc1 = nn.Linear(nin, n_hidden)
        self.fc2 = nn.Linear(int(n_hidden/2), 2, bias=False)
        
        torch.nn.init.normal_(self.fc1.weight, mean=0.0, std=5.)
        torch.nn.init.normal_(self.fc2.weight, mean=0.0, std=0.01)
        
    def forward(self, x, eps=0):
        # eps indicates the amount of exploration which will
        # affect the variance of actions generated.
        if len(self.masked_ftrs)>0:
            x_ = x[:,self.mask] #torch.masked_select(x, self.mask)
        else:
            x_ = x
        if self.ftr_normalizer is not None:
            x_ = x_*self.ftr_normalizer
        z1 = F.glu(self.fc1(x_))
        out = self.fc2(z1)
        
        mu, sigma_adjust = out[:,0], out[:,1] 
        
        sigma_perf = np.exp(3.5*eps-4.) #between 0.03, 0.6
        sigma_adjust = 0.5*torch.tanh(sigma_adjust) #between -0.5, 0.5
        sig_ = sigma_perf +sigma_adjust
        
#         if np.random.rand()<0.0001:
#             print('sigma_adjust , sigma_perf= ', sigma_adjust.item(), sigma_perf)
        sig = torch.clamp(sig_, self.std_min, self.std_max)
        
        dist = distributions.normal.Normal(mu, sig)
#         if out.dim()>1:
#             print('mu, sigma', mu, sig)
        return dist


class NetGlReLMu_old(nn.Module):
# Inputs:
# r, r', x, x', f_n, f_n', f1
# Outputs:
# Q
    def __init__(self, nin, n_hidden=128, std_min=0.03, std_max=2., masked_ftrs=[], ftr_normalizer=None):
        super().__init__()
        
        self.std_min = std_min
        self.std_max = std_max
        
        self.masked_ftrs = masked_ftrs
        self.ftr_normalizer = ftr_normalizer
        if self.ftr_normalizer is not None:
            self.ftr_normalizer = np.array(ftr_normalizer, dtype=np.float32)
        
        if len(masked_ftrs)>0:
            mask = torch.ones(nin)
            mask[masked_ftrs]=0
            self.mask = mask.ge(0.5) 
            nin = nin-len(masked_ftrs)
        self.fc1 = nn.Linear(nin, n_hidden)
        self.fc2 = nn.Linear(int(n_hidden/2), int(n_hidden/2), bias=False)
        self.fc3 = nn.Linear(int(n_hidden/2), 2, bias=False)
        
        torch.nn.init.normal_(self.fc1.weight, mean=0.0, std=10.)
        torch.nn.init.eye_(self.fc2.weight)#kaiming_normal_(self.fc2.weight, a=0, mode='fan_in', nonlinearity='relu')
        torch.nn.init.normal_(self.fc3.weight, mean=0.0, std=0.01)
        
    def forward(self, x, eps=0):
        # eps indicates the amount of exploration which will
        # affect the variance of actions generated.
        if len(self.masked_ftrs)>0:
            x_ = x[:,self.mask] #torch.masked_select(x, self.mask)
        else:
            x_ = x
        if self.ftr_normalizer is not None:
            x_ = x_*self.ftr_normalizer
        z1 = F.glu(self.fc1(x_))
        z2 = F.relu(self.fc2(z1))
        out = self.fc3(z2)
        
        mu, sigma_adjust = out[:,0], out[:,1] 
        
        sigma_perf = np.exp(3.5*eps-4.) #between 0.03, 0.6
        sigma_adjust = 0.5*torch.tanh(sigma_adjust) #between -0.5, 0.5
        sig_ = sigma_perf +sigma_adjust
        
#         if np.random.rand()<0.0001:
#             print('sigma_adjust , sigma_perf= ', sigma_adjust.item(), sigma_perf)
        sig = torch.clamp(sig_, self.std_min, self.std_max)
        
        dist = distributions.normal.Normal(mu, sig)
#         if out.dim()>1:
#             print('mu, sigma', mu, sig)
        return dist


class NetGlReLMu(nn.Module):
# Inputs:
# r, r', x, x', f_n, f_n', f1
# Outputs:
# Q
    def __init__(self, nin, n_hidden=128, std_min=0.03, std_max=2., masked_ftrs=[], ftr_normalizer=None):
        super().__init__()
        
        self.std_min = std_min
        self.std_max = std_max
        
        self.masked_ftrs = masked_ftrs
        self.ftr_normalizer = ftr_normalizer
        if self.ftr_normalizer is not None:
            self.ftr_normalizer = np.array(ftr_normalizer, dtype=np.float32)
        
        if len(masked_ftrs)>0:
            mask = torch.ones(nin)
            mask[masked_ftrs]=0
            self.mask = mask.ge(0.5) 
            nin = nin-len(masked_ftrs)
        self.fc1 = nn.Linear(nin, n_hidden)
        self.fc21 = nn.Linear(int(n_hidden/2), int(n_hidden/2), bias=False)
        self.fc22 = nn.Linear(int(n_hidden/2), int(n_hidden/2), bias=False)
        self.fc31 = nn.Linear(int(n_hidden), 2, bias=False)
        self.fc32 = nn.Linear(int(n_hidden/2), 2, bias=False)
        
        
        torch.nn.init.normal_(self.fc1.weight, mean=0.0, std=5.)
        self.fc21.weight = Myinit.eye(self.fc21.weight, .2)#kaiming_normal_(self.fc2.weight, a=0, mode='fan_in', nonlinearity='relu')
        self.fc22.weight = Myinit.eye(self.fc22.weight, -.2)#kaiming_normal_(self.fc2.weight, a=0, mode='fan_in', nonlinearity='relu')
        torch.nn.init.normal_(self.fc31.weight, mean=0.0, std=0.003)
        torch.nn.init.normal_(self.fc32.weight, mean=0.0, std=0.01)
        
    def forward(self, x, eps=0):
        # eps indicates the amount of exploration which will
        # affect the variance of actions generated.
        if len(self.masked_ftrs)>0:
            x_ = x[:,self.mask] #torch.masked_select(x, self.mask)
        else:
            x_ = x
        if self.ftr_normalizer is not None:
            x_ = x_*self.ftr_normalizer
            
        z1 = F.glu(self.fc1(x_))
        z21 = F.relu(self.fc21(z1))
        z22 = F.relu(self.fc22(z1)) #consider using a nn.threshold or hardshrink after this
        z2 = torch.cat((z21,z22), dim=1)
        out = self.fc31(z2)+self.fc32(z1)
        
        mu, sigma_adjust = out[:,0], out[:,1] 
        
        sigma_perf = np.exp(3.5*eps-4.) #between 0.03, 0.6
        sigma_adjust = 0.5*torch.tanh(sigma_adjust) #between -0.5, 0.5
        sig_ = sigma_perf +sigma_adjust
        
#         if np.random.rand()<0.0001:
#             print('sigma_adjust , sigma_perf= ', sigma_adjust.item(), sigma_perf)
        sig = torch.clamp(sig_, self.std_min, self.std_max)
        
        dist = distributions.normal.Normal(mu, sig)
#         if out.dim()>1:
#             print('mu, sigma', mu, sig)
        return dist


class NetLGlLMu(nn.Module):
# Inputs:
# r, r', x, x', f_n, f_n', f1
# Outputs:
# Q
    def __init__(self, nin, n_hidden=128, std_min=0.03, std_max=2., masked_ftrs=[], ftr_normalizer=None):
        super().__init__()
        
        self.std_min = std_min
        self.std_max = std_max
        
        self.masked_ftrs = masked_ftrs
        self.ftr_normalizer = ftr_normalizer
        if self.ftr_normalizer is not None:
            self.ftr_normalizer = np.array(ftr_normalizer, dtype=np.float32)
        
        if len(masked_ftrs)>0:
            mask = torch.ones(nin)
            mask[masked_ftrs]=0
            self.mask = mask.ge(0.5) 
            nin = nin-len(masked_ftrs)
            
        #The linear component
        self.c1fc1 = nn.Linear(nin, 2, bias=False)
#         torch.nn.init.normal_(self.fc1.weight, mean=0.0, std=0.03)
        torch.nn.init.normal_(self.c1fc1.weight, mean=0.0, std=0.1)
    
        # The nonlinear component w/ GLU
        self.c2fc1 = nn.Linear(nin, n_hidden)
        self.c2fc2 = nn.Linear(int(n_hidden/2), 2, bias=False)
        
        torch.nn.init.normal_(self.c2fc1.weight, mean=0.0, std=5.)
        torch.nn.init.normal_(self.c2fc2.weight, mean=0.0, std=0.01)
        
        
    def forward(self, x, eps=0):
        # eps indicates the amount of exploration which will
        # affect the variance of actions generated.
        if len(self.masked_ftrs)>0:
            x_ = x[:,self.mask] #torch.masked_select(x, self.mask)
        else:
            x_ = x
        if self.ftr_normalizer is not None:
            x_ = x_*self.ftr_normalizer
            
        y1 = self.c1fc1(x_)
        
        z1 = F.glu(self.c2fc1(x_))
        y2 = self.c2fc2(z1)
        out =  y1+y2
        
        
        mu, sigma_adjust = out[:,0], out[:,1] 
        
        sigma_perf = np.exp(3.5*eps-4.) #between 0.03, 0.6
        sigma_adjust = 0.5*torch.tanh(sigma_adjust) #between -0.5, 0.5
        sig_ = sigma_perf +sigma_adjust
        
#         if np.random.rand()<0.0001:
#             print('sigma_adjust , sigma_perf= ', sigma_adjust.item(), sigma_perf)
        sig = torch.clamp(sig_, self.std_min, self.std_max)
        
        dist = distributions.normal.Normal(mu, sig)
#         if out.dim()>1:
#             print('mu, sigma', mu, sig)
        return dist

class NetTanhReLG(nn.Module):
# Inputs:
# r, r', x, x', f_n, f_n', f1
# Outputs:
# Q
    def __init__(self, nin, n_hidden=128, log_std_min=-20, log_std_max=0., masked_ftrs=[], ftr_normalizer=None):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.masked_ftrs = masked_ftrs
        self.ftr_normalizer = ftr_normalizer
        if self.ftr_normalizer is not None:
            self.ftr_normalizer = np.array(ftr_normalizer, dtype=np.float32)
        
        if len(masked_ftrs)>0:
            mask = torch.ones(nin)
            mask[masked_ftrs]=0
            self.mask = mask.ge(0.5) 
            nin = nin-len(masked_ftrs)
        self.fc1 = nn.Linear(nin, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, 2, bias=False)
        
        torch.nn.init.normal_(self.fc1.weight, mean=0.0, std=.5)
        torch.nn.init.normal_(self.fc2.weight, mean=0.0, std=2.)
        torch.nn.init.normal_(self.fc3.weight, mean=0.0, std=0.01)
        
        torch.nn.init.normal_(self.fc1.bias, mean=0.0, std=0.05)
        torch.nn.init.normal_(self.fc2.bias, mean=0.0, std=0.2)
        
    def forward(self, x):
        if len(self.masked_ftrs)>0:
            x_ = x[:,self.mask] #torch.masked_select(x, self.mask)
        else:
            x_ = x
        if self.ftr_normalizer is not None:
            x_ = x_*self.ftr_normalizer
        z0 = self.fc1(x_)
        z0 = z0 +10*(torch.exp(1.+self.fc1.bias) -np.e )
        z1 = torch.tanh(z0)
        
#         z2 = F.relu(self.fc2(z1))
        out = self.fc3(z1)
        
        mu, log_std = out[:,0], out[:,1]

#         sig = nn.Softplus()(sig_)
        log_std = log_std -1.
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        sig = log_std.exp()
        
        dist = distributions.normal.Normal(mu, sig)
#         if out.dim()>1:
#             print('mu, sigma', mu, sig)
        return dist


class NetGlReLG(nn.Module):
# Inputs:
# r, r', x, x', f_n, f_n', f1
# Outputs:
# Q
    def __init__(self, nin, n_hidden=128, log_std_min=-20, log_std_max=0., masked_ftrs=[]):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.masked_ftrs = masked_ftrs
        if len(masked_ftrs)>0:
            mask = torch.ones(nin)
            mask[masked_ftrs]=0
            self.mask = mask.ge(0.5) 
            nin = nin-len(masked_ftrs)
        self.fc1 = nn.Linear(nin, n_hidden)
        self.fc2 = nn.Linear(int(n_hidden*1.5), 2, bias=False)
        
        torch.nn.init.normal_(self.fc1.weight, mean=0.0, std=5.)
        torch.nn.init.normal_(self.fc2.weight, mean=0.0, std=0.01)
        
    def forward(self, x):
        if len(self.masked_ftrs)>0:
            x_ = x[:,self.mask] #torch.masked_select(x, self.mask)
        else:
            x_ = x
        z11 = F.glu(self.fc1(x_))
        z12 = F.relu(self.fc1(x_))
        z1 = torch.cat((z11,z12), dim=1)
        out = self.fc2(z1)
        
        mu, log_std = out[:,0], out[:,1]

#         sig = nn.Softplus()(sig_)
        log_std = log_std -1.
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        sig = log_std.exp()
        
        dist = distributions.normal.Normal(mu, sig)
#         if out.dim()>1:
#             print('mu, sigma', mu, sig)
        return dist



class NetGru1L1G(nn.Module):
# Requires a batch for hidden states too.
    def __init__(self, nin, log_std_min=-20, log_std_max=0.):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        hidden_size = 2*nin
        
        self.gru1 = nn.GRU(input_size=nin, hidden_size=hidden_size, num_layers=1)
        self.fc1 = nn.Linear(hidden_size+nin, 2, bias=False)
#         torch.nn.init.normal_(self.fc1.weight, mean=0.0, std=0.03)
        torch.nn.init.normal_(self.fc1.weight, mean=0.0, std=0.1)
        self.gru1_hn = torch.zeros(1,1,hidden_size)
        
        
        
    def forward(self, x):
        self.gru1_hn = self.gru1(x, self.gru1_hn)
        

        features = torch.cat((x, self.gru1_hn), 1)
        out = self.fc1(features)
        mu, log_std = out[:,0], out[:,1]

#         sig = nn.Softplus()(sig_)
        
        log_std = log_std -1. #@@@@@@ to improve the range
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        sig = log_std.exp()
        
        dist = distributions.normal.Normal(mu, sig)
#         if out.dim()>1:
#             print('mu, sigma', mu, sig)
        return dist


### New architecture
class NetDiscPIDG(nn.Module):
# Inputs:
# r, r', x, x', f_n, f_n', f1
# Outputs:
# Q
    def __init__(self, nin, n_hidden=128, n_hl=1, 
                 std_min=0.03, std_max=2., 
                 masked_ftrs=[], ftr_normalizer=None,
                ctrl_ftrs=slice(3,6)):
        super().__init__()
        
        self.ctrl_ftr_i = ctrl_ftrs
        self.n_hl = n_hl
        self.len_ctrftr = ctrl_ftrs.stop - ctrl_ftrs.start
        self.std_min = std_min
        self.std_max = std_max
        
        self.masked_ftrs = masked_ftrs
        self.ftr_normalizer = ftr_normalizer
        if self.ftr_normalizer is not None:
            self.ftr_normalizer = np.array(ftr_normalizer, dtype=np.float32)
        
        if len(masked_ftrs)>0:
            mask = torch.ones(nin)
            mask[masked_ftrs]=0
            self.mask = mask.ge(0.5) 
            nin = nin-len(masked_ftrs)
        self.fc1 = nn.Linear(nin, n_hidden)
        self.fc2 = nn.Linear(int(n_hidden/2), n_hl, bias=False)
        
        torch.nn.init.normal_(self.fc1.weight, mean=0.0, std=5.)
        torch.nn.init.normal_(self.fc2.weight, mean=0.0, std=1.)
        
#         self.fc_cont = nn.Linear(self.len_ctrftr, 1, bias=True)
        self.pid = nn.Parameter(
            torch.tensor([3./n_hl, 0.3/n_hl, 0.03/n_hl]) )
        self.w_const = nn.Parameter(
            torch.tensor([0.1]) )
        
        self.fc_std = nn.Linear(n_hl, 1, bias=False)
        
        torch.nn.init.normal_(self.fc_std.weight, mean=0.0, std=1.)
                                
        
    def forward(self, x, eps=0, verbose=False):
        # eps indicates the amount of exploration which will
        # affect the variance of actions generated.
        if len(self.masked_ftrs)>0:
            x_ = x[:,self.mask] #torch.masked_select(x, self.mask)
        else:
            x_ = x
        if self.ftr_normalizer is not None:
            x_ = x_*self.ftr_normalizer
        
        err_ftrs = x_[:, self.ctrl_ftr_i]
        
        z1 = F.glu(self.fc1(x_))
        z2 = torch.sigmoid(self.fc2(z1)) #@@@tanh
        z3 = torch.sum(z2, dim=1).view(-1,1)
        
        
        effective_weights = torch.mm(z3, self.pid.view(1,-1))
#         effective_weights = torch.exp(effective_weights) #@@@@@@
        effective_bias = z3*self.w_const
        mu = torch.sum(effective_weights * err_ftrs, axis=1).view(-1,1) + effective_bias
        
#         mu = torch.dot(torch.exp(torch.dot(z3, self.pid), err_ftrs) ) #+z3*self.w_const
        
        sigma_adjust = self.fc_std(z2) 
        
        sigma_perf = np.exp(3.5*eps-4.) #between 0.03, 0.6
        sigma_adjust = 0.5*torch.tanh(sigma_adjust) #between -0.5, 0.5
        sig_ = sigma_perf +sigma_adjust
        
        if np.random.rand()<0.0001:
#             print('z3=', z3)
#             print('z3 = ', z3.detach.numpy())
            print('sigma_adjust , sigma_perf= ', sigma_adjust.detach().numpy(), sigma_perf)
        sig = torch.clamp(sig_, self.std_min, self.std_max)
        
        dist = distributions.normal.Normal(mu, sig)
        
        if verbose:
            return dist, z3
        else:
            return dist
        
        
class NetDiscPIDG_1(nn.Module):
# Inputs:
# r, r', x, x', f_n, f_n', f1
# Outputs:
# Q
    def __init__(self, nin, n_hidden=128, n_hl=1, 
                 std_min=0.03, std_max=2., 
                 masked_ftrs=[], ftr_normalizer=None,
                ctrl_ftrs=slice(3,6)):
        super().__init__()
        
        self.ctrl_ftr_i = ctrl_ftrs
        self.n_hl = n_hl
        self.len_ctrftr = ctrl_ftrs.stop - ctrl_ftrs.start
        self.std_min = std_min
        self.std_max = std_max
        
        self.masked_ftrs = masked_ftrs
        self.ftr_normalizer = ftr_normalizer
        if self.ftr_normalizer is not None:
            self.ftr_normalizer = np.array(ftr_normalizer, dtype=np.float32)
        
        if len(masked_ftrs)>0:
            mask = torch.ones(nin)
            mask[masked_ftrs]=0
            self.mask = mask.ge(0.5) 
            nin = nin-len(masked_ftrs)
        self.fc1 = nn.Linear(nin, n_hidden)
        self.fc2 = nn.Linear(int(n_hidden/2), n_hl, bias=False)
        
        torch.nn.init.normal_(self.fc1.weight, mean=0.0, std=5.)
        torch.nn.init.normal_(self.fc2.weight, mean=0.0, std=1.)
        
#         self.fc_cont = nn.Linear(self.len_ctrftr, 1, bias=True)
        self.pid = nn.Parameter(
            torch.tensor([3./n_hl, 0.3/n_hl, 0.03/n_hl]) )
        self.w_const = nn.Parameter(
            torch.tensor([0.1]) )
        
        self.fc_std = nn.Linear(n_hl, 1, bias=False)
        
        torch.nn.init.normal_(self.fc_std.weight, mean=0.0, std=1.)
                                
        
    def forward(self, x, eps=0, verbose=False):
        # eps indicates the amount of exploration which will
        # affect the variance of actions generated.
        if len(self.masked_ftrs)>0:
            x_ = x[:,self.mask] #torch.masked_select(x, self.mask)
        else:
            x_ = x
        if self.ftr_normalizer is not None:
            x_ = x_*self.ftr_normalizer
        
        err_ftrs = x_[:, self.ctrl_ftr_i]
        
        z1 = F.glu(self.fc1(x_))
        z2 = torch.sigmoid(self.fc2(z1)) #@@@tanh
        z3 = torch.sum(z2, dim=1).view(-1,1)
        
        sigma_adjust = self.fc_std(z2) 
#         sigma_perf = np.exp(3.5*eps-4.) #between 0.03, 0.6
        sigma_adjust = -3.+2.*torch.tanh(sigma_adjust) #between -5, 1-
        sig_ = torch.exp(sigma_adjust)#sigma_perf +sigma_adjust
        
        sig = torch.clamp(sig_, self.std_min, self.std_max)
        
        if eps>0:
            zdist = distributions.normal.Normal(z3, sig)
            zl = zdist.rsample()
        else:
            zl = z3
        
        
        effective_weights = torch.mm(zl, self.pid.view(1,-1))
#         effective_weights = torch.exp(effective_weights) #@@@@@@
        effective_bias = zl*self.w_const
        mu = torch.sum(effective_weights * err_ftrs, axis=1).view(-1,1) + effective_bias
        
#         mu = torch.dot(torch.exp(torch.dot(z3, self.pid), err_ftrs) ) #+z3*self.w_const
        
        if np.random.rand()<0.0001:
#             print('z3=', z3)
#             print('z3 = ', z3.detach.numpy())
            print('sigma', sig.detach().numpy())
        
        
        dist = distributions.normal.Normal(mu, 0.03)
        
        if verbose:
            return dist, z3
        else:
            return dist
        
        
class NetContPIDZG(nn.Module):
# Inputs:
# r, r', x, x', f_n, f_n', f1
# Outputs:
# Q
    def __init__(self, nin, n_hidden=128, n_hl=1, 
                 std_min=0.03, std_max=2., 
                 masked_ftrs=[], ftr_normalizer=None,
                ctrl_ftrs=slice(3,6)):
        super().__init__()
        
        self.ctrl_ftr_i = ctrl_ftrs
        self.n_hl = n_hl
        self.len_ctrftr = ctrl_ftrs.stop - ctrl_ftrs.start
        self.std_min = std_min
        self.std_max = std_max
        
        self.masked_ftrs = masked_ftrs
        self.ftr_normalizer = ftr_normalizer
        if self.ftr_normalizer is not None:
            self.ftr_normalizer = np.array(ftr_normalizer, dtype=np.float32)
        
        if len(masked_ftrs)>0:
            mask = torch.ones(nin)
            mask[masked_ftrs]=0
            self.mask = mask.ge(0.5) 
            nin = nin-len(masked_ftrs)
        self.fc1 = nn.Linear(nin, n_hidden)
        self.fc2 = nn.Linear(int(n_hidden/2), n_hl, bias=False)
        
        torch.nn.init.normal_(self.fc1.weight, mean=0.0, std=5.)
        torch.nn.init.normal_(self.fc2.weight, mean=0.0, std=1.)
        
#         self.fc_cont = nn.Linear(self.len_ctrftr, 1, bias=True)
        self.pid_fc = nn.Linear(int(n_hidden/2), 3, bias=True)
        torch.nn.init.normal_(self.pid_fc.weight, mean=0.0, std=1.)
        torch.nn.init.constant_(self.pid_fc.bias, 0.5)
        self.pid_scalers = torch.tensor([3./n_hl, 0.3/n_hl, 0.03/n_hl])
        
        self.fc_std = nn.Linear(n_hl, 1, bias=False)
        torch.nn.init.normal_(self.fc_std.weight, mean=0.0, std=1.)
                    
        
    def forward(self, x, eps=0, verbose=False):
        # eps indicates the amount of exploration which will
        # affect the variance of actions generated.
        if len(self.masked_ftrs)>0:
            x_ = x[:,self.mask] #torch.masked_select(x, self.mask)
        else:
            x_ = x
        if self.ftr_normalizer is not None:
            x_ = x_*self.ftr_normalizer
        
        err_ftrs = x_[:, self.ctrl_ftr_i]
        
        z1 = F.glu(self.fc1(x_))
        z2 = torch.sigmoid(self.fc2(z1)) #@@@tanh
        z3 = torch.sum(z2, dim=1).view(-1,1)
        
        pid_w = self.pid_scalers* torch.relu(self.pid_fc(z1))#sigmoid(self.pid_fc(z1))
        
        sigma_adjust = self.fc_std(z2) 
#         sigma_perf = np.exp(3.5*eps-4.) #between 0.03, 0.6
#         sigma_adjust = -3.+2.*torch.tanh(sigma_adjust) #between -5, 1-
#         sig_ = torch.exp(sigma_adjust)#+sigma_perf +sigma_adjust
        sigma_perf = np.exp(3.5*eps-4.) #between 0.03, 0.6
        sigma_adjust = 0.5*torch.tanh(sigma_adjust) #between -0.5, 0.5
        sig_ = sigma_perf +sigma_adjust
        
        sig = torch.clamp(sig_, self.std_min, self.std_max)
        
        if eps>0:
            zdist = distributions.normal.Normal(z3, sig)
            zl = zdist.rsample()
        else:
            zl = z3
        
        
        effective_weights = zl*pid_w
        mu = torch.sum(effective_weights * err_ftrs, axis=1).view(-1,1)
        
        if np.random.rand()<0.0001:
#             print('z3=', z3)
#             print('z3 = ', z3.detach.numpy())
            print('sigma', sig.detach().numpy())
        
        
        dist = distributions.normal.Normal(mu, 0.03)
        
        if verbose:
            return dist, z3.detach().numpy(), pid_w.detach().numpy()
        else:
            return dist
        
class NetContPDG_depr0(nn.Module):

    def __init__(self, nin, nout, n_hidden=128, 
                 std_min=0.03, std_max=2., 
                 masked_ftrs=[], ftr_normalizer=None):
        super().__init__()
        
        self.nout = nout
        self.std_min = std_min
        self.std_max = std_max
        
        self.masked_ftrs = masked_ftrs
        self.ftr_normalizer = ftr_normalizer
        if self.ftr_normalizer is not None:
            self.ftr_normalizer = np.array(ftr_normalizer, dtype=np.float32)
        
        if len(masked_ftrs)>0:
            mask = torch.ones(nin)
            mask[masked_ftrs]=0
            self.mask = mask.ge(0.5) 
            nin = nin-len(masked_ftrs)
        self.fc1 = nn.Linear(nin, n_hidden)
        torch.nn.init.normal_(self.fc1.weight, mean=0.0, std=1.)
        
        self.fc_mu = nn.Linear(int(n_hidden/2), nout, bias=False)
        torch.nn.init.normal_(self.fc_mu.weight, mean=0.0, std=4./8)
        
        self.fc_std = nn.Linear(int(n_hidden/2), nout, bias=False)
        torch.nn.init.normal_(self.fc_std.weight, mean=0.0, std=.1)
                    
        
    def forward(self, x, eps=0, verbose=False):
        # eps indicates the amount of exploration which will
        # affect the variance of actions generated.
        if len(self.masked_ftrs)>0:
            x_ = x[:,self.mask] #torch.masked_select(x, self.mask)
        else:
            x_ = x
        if self.ftr_normalizer is not None:
            x_ = x_*self.ftr_normalizer
        
        z1 = torch.tanh(F.glu(self.fc1(x_)))
#         z1 = F.glu(self.fc1(x_))
        
        mu = torch.relu(self.fc_mu(z1))#sigmoid(self.pid_fc(z1))
        
        sig_ = torch.tanh(self.fc_std(z1)) #[-1,1]
#         sig = torch.exp(0.8*sig_-3.1)# map to [-3.9, -2.3], then => [0.02, 0.1]
        sig = torch.exp(0.6*sig_-2.9)# map to [-3.5, -2.3], then => [0.02, 0.1]

        dist = MultivariateNormal(mu, scale_tril=torch.diag_embed(sig))
    
        if np.random.rand()<0.0001 and x.shape[0]==1:
            print('sigma', sig.detach().numpy())
        
        return dist
    
class NetContPDG_depr1(nn.Module):

    def __init__(self, nin, nout, n_hidden=128, n_hl=5,
                 std_min=0.03, std_max=.5, 
                 masked_ftrs=[], ftr_normalizer=None,
                device=None):
        super().__init__()
        
        if device is None:
            device = torch.device("cpu")
            
        self.nout = nout
        self.std_min = std_min
        self.std_max = std_max
        
        self.masked_ftrs = masked_ftrs
        self.ftr_normalizer = ftr_normalizer
        if self.ftr_normalizer is not None:
            self.ftr_normalizer = torch.tensor(ftr_normalizer, dtype=torch.float32,
                                              device=device)
        
        if len(masked_ftrs)>0:
            mask = torch.ones(nin, device=device)
            mask[masked_ftrs]=0
            self.mask = mask.ge(0.5) 
            nin = nin-len(masked_ftrs)
        self.fc1 = nn.Linear(nin, n_hidden)
        torch.nn.init.normal_(self.fc1.weight, mean=0.0, std=5.)
        
        self.fc2 = nn.Linear(int(n_hidden/2), n_hl, bias=False)
        torch.nn.init.normal_(self.fc2.weight, mean=0.0, std=0.1)
        
        self.fc_mu = nn.Linear(n_hl, nout, bias=False)
        torch.nn.init.normal_(self.fc_mu.weight, mean=0., std=.1/np.sqrt(n_hl))
        
#         self.fc_std = nn.Linear(n_hl, nout, bias=False)
#         torch.nn.init.normal_(self.fc_std.weight, mean=0.0, std=1.)
        self.std = torch.full((1,nout), 0.1, device=device)
                    
        
    def forward(self, x, eps=0, verbose=False):
        # eps indicates the amount of exploration which will
        # affect the variance of actions generated.
        if len(self.masked_ftrs)>0:
            x_ = torch.masked_select(x, self.mask) #x[:,self.mask] #
        else:
            x_ = x
        if self.ftr_normalizer is not None:
            x_ = x_*self.ftr_normalizer
        
        z1 = F.glu(self.fc1(x_))
        z2 = torch.sigmoid(self.fc2(z1))
        
        mu = torch.relu(self.fc_mu(z2))#sigmoid(self.pid_fc(z1))
        
#         sig_ = torch.tanh(self.fc_std(z2)) #[-1,1]
#         sig = torch.exp(0.6*sig_-2.9)# map to [-3.5, -2.3], then => [0.02, 0.1]
        sig = self.std
        dist = MultivariateNormal(mu, scale_tril=torch.diag_embed(sig))
    
#         if np.random.rand()<0.0001 and x.shape[0]==1:
#             print('sigma', sig.detach().numpy())
        
        if verbose:
            return mu.detach().numpy().squeeze(), (z1.detach().numpy().squeeze(), z2.detach().numpy().squeeze())
        return dist
    
    
class NetContPDG(nn.Module):

    def __init__(self, nin, nout, n_hidden=128, n_hl=5,
                 std_min=0.03, std_max=.5, 
                 masked_ftrs=[], ftr_normalizer=None,
                device=None):
        super().__init__()
        
        if device is None:
            device = torch.device("cpu")
            
        self.nout = nout
        self.std_min = std_min
        self.std_max = std_max
        
        self.masked_ftrs = masked_ftrs
        self.ftr_normalizer = ftr_normalizer
        if self.ftr_normalizer is not None:
            self.ftr_normalizer = torch.tensor(ftr_normalizer, dtype=torch.float32,
                                              device=device)
        
        if len(masked_ftrs)>0:
            mask = torch.ones(nin, device=device)
            mask[masked_ftrs]=0
            self.mask = mask.ge(0.5) 
            nin = nin-len(masked_ftrs)
        self.fc1 = nn.Linear(nin*2, n_hidden)
        torch.nn.init.normal_(self.fc1.weight, mean=0.0, std=5.)
        
        self.fc2 = nn.Linear(int(n_hidden/2), n_hl, bias=False)
        torch.nn.init.normal_(self.fc2.weight, mean=0.0, std=0.1)
        
        self.fc_mu = nn.Linear(n_hl, nout, bias=False)
        torch.nn.init.normal_(self.fc_mu.weight, mean=0., std=.1/np.sqrt(n_hl))
        
#         self.fc_std = nn.Linear(n_hl, nout, bias=False)
#         torch.nn.init.normal_(self.fc_std.weight, mean=0.0, std=1.)
        self.std = torch.full((1,nout), 0.1, device=device)
                    
        
    def forward(self, x, eps=0, verbose=False):
        # eps indicates the amount of exploration which will
        # affect the variance of actions generated.
        if len(self.masked_ftrs)>0:
            x_ = torch.masked_select(x, self.mask) #x[:,self.mask] #
        else:
            x_ = x
        if self.ftr_normalizer is not None:
            x_ = x_*self.ftr_normalizer
        
        x_symm = torch.cat((x_,-x_), dim=1)
        z1 = F.glu(self.fc1(x_symm))
        z2 = torch.sigmoid(self.fc2(z1))
        
        mu = torch.relu(self.fc_mu(z2))#sigmoid(self.pid_fc(z1))
        
#         sig_ = torch.tanh(self.fc_std(z2)) #[-1,1]
#         sig = torch.exp(0.6*sig_-2.9)# map to [-3.5, -2.3], then => [0.02, 0.1]
        sig = self.std
        dist = MultivariateNormal(mu, scale_tril=torch.diag_embed(sig))
    
#         if np.random.rand()<0.0001 and x.shape[0]==1:
#             print('sigma', sig.detach().numpy())
        
        if verbose:
            return mu.detach().numpy().squeeze(), (z1.detach().numpy().squeeze(), z2.detach().numpy().squeeze())
        return dist
    
    
class NetContPDG2(nn.Module):

    def __init__(self, nin, nout, n_hidden=128, n_hl=5,
                 std_min=0.03, std_max=.5, 
                 masked_ftrs=[], ftr_normalizer=None,
                device=None):
        super().__init__()
        
        if device is None:
            device = torch.device("cpu")
            
        self.nout = nout
        self.std_min = std_min
        self.std_max = std_max
        
        self.masked_ftrs = masked_ftrs
        self.ftr_normalizer = ftr_normalizer
        if self.ftr_normalizer is not None:
            self.ftr_normalizer = torch.tensor(ftr_normalizer, dtype=torch.float32,
                                              device=device)
        
        if len(masked_ftrs)>0:
            mask = torch.ones(nin, device=device)
            mask[masked_ftrs]=0
            self.mask = mask.ge(0.5) 
            nin = nin-len(masked_ftrs)
        self.fc1 = nn.Linear(nin*2, n_hidden)
        torch.nn.init.normal_(self.fc1.weight, mean=0.0, std=0.1)
        torch.nn.init.normal_(self.fc1.bias, mean=0.0, std=0.1)
        
        self.fc2 = nn.Linear(int(n_hidden/2), n_hl, bias=False)
        torch.nn.init.normal_(self.fc2.weight, mean=0.0, std=0.1)
        
        self.fc_mu = nn.Linear(n_hl, nout, bias=False)
        torch.nn.init.normal_(self.fc_mu.weight, mean=0., std=.1/np.sqrt(n_hl))
        
#         self.fc_std = nn.Linear(n_hl, nout, bias=False)
#         torch.nn.init.normal_(self.fc_std.weight, mean=0.0, std=1.)
        self.stdmax = torch.full((1,nout), 0.5, device=device)
                    
        
    def forward(self, x, eps=0, verbose=False):
        # eps indicates the amount of exploration which will
        # affect the variance of actions generated.
        if len(self.masked_ftrs)>0:
            x_ = torch.masked_select(x, self.mask) #x[:,self.mask] #
        else:
            x_ = x
        if self.ftr_normalizer is not None:
            x_ = x_*self.ftr_normalizer
        
        x_symm = torch.cat((x_,-x_), dim=1)
        z1 = F.glu(self.fc1(x_symm))
        z2 = torch.tanh(self.fc2(z1)) #@@@
        
#         mu = torch.relu(self.fc_mu(z2))#sigmoid(self.pid_fc(z1))
        mu = torch.exp(self.fc_mu(z2)-1.) #@@@
    
#         sig_ = torch.tanh(self.fc_std(z2)) #[-1,1]
#         sig = torch.exp(0.6*sig_-2.9)# map to [-3.5, -2.3], then => [0.02, 0.1]
        sig = self.stdmax *(mu.detach())
        dist = MultivariateNormal(mu, scale_tril=torch.diag_embed(sig))
    
#         if np.random.rand()<0.0001 and x.shape[0]==1:
#             print('sigma', sig.detach().numpy())
        
        if verbose:
            return mu.detach().numpy().squeeze(), (z1.detach().numpy().squeeze(), z2.detach().numpy().squeeze())
        return dist
    
    
class NetContPDG3(nn.Module):

    def __init__(self, nin, nout, n_hidden=128, n_hl=5,
                 std_min=0.03, std_max=.5, 
                 masked_ftrs=[], ftr_normalizer=None,
                device=None):
        super().__init__()
        
        if device is None:
            device = torch.device("cpu")
            
        self.nout = nout
        self.std_min = std_min
        self.std_max = std_max
        
        self.masked_ftrs = masked_ftrs
        self.ftr_normalizer = ftr_normalizer
        if self.ftr_normalizer is not None:
            self.ftr_normalizer = torch.tensor(ftr_normalizer, dtype=torch.float32,
                                              device=device)
        
        if len(masked_ftrs)>0:
            mask = torch.ones(nin, device=device)
            mask[masked_ftrs]=0
            self.mask = mask.ge(0.5) 
            nin = nin-len(masked_ftrs)
        self.fc1 = nn.Linear(nin*2, n_hidden, bias=False)
        torch.nn.init.sparse_(self.fc1.weight, sparsity=1./nin, std=0.1)
        # relu
        
        self.fc15 = nn.Linear(n_hidden, int(n_hidden/2), bias=True)
        torch.nn.init.sparse_(self.fc15.weight, sparsity=0.1, std=0.1)
        torch.nn.init.normal_(self.fc15.bias, mean=0.0, std=0.01)
        
        # relu
        
        self.fc2 = nn.Linear(int(n_hidden/2), n_hl, bias=False)
        torch.nn.init.sparse_(self.fc2.weight, sparsity=0.1, std=0.1)
        
        self.fc_mu = nn.Linear(n_hl, nout, bias=False)
        torch.nn.init.normal_(self.fc_mu.weight, mean=0., std=.1/np.sqrt(n_hl))
        
#         self.fc_std = nn.Linear(n_hl, nout, bias=False)
#         torch.nn.init.normal_(self.fc_std.weight, mean=0.0, std=1.)
        self.stdmax = torch.full((1,nout), 0.5, device=device)
                    
        
    def forward(self, x, eps=0, verbose=False):
        # eps indicates the amount of exploration which will
        # affect the variance of actions generated.
        if len(self.masked_ftrs)>0:
            x_ = torch.masked_select(x, self.mask) #x[:,self.mask] #
        else:
            x_ = x
        if self.ftr_normalizer is not None:
            x_ = x_*self.ftr_normalizer
        
        x_symm = torch.cat((x_,-x_), dim=1)
        z1 = F.leaky_relu(self.fc1(x_symm))
        z15 = F.leaky_relu(self.fc15(z1))
        z2 = torch.tanh(self.fc2(z15)) #@@@
        
#         mu = torch.relu(self.fc_mu(z2))#sigmoid(self.pid_fc(z1))
        mu = torch.exp(self.fc_mu(z2)-1.) #@@@
    
#         sig_ = torch.tanh(self.fc_std(z2)) #[-1,1]
#         sig = torch.exp(0.6*sig_-2.9)# map to [-3.5, -2.3], then => [0.02, 0.1]
        sig = self.stdmax *(mu.detach())
        dist = MultivariateNormal(mu, scale_tril=torch.diag_embed(sig))
    
#         if np.random.rand()<0.0001 and x.shape[0]==1:
#             print('sigma', sig.detach().numpy())
        
        if verbose:
            return mu.detach().numpy().squeeze(), (z1.detach().numpy().squeeze(), z2.detach().numpy().squeeze())
        return dist
    
    
class NetContPDG4(nn.Module):

    def __init__(self, nin, nout, n_hidden=128, n_hl=5,
                 std_min=0.03, std_max=.5, 
                 masked_ftrs=[], ftr_normalizer=None,
                device=None):
        super().__init__()
        
        if device is None:
            device = torch.device("cpu")
            
        self.nout = nout
        self.std_min = std_min
        self.std_max = std_max
        
        self.masked_ftrs = masked_ftrs
        self.ftr_normalizer = ftr_normalizer
        if self.ftr_normalizer is not None:
            self.ftr_normalizer = torch.tensor(ftr_normalizer, dtype=torch.float32,
                                              device=device)
        
        if len(masked_ftrs)>0:
            mask = torch.ones(nin, device=device)
            mask[masked_ftrs]=0
            self.mask = mask.ge(0.5) 
            nin = nin-len(masked_ftrs)
        self.fc1 = nn.Linear(nin*2, n_hidden, bias=False)
        torch.nn.init.sparse_(self.fc1.weight, sparsity=1-2./nin, std=0.5)
        # relu
        
#         self.fc15 = nn.Linear(n_hidden, int(n_hidden/2), bias=True)
#         torch.nn.init.sparse_(self.fc15.weight, sparsity=0.1, std=0.1)
#         torch.nn.init.normal_(self.fc15.bias, mean=0.0, std=0.01)
        
        # relu
        
        self.fc2 = nn.Linear(n_hidden, n_hl, bias=False)
#         torch.nn.init.normal_(self.fc2.weight, mean=1./n_hidden, std=1e-3)
        torch.nn.init.sparse_(self.fc2.weight, sparsity=1-1./(n_hl-2), std=0.25)#n_hl/n_hidden)#
        
        self.fc_mu = nn.Linear(n_hl, nout, bias=False)
        torch.nn.init.normal_(self.fc_mu.weight, mean=0.5/n_hl, std=1e-3)
#         torch.nn.init.constant_(self.fc_mu.weight, 0.1/n_hl)
        
#         self.fc_std = nn.Linear(n_hl, nout, bias=False)
#         torch.nn.init.normal_(self.fc_std.weight, mean=0.0, std=1.)
        self.stdmax = torch.full((1,nout), 0.1, device=device)
                    
        
    def forward(self, x, eps=0, verbose=False):
        # eps indicates the amount of exploration which will
        # affect the variance of actions generated.
        if len(self.masked_ftrs)>0:
            x_ = torch.masked_select(x, self.mask) #x[:,self.mask] #
        else:
            x_ = x
        if self.ftr_normalizer is not None:
            x_ = x_*self.ftr_normalizer
        
        x_symm = torch.cat((x_,-x_), dim=1)
        z1 = F.leaky_relu(self.fc1(x_symm))
#         z15 = F.leaky_relu(self.fc15(z1))
        z2 = torch.sigmoid(self.fc2(z1)) #@@@
        
        mu = torch.relu(self.fc_mu(z2))#sigmoid(self.pid_fc(z1))
#         mu = torch.exp(self.fc_mu(z2)-1.) #@@@
    
#         sig_ = torch.tanh(self.fc_std(z2)) #[-1,1]
#         sig = torch.exp(0.6*sig_-2.9)# map to [-3.5, -2.3], then => [0.02, 0.1]
        sig = self.stdmax #*(mu.detach())
        dist = MultivariateNormal(mu, scale_tril=torch.diag_embed(sig))
    
#         if np.random.rand()<0.0001 and x.shape[0]==1:
#             print('sigma', sig.detach().numpy())
        
        if verbose:
            return mu.detach().numpy().squeeze(), (z1.detach().numpy().squeeze(), z2.detach().numpy().squeeze())
        return dist
    
    
class NetContPDG5(nn.Module):

    def __init__(self, nin, nout, n_hidden=128, n_hl=5,
                 std_min=0.03, std_max=.5, 
                 masked_ftrs=[], ftr_normalizer=None,
                device=None):
        super().__init__()
        
        if device is None:
            device = torch.device("cpu")
            
        self.nout = nout
        self.std_min = std_min
        self.std_max = std_max
        
        self.masked_ftrs = masked_ftrs
        self.ftr_normalizer = ftr_normalizer
        if self.ftr_normalizer is not None:
            self.ftr_normalizer = torch.tensor(ftr_normalizer, dtype=torch.float32,
                                              device=device)
        
        if len(masked_ftrs)>0:
            mask = torch.ones(nin, device=device)
            mask[masked_ftrs]=0
            self.mask = mask.ge(0.5) 
            nin = nin-len(masked_ftrs)
        self.fc1 = nn.Linear(nin*2, n_hidden, bias=False)
        torch.nn.init.sparse_(self.fc1.weight, sparsity=1-2./nin, std=0.5)
        # relu
        
#         self.fc15 = nn.Linear(n_hidden, int(n_hidden/2), bias=True)
#         torch.nn.init.sparse_(self.fc15.weight, sparsity=0.1, std=0.1)
#         torch.nn.init.normal_(self.fc15.bias, mean=0.0, std=0.01)
        
        # relu
        
#         self.fc2 = nn.Linear(n_hidden, 20, bias=False)
# #         torch.nn.init.normal_(self.fc2.weight, mean=1./n_hidden, std=1e-3)
#         torch.nn.init.sparse_(self.fc2.weight, sparsity=1-1./(20-2), std=0.25)#n_hl/n_hidden)#
        
#         self.fc3 = nn.Linear(20, n_hl, bias=False)
# #         torch.nn.init.normal_(self.fc2.weight, mean=1./n_hidden, std=1e-3)
#         torch.nn.init.sparse_(self.fc3.weight, sparsity=1-1./(n_hl-2), std=0.25)#n_hl/n_hidden)#
    
        self.fc_mu = nn.Linear(n_hidden, nout, bias=False)
        torch.nn.init.normal_(self.fc_mu.weight, mean=0.5/n_hidden, std=1e-3)
#         torch.nn.init.constant_(self.fc_mu.weight, 0.1/n_hl)
        
#         self.fc_std = nn.Linear(n_hl, nout, bias=False)
#         torch.nn.init.normal_(self.fc_std.weight, mean=0.0, std=1.)
        self.stdmax = torch.full((1,nout), 0.1, device=device)
                    
        
    def forward(self, x, eps=0, verbose=False):
        # eps indicates the amount of exploration which will
        # affect the variance of actions generated.
        if len(self.masked_ftrs)>0:
            x_ = torch.masked_select(x, self.mask) #x[:,self.mask] #
        else:
            x_ = x
        if self.ftr_normalizer is not None:
            x_ = x_*self.ftr_normalizer
        
        x_symm = torch.cat((x_,-x_), dim=1)
        z1 = F.leaky_relu(self.fc1(x_symm))
#         z15 = F.leaky_relu(self.fc15(z1))
#         z2 = F.leaky_relu(self.fc2(z1))
#         z3 = torch.sigmoid(self.fc3(z2)) #@@@
        
        mu = torch.relu(self.fc_mu(z1))#sigmoid(self.pid_fc(z1))
#         mu = torch.exp(self.fc_mu(z2)-1.) #@@@
    
#         sig_ = torch.tanh(self.fc_std(z2)) #[-1,1]
#         sig = torch.exp(0.6*sig_-2.9)# map to [-3.5, -2.3], then => [0.02, 0.1]
        sig = self.stdmax #*(mu.detach())
        dist = MultivariateNormal(mu, scale_tril=torch.diag_embed(sig))
    
#         if np.random.rand()<0.0001 and x.shape[0]==1:
#             print('sigma', sig.detach().numpy())
        
        if verbose:
            return mu.detach().numpy().squeeze(), (z1.detach().numpy().squeeze(), z1.detach().numpy().squeeze())
        return dist
    
    
class NetContPDG6(nn.Module):

    def __init__(self, nin, nout, n_hidden=128, n_hl=5,
                 std_min=0.03, std_max=.5, 
                 masked_ftrs=[], ftr_normalizer=None,
                device=None):
        super().__init__()
        
        if device is None:
            device = torch.device("cpu")
            
        self.nout = nout
        self.std_min = std_min
        self.std_max = std_max
        
        self.masked_ftrs = masked_ftrs
        self.ftr_normalizer = ftr_normalizer
        if self.ftr_normalizer is not None:
            self.ftr_normalizer = torch.tensor(ftr_normalizer, dtype=torch.float32,
                                              device=device)
        
        if len(masked_ftrs)>0:
            mask = torch.ones(nin, device=device)
            mask[masked_ftrs]=0
            self.mask = mask.ge(0.5) 
            nin = nin-len(masked_ftrs)
        self.fc1 = nn.Linear(nin*2, n_hidden, bias=False)
        torch.nn.init.sparse_(self.fc1.weight, sparsity=1-2./nin, std=0.5)
        # 0.5+0.5tanh
    
        self.fc_mu = nn.Linear(n_hidden, nout, bias=False)
        torch.nn.init.normal_(self.fc_mu.weight, mean=0.5/n_hidden, std=1e-3)
#         torch.nn.init.constant_(self.fc_mu.weight, 0.1/n_hl)
        
#         self.fc_std = nn.Linear(n_hl, nout, bias=False)
#         torch.nn.init.normal_(self.fc_std.weight, mean=0.0, std=1.)
        self.stdmax = torch.full((1,nout), 0.1, device=device)
                    
        
    def forward(self, x, eps=0, verbose=False):
        # eps indicates the amount of exploration which will
        # affect the variance of actions generated.
        if len(self.masked_ftrs)>0:
            x_ = torch.masked_select(x, self.mask) #x[:,self.mask] #
        else:
            x_ = x
        if self.ftr_normalizer is not None:
            x_ = x_*self.ftr_normalizer
        
        x_symm = torch.cat((x_,-x_), dim=1)
        z1 = 0.5+0.5*torch.tanh(2.*self.fc1(x_symm))
#         z1 = torch.sigmoid(self.fc1(x_symm))
        mu = torch.relu(self.fc_mu(z1))
        
        sig = self.stdmax #*(mu.detach())
        dist = MultivariateNormal(mu, scale_tril=torch.diag_embed(sig))
    
        
        if verbose:
            return mu.detach().numpy().squeeze(), (z1.detach().numpy().squeeze(), z1.detach().numpy().squeeze())
        return dist
    
    
class NetContPDG_dummy(nn.Module):

    def __init__(self, nin, nout, n_hidden=128, n_hl=5,
                 std_min=0.03, std_max=.5, 
                 masked_ftrs=[], ftr_normalizer=None,
                device=None):
        super().__init__()
        
        if device is None:
            device = torch.device("cpu")
            
        self.nout = nout
        
        self.masked_ftrs = masked_ftrs
        self.ftr_normalizer = ftr_normalizer
        if self.ftr_normalizer is not None:
            self.ftr_normalizer = torch.tensor(ftr_normalizer, dtype=torch.float32,
                                              device=device)
        
        if len(masked_ftrs)>0:
            mask = torch.ones(nin, device=device)
            mask[masked_ftrs]=0
            self.mask = mask.ge(0.5) 
            nin = nin-len(masked_ftrs)
            
            
        self.pdg1 = nn.Parameter(torch.zeros(nout, requires_grad=True))
        torch.nn.init.normal_(self.pdg1, mean=0.5, std=0.1)
        
        self.pdg2 = nn.Parameter(torch.zeros(nout, requires_grad=True))
        torch.nn.init.normal_(self.pdg2, mean=0.5, std=0.1)
        
        self.pdg3 = nn.Parameter(torch.zeros(nout, requires_grad=True))
        torch.nn.init.normal_(self.pdg3, mean=0.5, std=0.1)
        
        self.pdg4 = nn.Parameter(torch.zeros(nout, requires_grad=True))
        torch.nn.init.normal_(self.pdg4, mean=0.5, std=0.1)
        
        self.stdmax = torch.full((1,nout), 0.1, device=device)
                    
        
    def forward(self, x, eps=0, verbose=False):
        # eps indicates the amount of exploration which will
        # affect the variance of actions generated.
        if len(self.masked_ftrs)>0:
            x_ = torch.masked_select(x, self.mask) #x[:,self.mask] #
        else:
            x_ = x
        if self.ftr_normalizer is not None:
            x_ = x_*self.ftr_normalizer
        
#         chosen_ftr = x_[:,3]*x_[:,4]
#         cond1 = (chosen_ftr>0).view(-1,1).repeat(1, self.nout)
#         cond2 = (chosen_ftr<0).view(-1,1).repeat(1, self.nout)
#         mu = self.pdg1*cond1 + self.pdg2*cond2 
        
        ftr1 = x_[:,3]
        ftr2 = x_[:,4]
        cond1 = ((ftr1>0)*(ftr2>0)).view(-1,1).repeat(1, self.nout)
        cond2 = ((ftr1>0)*(ftr2<0)).view(-1,1).repeat(1, self.nout)
        cond3 = ((ftr1<0)*(ftr2>0)).view(-1,1).repeat(1, self.nout)
        cond4 = ((ftr1<0)*(ftr2<0)).view(-1,1).repeat(1, self.nout)
        mu = self.pdg1*cond1 + self.pdg2*cond2 +self.pdg3*cond3 + self.pdg4*cond4
        
        sig = self.stdmax #*(mu.detach())
        dist = MultivariateNormal(mu, scale_tril=torch.diag_embed(sig))
    
        
        if verbose:
            return mu.detach().numpy().squeeze(), (np.zeros(self.nout), np.zeros(self.nout))
        return dist
    
    
class NetContPDG_d(nn.Module):
    
    def __init__(self, nin, nout, n_hidden=None, n_hl=None,
                 init_mu=0.3, init_std=0.1, 
                 out_std=0.1, std_min=None, std_max=None, 
                 masked_ftrs=None, ftr_normalizer=None,
                disc_bounds=[-0.1, 0.1], device=None):
        super().__init__()
        
        if device is None:
            device = torch.device("cpu")
            
        self.nout = nout
        
        self.masked_ftrs = masked_ftrs
        self.ftr_normalizer = ftr_normalizer
        if self.ftr_normalizer is not None:
            self.ftr_normalizer = torch.tensor(ftr_normalizer, dtype=torch.float32,
                                              device=device)
        
        if len(masked_ftrs)>0:
            mask = torch.ones(nin, device=device)
            mask[masked_ftrs]=0
            self.mask = mask.ge(0.5) 
            nin = nin-len(masked_ftrs)
            
        self.nin = nin
        self.disc_bounds = torch.tensor(disc_bounds, device=device)
        self.n_disc = len(disc_bounds)+1
        self.dvec2int_mat = torch.tensor([self.n_disc**i for i in range(nin)], 
                                   device=device).view(-1,1)
        self.n_hidden = self.n_disc**nin
            
        self.fc_mu = nn.Linear(self.n_hidden, nout, bias=False)
        torch.nn.init.normal_(self.fc_mu.weight, mean=init_mu, std=init_std)
        
        self.stdmax = torch.full((1,nout), out_std, device=device)
                    
        
    def forward(self, x, eps=0, verbose=False):
        # eps indicates the amount of exploration which will
        # affect the variance of actions generated.
        if len(self.masked_ftrs)>0:
            x_ = torch.masked_select(x, self.mask).view(-1,self.nin) #x[:,self.mask] #
        else:
            x_ = x
        
        if self.ftr_normalizer is not None:
            x_ = x_*self.ftr_normalizer
        
        
        x_dvec = torch.bucketize(x_, self.disc_bounds).view(-1, self.nin)
#         print('x_dvec = ', x_dvec.shape)

        x_int = torch.mm(x_dvec, self.dvec2int_mat) #Transforms each vector to an integer of base n_disc.
        x_onehot = F.one_hot(x_int, num_classes=self.n_hidden).squeeze(dim=1).type(torch.FloatTensor)
        
        mu = self.fc_mu(x_onehot)
        
        sig = self.stdmax #*(mu.detach())
        dist = MultivariateNormal(mu, scale_tril=torch.diag_embed(sig))
    
        
        if verbose:
            return mu.detach().numpy().squeeze(), (x_dvec.detach().numpy().squeeze(), x_int.detach().numpy().squeeze())
        return dist
    
    
class NetContPDG_d2(nn.Module):

    def __init__(self, nin, nout, n_hidden=1024, n_hl=None,
                 init_mu=0.3, init_std=0.1, 
                 out_std=0.1, std_min=None, std_max=None, 
                 disc_bounds=[-0.1, 0.1], masked_ftrs=[], ftr_normalizer=None,
                device=None):
        super().__init__()
        
        if device is None:
            device = torch.device("cpu")
            
        self.nout = nout
        
        self.masked_ftrs = masked_ftrs
        self.ftr_normalizer = ftr_normalizer
        if self.ftr_normalizer is not None:
            self.ftr_normalizer = torch.tensor(ftr_normalizer, dtype=torch.float32,
                                              device=device)
        
        if len(masked_ftrs)>0:
            mask = torch.ones(nin, device=device)
            mask[masked_ftrs]=0
            self.mask = mask.ge(0.5) 
            nin = nin-len(masked_ftrs)
        self.nin = nin
        self.disc_bounds = torch.tensor(disc_bounds, device=device)
        self.n_disc = len(disc_bounds)+1
#         self.dvec2int_mat = torch.tensor([self.n_disc**i for i in range(nin)], 
#                                    device=device).view(-1,1)
        self.n_hidden = nin*self.n_disc
    
        self.fc_mu = nn.Linear(self.n_hidden, nout, bias=False)
        torch.nn.init.normal_(self.fc_mu.weight, mean=init_mu/nin, std=init_std/nin)
#         torch.nn.init.uniform(self.fc_mu.weight, mean=0.5/n_hidden, std=1e-3)
#         torch.nn.init.constant_(self.fc_mu.weight, 0.1/n_hl)
        
#         self.fc_std = nn.Linear(n_hl, nout, bias=False)
#         torch.nn.init.normal_(self.fc_std.weight, mean=0.0, std=1.)
        self.stdmax = torch.full((1,nout), out_std, device=device)
    
    def forward(self, x, eps=0, verbose=False):
        # eps indicates the amount of exploration which will
        # affect the variance of actions generated.
        if len(self.masked_ftrs)>0:
            x_ = torch.masked_select(x, self.mask).view(-1,self.nin) #x[:,self.mask] #
        else:
            x_ = x
        
        if self.ftr_normalizer is not None:
            x_ = x_*self.ftr_normalizer
        
        x_dvec = torch.bucketize(x_, self.disc_bounds).view(-1, self.nin)


#         x_int = torch.mm(x_dvec, self.dvec2int_mat) #Transforms each vector to an integer of base n_disc.
        x_onehot = F.one_hot(x_dvec, num_classes=self.n_disc).view(-1, self.n_hidden).type(torch.FloatTensor)
        
#         x_symm = torch.cat((x_,-x_), dim=-1)
#         z1 = F.softmax(self.fc1(x_symm), dim=-1)
#         z2 = F.softmax(self.fc2(z1), dim=-1)
#         z1 = torch.sigmoid(self.fc1(x_symm))
#         print('x_onehot shape= ', x_onehot.shape)
        mu = self.fc_mu(x_onehot)
        
        sig = self.stdmax #*(mu.detach())
        dist = MultivariateNormal(mu, scale_tril=torch.diag_embed(sig))
    
        
        if verbose:
            return mu.detach().numpy().squeeze(), (x_dvec.detach().numpy().squeeze(), x_onehot.detach().numpy().squeeze())
        return dist
    
class NetOhOhL(nn.Module):

    def __init__(self, nin, nout, n_hl=256,
                 init_mu=0.3, init_std=0.1, 
                 out_std=0.1, std_min=None, std_max=None, 
                 disc_bounds=[-0.1, 0.1], masked_ftrs=[], ftr_normalizer=None,
                device=None):
        super().__init__()
        
        if device is None:
            device = torch.device("cpu")
            
        self.nout = nout
        
        self.masked_ftrs = masked_ftrs
        self.ftr_normalizer = ftr_normalizer
        if self.ftr_normalizer is not None:
            self.ftr_normalizer = torch.tensor(ftr_normalizer, dtype=torch.float32,
                                              device=device)
        
        if len(masked_ftrs)>0:
            mask = torch.ones(nin, device=device)
            mask[masked_ftrs]=0
            self.mask = mask.ge(0.5) 
            nin = nin-len(masked_ftrs)
        self.nin = nin
        self.disc_bounds = torch.tensor(disc_bounds, device=device)
        self.n_disc = len(disc_bounds)+1
#         self.dvec2int_mat = torch.tensor([self.n_disc**i for i in range(nin)], 
#                                    device=device).view(-1,1)
        self.n_hidden = nin*self.n_disc
        self.n_hl = n_hl
        
        self.fc1 = nn.Linear(self.n_hidden, n_hl, bias=False)
        torch.nn.init.normal_(self.fc1.weight, mean=1., std=1.)
#         torch.nn.init.sparse_(self.fc1.weight, sparsity=1-1./(self.n_hidden-4), std=2.)#n_hl/n_hidden)#
#         torch.nn.init.constant_(self.fc1.weight, 0.15)
#         torch.nn.init.xavier_normal_(self.fc1.weight, gain=1.0)
        
        self.DO = nn.Dropout(0.)
        self.fc_mu = nn.Linear(n_hl, nout, bias=False)
        torch.nn.init.normal_(self.fc_mu.weight, mean=init_mu, std=init_std)
        
        self.stdmax = torch.full((1,nout), out_std, device=device)
    
    def forward(self, x, eps=0, verbose=False):
        # eps indicates the amount of exploration which will
        # affect the variance of actions generated.
        if len(self.masked_ftrs)>0:
            x_ = torch.masked_select(x, self.mask).view(-1,self.nin) #x[:,self.mask] #
        else:
            x_ = x
        
        if self.ftr_normalizer is not None:
            x_ = x_*self.ftr_normalizer
        
        x_dvec = torch.bucketize(x_, self.disc_bounds).view(-1, self.nin)


#         x_int = torch.mm(x_dvec, self.dvec2int_mat) #Transforms each vector to an integer of base n_disc.
        x_onehot = F.one_hot(x_dvec, num_classes=self.n_disc).view(-1, self.n_hidden).type(torch.FloatTensor)
        
        z_l = F.softmax(self.fc1(x_onehot), dim=-1)
#         z_l = F.relu(self.fc1(x_onehot))
        
#         x_symm = torch.cat((x_,-x_), dim=-1)
#         z1 = F.softmax(self.fc1(x_symm), dim=-1)
#         z2 = F.softmax(self.fc2(z1), dim=-1)
#         z1 = torch.sigmoid(self.fc1(x_symm))
#         print('x_onehot shape= ', x_onehot.shape)
        mu = self.fc_mu(self.DO(z_l))
        
        sig = self.stdmax #*(mu.detach())
        dist = MultivariateNormal(mu, scale_tril=torch.diag_embed(sig))
    
        if verbose:
            return mu.detach().numpy().squeeze(), (z_l.detach().numpy().squeeze(), x_onehot.detach().numpy().squeeze())
        return dist
    

class NetContPID_OhSoft(nn.Module):

    def __init__(self, nin, nout, kernel, n_hidden=1024, n_hl=None,
                 init_mu=0.3, init_std=0.1, 
                 out_std=0.1, std_min=None, std_max=None, 
                 disc_bounds=[-0.1, 0.1], masked_ftrs=[], ftr_normalizer=None,
                device=None):
        super().__init__()
        
        if device is None:
            device = torch.device("cpu")
        self.device = device
            
        self.nout = nout
        
        self.masked_ftrs = masked_ftrs
        self.ftr_normalizer = ftr_normalizer
        if self.ftr_normalizer is not None:
            self.ftr_normalizer = torch.tensor(ftr_normalizer, dtype=torch.float32,
                                              device=device)
        
        if len(masked_ftrs)>0:
            mask = torch.ones(nin, device=device)
            mask[masked_ftrs]=0
            self.mask = mask.ge(0.5) 
            nin = nin-len(masked_ftrs)
        self.nin = nin
        self.disc_bounds = torch.tensor(disc_bounds, device=device)
        self.n_disc = len(disc_bounds)+1
        self.n_hidden = self.n_disc **nin
        self.n_kern = len(kernel)
        self.kernel = torch.tensor(kernel, dtype=torch.float32, device=device).view(1,1, self.n_kern,self.n_kern,self.n_kern)
        
        self.fc_mu = nn.Linear(self.n_hidden, nout, bias=False)
        torch.nn.init.normal_(self.fc_mu.weight, mean=init_mu, std=init_std/nin)
        
        self.stdmax = torch.full((1,nout), out_std, device=device)
    
    def forward(self, x, eps=0, verbose=False):
        # eps indicates the amount of exploration which will
        # affect the variance of actions generated.
        if len(self.masked_ftrs)>0:
            x_ = torch.masked_select(x, self.mask).view(-1,self.nin)
        else:
            x_ = x
        
        if self.ftr_normalizer is not None:
            x_ = x_*self.ftr_normalizer
        
        b_size = x_.shape[0]
        x_dvec = torch.bucketize(x_, self.disc_bounds).view(-1, self.nin)
        x_onehot = torch.zeros(b_size, 1, self.n_disc, self.n_disc, self.n_disc, dtype=torch.float32, device=self.device)
        for i in range(b_size):
            x_onehot[i,0, x_dvec[i,0],x_dvec[i,1],x_dvec[i,2]] = 1
        x_ohsoft = F.conv3d(x_onehot, self.kernel, stride=1, padding=int(self.n_kern/2), dilation=1).view(b_size, -1)

#         print('x_ohsoft.shape: ', x_ohsoft.shape)
        mu = self.fc_mu(x_ohsoft)
        
        sig = self.stdmax #*(mu.detach())
        dist = MultivariateNormal(mu, scale_tril=torch.diag_embed(sig))
    
        if verbose:
            return mu.detach().numpy().squeeze(), (x_dvec.detach().numpy().squeeze(), x_ohsoft.detach().numpy().squeeze())
        return dist