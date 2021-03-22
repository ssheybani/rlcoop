import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch import distributions
import numpy as np

    

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
        self.bn4 = nn.BatchNorm1d(128)
        
        self.fc5 = nn.Linear(n_hidden, nout)
        torch.nn.init.normal_(self.fc5.weight, mean=0.0, std=0.03)
        
    def forward(self, x, return_hidden=False):
        if x.dim()==1:
            x = x.view(1,-1)
        z1 = F.relu(self.fc1(x))
        z2 = F.glu(self.bn2(self.fc2(z1)))
        z3 = F.relu(self.fc3(z2))
        z4 = F.glu(self.bn4(self.fc4(z3)))
        out = self.fc5(z4)
#         z3 = F.relu(self.fc3(z2))
#         out = self.fc4(z3)
        if out.shape[0]==1:
            out = out.view(-1)
        if return_hidden:
            return out, z4
        else:
            return out
        
        
class NetL1G(nn.Module):
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
        if out.dim()>1:
            mu, log_std = out[:,0], out[:,1]
        else:
            mu, log_std = out
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
        
        torch.nn.init.normal_(self.fc4.weight, mean=0.0, std=0.5)
        torch.nn.init.normal_(self.fc5.weight, mean=0.0, std=0.03)
        
    def forward(self, x):
        z1 = F.relu(self.fc1(x))
        z2 = F.glu(self.fc2(z1))
        z3 = F.relu(self.fc3(z2))
        z4 = F.glu(self.fc4(z3))
        out = self.fc5(z4)
        
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
        
        if x.dim()>1:
            features = torch.cat((x, self.gru1_hn), 1)
            out = self.fc1(features)
            mu, log_std = out[:,0], out[:,1]
        else:
            features = torch.cat((x, hidden1), 0)
            out = self.fc1(features)
            mu, log_std = out
#         sig = nn.Softplus()(sig_)
        
        log_std = log_std -1. #@@@@@@ to improve the range
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        sig = log_std.exp()
        
        dist = distributions.normal.Normal(mu, sig)
#         if out.dim()>1:
#             print('mu, sigma', mu, sig)
        return dist