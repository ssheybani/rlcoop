
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import FloatTensor as tarr
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
import time
import scipy
from scipy import signal
from sklearn import metrics, preprocessing


"""
What we want from the RNN?
Integrate the weighted stimuli over time and "fire" differentiably.

h[t] =  A h[t-1] + B x[t]
y[t] = tanh ( C h[t] + D x[t])


The sequence length can be either around 10 or around 100.

A,B,C,D must be trained by SGD. 
    A represents the internal dynamics and ideally provides rich dynamics.
        For that purpose, it's best if it has complex eigen values inside
        the unit circle, somewhat close to the radius. -1/tau.
        # lambda = exp(-ln(50) /4tau_d), where tau_d is an integer: tau_d=tau/tstep
        # For initialization, we can randomly choose tau from the range [0.5-2]
     
    B is the effect of input on the hidden state. 
    It'd make sense if B = 1-lambda, so that we get something like h= alpha h + (1-alpha)x
    
    C,D are the readout gains
        In initialization, set D smaller than C.
        They may get a larger learning rate.
    
    The potential problem of vanishing gradients must be dealt with.

"""

class Net_R1L(nn.Module):
    def __init__(self, nin, nout, n_hidden, n_layer=1):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_layer=n_layer
        self.rnn = nn.RNN(nin, n_hidden, n_layer, 
                          nonlinearity ='relu', batch_first=True, dropout=0).double()
        # If batch_first=True, then the input and output tensors are provided as (batch, seq, feature) instead of (seq, batch, feature). Note that this does not apply to hidden or cell states. 
        self.readout = nn.Linear(n_hidden, nout).double()
        self.hidden_cell = None
        
        
    def get_init_state(self, batch_size, device=None):
        # self.hidden_cell = torch.zeros(1, batch_size, self.n_hidden).double()#.to(device),
                            # torch.zeros(1, batch_size, self.n_hidden).to(device))
        return torch.zeros(self.n_layer, batch_size, self.n_hidden).double()
    
    
    def forward(self, state):
        batch_size = state.shape[0]
        # device = state.device
        # if self.hidden_cell is None or batch_size != self.hidden_cell.shape[1]:
        #     self.get_init_state(batch_size, device)
        hidden = self.get_init_state(batch_size)
        # _, self.hidden_cell = self.rnn(state, self.hidden_cell)
        _, hidden = self.rnn(state, hidden)
        out = torch.tanh(
            self.readout(hidden[-1,...]))
            # self.readout(self.hidden_cell[0,...]))
        return out

    
"""
Consider creating a Trajectory Dataset like in the example
It's an iterable, implementing __iter__, __next__
Never mind! Don't use GPU at all! We don't have a multilayer CNN.
"""
def create_dataloader_from_ts(input_ts, output_ts, tw):
    """
    Parameter
        input_ts: 1D array/list 
        tw: int time width
        
    Returns
        a list of tuples. In each tuple, the first element is the sequence and 
        the second element is the label.
        
    """
    inout_seq = []
    L = input_ts.shape[0]
    for i in range(tw,L):
        train_seq = input_ts[i-tw:i, ...]
        train_label = output_ts[i-1, ...]
        inout_seq.append((train_seq ,train_label))

    return inout_seq 

# Prepare the mock dataset
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
    
class ToyProcess():
    def __init__(self):
        self.svar1, self.svar2 = 0., 0.
    def reset(self):
        self.__init__()
    def respond(self, inputs):
        in1,in2 = inputs
        self.svar1 = 0.9*self.svar1 -0.1*self.svar2 +0.15*in1 +0.05*in2
        self.svar2 = 0.2*self.svar1 +0.8*self.svar2 -0.2*in1 +0.2*in2
        out = np.tanh(0.5*self.svar1+ 0.5*self.svar2)
        return out

toy_sys = ToyProcess()
dt=0.02
tt = np.arange(0, 1000, dt)
# sig1 = np.sin(tt)
freq1 = np.array([0.1, 0.25, 0.55])
amp1 = 0.1/(freq1*2*np.pi)
sig1, _ = sumOfSines(tt, freq1, amp1)

# sig2 = np.zeros_like(tt)


# for i, xtt in enumerate(tt):
#     if i%50==0:
#         omega = 3+ (np.random.rand()-0.5)
#     sig2[i] = np.cos(omega*xtt)
# sig2 = np.pad(sig1, pad_width=(int(0.2/dt),0), mode='constant',constant_values=0.)
# sig2 = sig2[:len(tt)]
sig2 = np.zeros_like(tt)
tmp_dsig2, tmp_ddsig2 = 0.,0.
for i, xtt in enumerate(tt):
    if i%1==0:
        tmp_dsig2 = 1*(np.random.rand()-0.5)
    sig2[i] = 0.
    if i>(0.3/dt):
        tmp = sig1[i-int(0.2/dt)]
        sig2[i] = sig2[i-1]+ dt*(tmp_dsig2+ 0.8*(tmp-sig2[i-1]))

out_var = np.zeros_like(tt)      
for i, xtt in enumerate(tt):
    out_var[i] = toy_sys.respond((sig1[i],sig2[i]))

tslice=slice(0,500)
fig, ax = plt.subplots()
ax.plot(tt[tslice], sig1[tslice], label='input1')
ax.plot(tt[tslice], sig2[tslice], label='input2')
ax.plot(tt[tslice], out_var[tslice], label='output')
ax.legend()

xdata_in = np.concatenate((sig1.reshape(-1,1),sig2.reshape(-1,1)),axis=1)
xdata_out = out_var

if np.exp(abs(np.log(np.std(sig2)/np.std(xdata_out)))) >2:
    raise ValueError('Separate scaling ratios are needed')
    
scale_r = 1./(4*np.std(xdata_out))
# Train the network
n_hidden = 5
xnet1 = Net_R1L(2, 1, n_hidden, n_layer=1)

lr = 1e-3
wd = lr
amsgrad = False

n_epoch = 50
# The sequence length is chosen based on an assumption about the largest time 
# constant of the hidden processes that generate the data. The goal is that the
# sequence is large enough that the effect of the input signals completely unfold
# within one sequence.
# If the generative process is described by a scalar linear discrete state space model, 
#      x[t] = alpha x[t-1] + beta u[t], 
# then the following relationship holds between its time constant and alpha:
#      tau = -np.log(50)/(4*np.log(alpha))
# To ensure the sequence cover all processes, the largest alpha must be used.
# Note that because of the log, tau is very sensitive to alpha.
alpha = 0.9 # In this case, the two hidden processes have alpha=0.7, 0.9. 
tau = -np.log(50)/(4*np.log(alpha))
seq_len = 2*int(tau)


# seq_gen = lambda ts,i, tau: ts[i-tau:i, ...]
training_data = create_dataloader_from_ts(scale_r*xdata_in, scale_r*xdata_out, seq_len)
train_loader = DataLoader(training_data, batch_size=64, shuffle=True)


criterion = nn.MSELoss()
optimizer = optim.Adam(xnet1.parameters(), lr=lr, 
                       weight_decay=wd, amsgrad=amsgrad)

t0 = time.time()
pcurve_x, pcurve_y = [], []
for i_ep in range(n_epoch):
    for batch_idx, (xfeatures, xtarget) in enumerate(train_loader):
        # print(xfeatures.shape, xtarget.shape)
        # state = x_data_in
        optimizer.zero_grad()
        xpred = xnet1(xfeatures)
        loss = 100*criterion(xpred.view(-1), xtarget)
        loss.backward()
        optimizer.step()
    pcurve_x.append(i_ep)
    pcurve_y.append(loss.item())
    if i_ep % 5 == 1:
        tmp_t = time.time()-t0
        print('Processing %d th epoch from %d . ETA: %d seconds'
              %(i_ep, n_epoch, (n_epoch/i_ep -1.)*tmp_t))
        print("Epoch {}: loss {}".format(i_ep, loss.item()))
    
fig,ax = plt.subplots()
# ax.plot(pcurve_x, np.log10(pcurve_y))
ax.plot(pcurve_x, pcurve_y)
ax.set_xlabel('epoch')
ax.set_ylabel('MSE loss (scaled by 100)')
# ax.set_ylim([0,2])

# Compare the trained model with the toy model on other signals
duration=10.
tt_test = np.arange(0,duration, 0.02)
# sig1_test = signal.gausspulse(tt_test-duration/2, fc=0.5)
# sig2_test = signal.chirp(tt_test, f0=0.5, f1=0.2, t1=duration, 
                          # phi=45, method='linear')

sig1_test = sig1
sig2_test = sig2
scale_r = 1/(4*np.std(np.array([sig1_test, sig2_test])))

sig1_test_te = scale_r*tarr(sig1_test).view(1,-1,1).double()
sig2_test_te = scale_r*tarr(sig2_test).view(1,-1,1).double()

sys_out = np.zeros_like(tt_test)
model_out = np.zeros_like(tt_test)

toy_sys.reset()
for i, xtt in enumerate(tt_test):
    sys_out[i] = toy_sys.respond((sig1_test[i],sig2_test[i]))
    
    if i<seq_len:
        input_te = torch.cat(
            (sig1_test_te[:,:i+1,:], sig2_test_te[:,:i+1,:]), 2)
    else:
        input_te = torch.cat(
            (sig1_test_te[:,i-seq_len:i+1,:], sig2_test_te[:,i-seq_len:i+1,:]), 2)
    model_out[i] = xnet1(input_te).view(-1)

model_pred = model_out/scale_r

tslice = slice(2,500)
fig, ax = plt.subplots()
ax.plot(tt_test[tslice], sig1_test[tslice], label='input1', ls='--', alpha=0.7)
ax.plot(tt_test[tslice], sig2_test[tslice], label='input2', ls='--', alpha=0.7)
ax.plot(tt_test[tslice], sys_out[tslice], label='sys output')
ax.plot(tt_test[tslice], model_pred[tslice], label='model output')
ax.set_xlabel('time(s)')
ax.set_ylabel('signal')
ax.legend()
ax.set_ylim([-.5/scale_r,.5/scale_r])
# ax.set_ylim([-2,2])
mse_str = 'MSE = %.3f' %(
    metrics.mean_squared_error(sys_out, model_pred))
ax.annotate(mse_str, (0.8,0.9), xycoords='axes fraction')

