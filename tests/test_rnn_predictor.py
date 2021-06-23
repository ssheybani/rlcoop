import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import FloatTensor as tarr
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
import os, time, pickle
# from scipy import signal
from sklearn import metrics, preprocessing

import context
from context import rlcoop, DATA_PATH, CONFIG_PATH

def create_dataset_from_eplist(ep_list, inout_idx, tw, scaler=None):
    """
    Parameter
        ep_list: list of 2d arrays
            each 2d array is a time series of shape (n_samples, n_features)
        in_idx, out_idx: int or sequence or slice
            specifies the indices on the data to be selected for input and output
        tw: int 
            time width
        scaler: object
            has method transform(data) 
                with data being of shape (n_samples, n_features)
        
    Returns
        a list of tuples. In each tuple, the first element is the sequence and 
        the second element is the label.
        
    """
    in_idx, out_idx = inout_idx
    inout_seq = []
    for ep_ts in ep_list:
        if scaler is not None:
            ep_ts = scaler.transform(ep_ts)
        L = ep_ts.shape[0]
        for i in range(tw,L):
            train_seq = ep_ts[i-tw:i, in_idx]
            train_label = ep_ts[i-1, out_idx]
            inout_seq.append((train_seq ,train_label))

    return inout_seq 

def get_nn_predictions(model, ep_ts, seq_len, in_idx, out_idx, scaler=None):
    if scaler is not None:
        ep_ts = scaler.transform(ep_ts)
    L = ep_ts.shape[0]
    preds = []
    ep_ts_te = tarr(ep_ts).unsqueeze(0).double()
    for i in range(L):
        if i<seq_len:
            input_te = ep_ts_te[:,:i+1, in_idx]
        else:
            input_te = ep_ts_te[:,i-seq_len:i+1, in_idx]
        preds.append(
            xnet1(input_te).squeeze().detach().numpy())
    preds = np.asarray(preds)*scaler.scale_[out_idx]
    return preds

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


            
            
n_ep_to_load = 1000

dataset_name = 'ds_run_17.59.14'
dataset_dir = DATA_PATH + dataset_name

ep_trials = []
for filename in os.listdir(dataset_dir):
    if len(ep_trials)<n_ep_to_load:
        with open(dataset_dir+'/'+filename, 'rb') as f:
            file_eps = pickle.load(f)
            ep_trials += file_eps

# ep_trials = np.asarray(ep_trials)
ts_dict = {'t':0,
            'e1':1, 'e1dot':2, 'fn1':3, 'act1':4, 'f1':5, 
            'e2':6, 'e2dot':7, 'fn2':8, 'act2':9,'f2':10}

# for i, ep in enumerate(ep_trials):
#     tmp = ep_trials[i][:, ]

# Prepare the dataloader
in_idx = [ts_dict['f1'], ts_dict['fn1']]
out_idx = slice(1,3)
dt_d = 0.02
batch_size = 64
seq_len = int(1./dt_d)
ep_trials_arr_tmp = np.array(ep_trials).reshape(-1,11)
scaler = preprocessing.MaxAbsScaler().fit(ep_trials_arr_tmp)
del ep_trials_arr_tmp
training_data = create_dataset_from_eplist(ep_trials, (in_idx, out_idx), 
                                           seq_len, scaler=scaler)
train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
del ep_trials
# Prepare the PyTorch elements: network model and the optimizer
n_hidden = 5
xnet1 = Net_R1L(2, 2, n_hidden, n_layer=2)

lr = 1e-3
wd = lr
amsgrad = False

n_epoch = 1
criterion = nn.MSELoss()
optimizer = optim.Adam(xnet1.parameters(), lr=lr, 
                       weight_decay=wd, amsgrad=amsgrad)

# Training loop
t0 = time.time()
pcurve_x, pcurve_y = [], []
n_batch = len(train_loader)
for i_ep in range(n_epoch):
    for batch_idx, (xfeatures, xtarget) in enumerate(train_loader):
        # print(xfeatures.shape, xtarget.shape)
        # state = x_data_in
        optimizer.zero_grad()
        xpred = xnet1(xfeatures)
        loss = 100*criterion(xpred, xtarget)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 1000 == 1:
            pcurve_x.append(batch_idx)
            pcurve_y.append(loss.item())
        
            tmp_t = time.time()-t0
            print('Processing %d th batch from %d . ETA: %d seconds'
                  %(batch_idx, n_batch, (n_batch/batch_idx -1.)*tmp_t))
            print("Batch {}: loss {}".format(batch_idx, loss.item()))
            
fig,ax = plt.subplots()
# ax.plot(pcurve_x, np.log10(pcurve_y))
ax.plot(pcurve_x, pcurve_y)
ax.set_xlabel('batch')
ax.set_ylabel('MSE loss (scaled by 100)')
# ax.set_ylim([0,2])

# Visualize the predictions
n_ep_to_load = 1 #1000

dataset_name = 'ds_run_17.59.14'
dataset_dir = DATA_PATH + dataset_name

ep_test_trials = []
for filename in os.listdir(dataset_dir):
    if len(ep_test_trials)<n_ep_to_load:
        with open(dataset_dir+'/'+filename, 'rb') as f:
            file_eps = pickle.load(f)
            ep_test_trials += file_eps

ep_num = 33
test_ep = ep_test_trials[ep_num]
# Form the predictions
preds = get_nn_predictions(xnet1, test_ep, seq_len, 
                           in_idx, out_idx, scaler=scaler)
fig, ax = plt.subplots(2,1, sharex=True)
ax[0].plot(test_ep[:,ts_dict['t']], test_ep[:,ts_dict['e1']], label='e target')
ax[0].plot(test_ep[:,ts_dict['t']], preds[:,0], ls='--', label='e pred')
ax[1].plot(test_ep[:,ts_dict['t']], test_ep[:,ts_dict['e1dot']], label='e\' target')
ax[1].plot(test_ep[:,ts_dict['t']], preds[:,1], ls='--', label='e\' pred')
for axij in ax:
    axij.legend()
ax[-1].set_xlabel('Time(s)')

