# Feature encoding
# Approach 1: Discretize + Convolve
b_size = 256
n_try = 200

t0 = time.time()
for j in range(n_try):
    x_ = 4*(2*torch.rand(b_size, 3)-1)

    x_dvec = torch.bucketize(x_, disc_bounds).view(-1, nin) #30ms
    x_oh = torch.zeros(b_size, 1, n_disc, n_disc, n_disc, dtype=torch.float32) #5ms
    for i in range(b_size): #5ms
        x_oh[i,0, x_dvec[i,0],x_dvec[i,1],x_dvec[i,2]] = 1
    # x_oh
    xtt = F.conv3d(x_oh, kernel3d, stride=1, padding=int(n_kern/2), dilation=1) #5ms

print((time.time() - t0)/n_try)

# Approach 2: Shift + Meshgrid (slower and maybe less accurate)
n_try = 200
b_size = 256
n_disc = len(kernel1d)

t0 = time.time()

for j in range(n_try):
    x_ = 4*(2*torch.rand(b_size, 3)-1)
    x_oh = torch.zeros(b_size, 1, n_disc, n_disc, n_disc, dtype=torch.float32) #5ms
    for i in range(b_size): #5ms
        xtt_x0 = torch.roll(kernel1d, int(x_[i,0])) 
        xtt_x1 = torch.roll(kernel1d, int(x_[i,1]))
        xtt_x2 = torch.roll(kernel1d, int(x_[i,2]))
        x_oh[i,0,...] = torch.matmul( torch.outer(xtt_x0, xtt_x1).unsqueeze(dim=-1), xtt_x2.unsqueeze(dim=0)) #10ms

print((time.time() - t0)/n_try)


# For reshaping a 1D list to a 2D list
def list_reshape(lst, new_shape):
    n_row, n_col = new_shape
    if len(lst)!= n_row*n_col:
        raise ValueError
    new_lst = []
    for i in range(n_row):
        tmp_lst = []
        for j in range(n_col):
            tmp_lst.append(lst[i*n_col +j])
        new_lst.append(tmp_lst)
    return new_lst

n_runs=3
trained_agents_resh = list_reshape(trained_agents, ( int(len(trained_agents)/n_runs), n_runs ))
len(trained_agents_resh)


def pid_interp_te(te, scalers=None):
    xtt = te>1; f_xtt = te+20
    ytt = te<0; f_ytt = te+1
    ztt = (te<=1) * (te>=0); f_ztt = 20*te+1
    te_c = f_xtt*xtt + f_ytt*ytt + f_ztt*ztt
    
    if scalers is not None:
        return scalers *te_c
    return te_c