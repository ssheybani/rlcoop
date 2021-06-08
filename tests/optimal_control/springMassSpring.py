import numpy as np
import matplotlib.pyplot as plt

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


######################
# Sys config
######################

m1 = 1
m2 = 1
M = 5
k1 = 100.#1000 #For k>100, x1, xm remain indistinguishably close.
k2 = 100
zeta = 1.
b = zeta * np.sqrt((M+m1)*8*k1)

# Controller gains
Kp1 = 0. #20
Kp2 = 0. #10

Kd1 = 0.#30.
Kd2 = 0. #30.

Ki1 = 1.5
Ki2 = 1.5

Kp = np.array([Kp1, -Kp2])
Kd = np.array([Kd1, -Kd2])
Ki = np.array([Ki1, -Ki2])


# State vector q
# q = [x1, x2, xm, dx1, dx2, dxm]
# pdict = {'x1':0, 'x2':1, 'xm':2,
#           'dx1':3, 'dx2':4, 'dxm':5}
X1i, X2i, XMi, dX1i, dX2i, dXMi = 0,1,2,3,4,5
sys_MSMSM = lambda q, u, t: np.array([q[dX1i],
                                      q[dX2i],
                                      q[dXMi],
                                      (1/m1)*( u[0] + k1*(q[XMi]-q[X1i]) + b*(q[dXMi]-q[dX1i])),
                                      (1/m2)*(-u[1] + k2*(q[XMi]-q[X2i]) + b*(q[dXMi]-q[dX2i])),
                                      (1/M)*(k1*(q[X1i]-q[XMi]) + k2*(q[X2i]-q[XMi]) + b*(q[dX1i]-q[dXMi])+ b*(q[dX2i]-q[dXMi]))])


kf1, kf2 = 0.5, 0.5

sys_A = np.array([
    [0, 0, 0, 1., 0, 0],
    [0, 0, 0, 0, 1., 0],
    [0, 0, 0, 0, 0, 1.],
    [-k1/m1, 0, k1/m1, -b/m1, 0, b/m1],
    [0, -k2/m2, k2/m2, 0, -b/m2, b/m2],
    [k1/M, k2/M, (-k1-k2)/M, b/M, b/M, -2*b/M],
     ])

sys_B = np.array([
    [0, 0],
    [0, 0],
    [0, 0],
    [1./m1, 0],
    [0, 1./m2],
    [0, 0]
    ])
sys_C = np.array([
    [0, 0, -1., 0, 0, 0,],
    [0, 0, 0, 0, 0, -1.,],
    [kf1, 0, -kf1, 0, 0, 0],
    [0, kf2, -kf2, 0, 0, 0]
    ])
sys_Dp = np.array([
    [1, 0],
    [0, 1],
    [0, 0],
    [0, 0]
    ]) 

# U' = [r, r']
# U = [f1, f2]

dt = 0.05 #0.025
sys_Ap = sys_A*dt + np.eye(6)
sys_Bp = sys_B*dt 

def term_i(i, A, B, C):
    CA = C 
    j=0
    while j<i:
       CA = np.matmul(CA, A)
       j+=1
    return np.matmul(CA, B)


# Optimal control law
#    u = -0.5 * np.matmul( 
                # Cu^T * Y_rx  +   Y_rx ^T * Cu , 
                # inv( Cu^T Q Cu)
                #        ) 
    
# sys_MSMSM = lambda q, u, t: np.array([q[3],
#                                       q[4],
#                                       q[5],
#                                       (1/m1)*(u[0] + k1*(q[2]-q[0]) + b*(q[5]-q[3])),
#                                       (1/m2)*(-u[1] + k2*(q[2]-q[1]) + b*(q[5]-q[4])),
#                                       (1/M)*(k1*(q[0]-q[2]) + k2*(q[1]-q[2]) + b*(q[3]-q[5])+ b*(q[4]-q[5]))])


######################
# Reference config
######################
dt = 0.001
duration = 10
t = np.arange(0,duration, dt)

freq1 = np.array([0.1, 0.25, 0.55])
#req1 = np.array([0.05, 0.15])
amp1 = 0.1/(freq1*2*np.pi)
#pha1 = np.random.uniform(-np.pi, np.pi, freq1.shape)
# u1 = sumOfSines(t, freq1, amp1)

ref, refdot = sumOfSines(t, freq1, amp1)


# freq2 = np.array([-0.1, 0.15, 0.35])
# #freq2 = np.array([0.05, 0.25])
# amp2 = 0.1/(freq2*2*np.pi)
# #pha2 = np.random.uniform(-np.pi, np.pi, freq2.shape)
# u2 = sumOfSines(t, freq2, amp2)


# freq2 = np.array([0])
# amp2 = np.array([0.1])
# pha2 = np.array([np.pi/2])
# u2 = sumOfSines(t, freq2, amp2, pha2)

# u2 = -0.5*u1

######################
# Controller Config
######################
# pcont = lambda e: Kp1*e


######################
# Simulation Loop
######################
# u = np.stack((u1, u2), 0)
u = np.zeros((2, len(t)))
q = np.zeros((6,len(t)))

e_int = 0.
alpha=0.
for k in range(len(t[:-1])):
    e = ref[k] - q[2,k]
    e_int = alpha*e_int + (1-alpha)*e
    edot = refdot[k] - q[5,k]
    u[:,k] = Kp*e + Kd*edot + Ki*e_int
    q[:,k+1] = RK4Step(q[:,k], sys_MSMSM, dt, u=u[:,k], t=t)

x1 = q[0,:]
x2 = q[1,:]
xm = q[2,:]
f1 = u[0,:]
f2 = u[1,:]
fn1 = k1*(xm-x1)
fn2 = -k2*(xm-x2)

######################
# Analysis
######################

plt.figure(1)
plt.plot(t, x1, label = r'x_1')
plt.plot(t, x2, label = r'x_2')
plt.plot(t, xm, label = r'x_m')
plt.plot(t, ref, 'k--', label = r'ref')
plt.legend()

fig2, ax2 = plt.subplots(2,1, sharey= True, sharex = True)
ax2[0].plot(t, f1, 'r--', label = 'applied force')
ax2[0].plot(t, -fn1, 'g', label = '(inverted) normal force')
ax2[0].plot(t, fn1+f1, 'b:', label = 'diff normal and applied')
ax2[0].axhline(0, lw=0.3, color='k')
ax2[0].legend()

ax2[1].plot(t, f2, 'r--')
ax2[1].plot(t, -fn2, 'g')
ax2[1].plot(t, fn2+f2, 'b:')
ax2[1].set_ylim([-3,3])
ax2[1].axhline(0, lw=0.3, color='k')

plt.show()