"""
X' = A_x X + BU
R' = A_r R

E' = R'-X' = A_r R - A_x X + (-BU)
E' = A_x (E) +(-A_x+A_r)R +(-BU)

r0_vec = (-A_x+A_r)R  #(6x1)

A_e = [[A_x, r0_vec],
       0,     1]
Be = -B

E= [r-x],
    1]

r[t] = r0
r'[t] = rdot0
R[t] = []


"""

"""
Method:
    
The solution to the short-horizon discrete-time optimal control problem is 
found by forming the most recent state-space representation of the system 
(using the ground-truth knowledge of the system dynamics), finding P, K by 
numerically solving the Riccati difference equation backwards in time.
https://en.wikipedia.org/wiki/Linear%E2%80%93quadratic_regulator#Finite-horizon,_discrete-time_LQR


SS representation is created in error coordinates. However, the following 
assumptions allow for A to be an affine matrix:
    1- No dynamics for r. Just replace x, x’ with x-x_d and x’-v_d. 
    x_d would be the average r for the horizon, assuming constant speed for r. 
    This tracking goal is equivalent to aiming to reach r0+r’T/2 while 
    maintaining constant speed of r’. 
    
    We can increase the Q for x’-v_r to get a controller closer to a 
    D-controller (which is a sufficient solution for this task)
    

Riccati difference equation:
    P_{k-1} = A^T P_k A -(A^T P_k B +N) (R+B^T P_k B)^{-1} (B^T P_k A +N^T) +Q
    terminal condition: P_N = Q
    
    F_k = (R +B^T P_{k+1} B)^{-1} (B^T P_{k+1} A +N^T)
    u_k = -F_k x_k
    
    
Forming the matrices:

Ac, Bc are given.
Ad = exp(Ac*Ts)
Bd = inv(A)*(Ad-I) Bc ...OR (Ad-I)*inv(A) Bc
Doesn't make a difference because A^{-1} Ad = Ad A^{-1}. Can be shown using the
Taylor expansion of Ad in terms of Ac.
http://eceweb1.rutgers.edu/~gajic/solmanual/slides/chapter8_DIS.pdf
https://en.wikipedia.org/wiki/Discretization
Then at each planning time, we form the affine version of Ad, Bd to achieve xd, vd:

    vd = r'[t]
    xd = r[t]+ 0.5*vd* Th
    xd_res = xd * sum_{j=1:3} Ad[:,j] + vd * sum_{j=4:6} Ad[:,j] -xd
    vd_res = xd * sum_{j=1:3} Ad[:,j] + vd * sum_{j=4:6} Ad[:,j] -vd

    A = [[Ad[0:2], xd_res], 
         [Ad[3:5], vd_res], 
         [0      , 1]]
    B = [[Bd],
         [0]]

    Calculate all P_k backwards
    Calculate all F_k backwards
    Form u_k


Try with Q=R=I, N=0, Th = 0.5

"""
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import functools, itertools

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

######################
# Sys config
######################

m1 = 1
m2 = 1
M = 5
k1 = 100.#1000 #For k>100, x1, xm remain indistinguishably close.
k2 = 100.
zeta = 1.
b = zeta * np.sqrt((M+m1)*8*k1)

# Controller gains
Kp1 = 0. #20
Kp2 = 0. #10

Kd1 = 30.
Kd2 = 30.

Kp = np.array([Kp1, -Kp2])
Kd = np.array([Kd1, -Kd2])


# State vector q
# q = [x1, x2, xm, dx1, dx2, dxm]
# pdict = {'x1':0, 'x2':1, 'xm':2,
#           'dx1':3, 'dx2':4, 'dxm':5}
X1i, X2i, XMi, dX1i, dX2i, dXMi = 0,1,2,3,4,5
sys_MSMSM = lambda q, u, t: np.array([q[dX1i],
                                      q[dX2i],
                                      q[dXMi],
                                      (1/m1)*( u[0] + k1*(q[XMi]-q[X1i]) + b*(q[dXMi]-q[dX1i])),
                                      (1/m2)*( u[1] + k2*(q[XMi]-q[X2i]) + b*(q[dXMi]-q[dX2i])),
                                      (1/M)*(k1*(q[X1i]-q[XMi]) + k2*(q[X2i]-q[XMi]) + b*(q[dX1i]-q[dXMi])+ b*(q[dX2i]-q[dXMi]))])

nS = 6+1 # one for the constant term
nA = 2

kf1, kf2 = 1., 1. #spring stiffness for each agent

sys_Ac = np.array([
    [0, 0, 0, 1., 0, 0],
    [0, 0, 0, 0, 1., 0],
    [0, 0, 0, 0, 0, 1.],
    [-k1/m1, 0, k1/m1, -b/m1, 0, b/m1],
    [0, -k2/m2, k2/m2, 0, -b/m2, b/m2],
    [k1/M, k2/M, (-k1-k2)/M, b/M, b/M, -2*b/M],
     ])

sys_Bc = np.array([
    [0, 0],
    [0, 0],
    [0, 0],
    [1./m1, 0],
    [0, 1./m2],
    [0, 0]
    ])
# sys_C = 1e0* np.array([
#     [0, 0, -1., 0, 0, 0,],
#     [0, 0, 0, 0, 0, -1.,],
#     [kf1, 0, -kf1, 0, 0, 0],
#     [0, kf2, -kf2, 0, 0, 0]
#     ])
# sys_Dr = 1e0* np.array([
#     [1, 0],
#     [0, 1],
#     [0, 0],
#     [0, 0]
#     ]) 

# U = [f1, f2]

######################
# Planner config
######################
Q_mat = 1e-3*np.eye(nS)
Q_mat[2,2] = 1e1#0
Q_mat[5,5] = 1e2#0

R_mat = 1e0* np.eye(nA)
N_mat = np.zeros((nS, nA))

dt_d = 0.005 #discretization period. 0.001<  <0.01
Th = 3. #0.2 #plannig horizon
replanning_period = 1.#0.2#dt_d


nsteps = int(Th/dt_d)

# Precomputed variables
sys_Ad = la.expm(sys_Ac*dt_d) #exp(Ac*Ts)
sys_Bd = mat_mul_series(
    sys_Ad-np.eye(len(sys_Ad)),
    la.inv(sys_Ac),
    sys_Bc) #inv(A)*(Ad-I) Bc  # should be all positive elements
Ax = np.pad(sys_Ad, ((0,1),(0,1)), mode='constant', constant_values=0.)


######################
# Reference config
######################
dt = dt_d #0.002
duration = 10. 
t = np.arange(0,duration, dt)

freq1 = np.array([0.1, 0.25, 0.55])
amp1 = 0.1/(freq1*2*np.pi)
ref, refdot = sumOfSines(t, freq1, amp1)

######################
# Simulation Loop
######################
# u = np.stack((u1, u2), 0)
u = np.zeros((2, len(t)))
q = np.zeros((6,len(t)))

gains = np.zeros((nA, nS, len(t)))

Ae = np.copy(Ax) #(nS,nS)
Be = np.concatenate(
            (
                -sys_Bd, 
                np.zeros((1, sys_Bd.shape[1]))
            ), axis=0) #(nS,nA)

for k in range(len(t[:-1])):
    
    # At each planning time, we form the affine version of Ad, Bd to achieve xd, vd:
    
    r0, rdot0 = ref[k], refdot[k]
    r_vec = np.array([r0, r0, r0, rdot0, rdot0, rdot0, 1.])
    e_k = r_vec - np.append(q[:,k], 0)
    
    # if rdot0:
    #     R_mat = 
    if k% int(replanning_period/dt)==0:
        i_Pti = 0 #index to the current Pti (to compute gain)
        # Plan controller gain
        
        # Form A,B in error coordinates
        Ar = np.zeros_like(Ae)
        Ar[:3, 0] = 1.; Ar[3:6, 5] = 1.; Ar[6, 6] = 1.
        Ar[:3,5] = dt_d
        
        AeV = mat_mul_series(Ar-Ax, r_vec) 
        
        Ae[:,6] = AeV
        
    
    
        # Compute all P_k backwards from the horizon, then calculate u_k
        # P_{k-1} = A^T P_k A -(A^T P_k B +N) (R+B^T P_k B)^{-1} (B^T P_k A +N^T) +Q
        # terminal condition: P_N = Q

        # Takes 0.025s to plan for 500 steps
        Pti = Q_mat
        Pvec = []
        for ti in range(nsteps, 0, -1):
            Pti = mat_mul_series(Ae.T, Pti, Ae) - \
                mat_mul_series(
                    mat_mul_series(Ae.T, Pti, Be) +N_mat,
                    la.inv(
                        (R_mat + mat_mul_series(Be.T, Pti, Be))
                     ),
                    mat_mul_series( Be.T, Pti, Ae) +N_mat.T
                    ) +\
                Q_mat
            Pvec.append(Pti)
        Pvec.reverse()


    # F_k = (R +B^T P_{k+1} B)^{-1} (B^T P_{k+1} A +N^T)
    # x_k = [q- [xd,vd], [1] ]
    # u_k = -F_k x_k        
    Pti = Pvec[i_Pti]
    Kti = mat_mul_series(
        la.inv(R_mat +\
               mat_mul_series(Be.T, Pti, Be)
               ),
        mat_mul_series(Be.T, Pti, Ae) +N_mat.T
        )

    U_opt = -mat_mul_series(Kti, e_k)
    
    i_Pti +=1
    
    gains[...,k] = Kti
    u[:,k] = U_opt #Kp*e + Kd*edot
    # u[1,k] = 0. #@@@@
    q[:,k+1] = RK4Step(q[:,k], sys_MSMSM, dt, u=u[:,k], t=t)

x1 = q[0,:]
x2 = q[1,:]
xm = q[2,:]
f1 = u[0,:]
f2 = u[1,:]
fn1 = k1*(xm-x1)
fn2 = -k2*(xm-x2)
e_sig = ref-xm

######################
# Analysis
######################

plt.figure(1)
plt.plot(t, x1, label = r'x_1')
plt.plot(t, x2, label = r'x_2')
plt.plot(t, xm, label = r'x_m')
plt.plot(t, ref, 'k--', label = r'ref')
plt.plot(t, e_sig, 'c:', label='e')
plt.legend()

fig2, ax2 = plt.subplots(2,1, sharey= True, sharex = True)
ax2[0].plot(t, f1, 'r', label = 'applied force')
ax2[0].plot(t, -fn1, 'g--', label = '(inverted) normal force')
ax2[0].plot(t, fn1+f1, 'b:', label = 'diff normal and applied')
ax2[0].axhline(0, lw=0.3, color='k')
ax2[0].legend()

ax2[1].plot(t, f2, 'r')
ax2[1].plot(t, -fn2, 'g--')
ax2[1].plot(t, fn2+f2, 'b:')
# ax2[1].set_ylim([-3,3])
ax2[1].axhline(0, lw=0.3, color='k')

fig3, ax3 = plt.subplots(2,1, sharey= True, sharex = True)
ax3[0].plot(t, f1, 'r--', label = 'applied force')
ax3[0].plot(t, gains[0, 2], 'g', label = 'p gain')
ax3[0].plot(t, gains[0, 5], 'b', label = 'd gain')
# ax3[0].plot(t, -fn1, 'g', label = '(inverted) normal force')
# ax3[0].plot(t, fn1+f1, 'b:', label = 'diff normal and applied')
ax3[0].axhline(0, lw=0.3, color='k')
ax3[0].legend()

ax3[1].plot(t, f2, 'r--', label = 'applied force')
ax3[1].plot(t, gains[1, 2], 'g', label = 'p gain')
ax3[1].plot(t, gains[1, 5], 'b', label = 'd gain')
ax3[1].axhline(0, lw=0.3, color='k')
ax3[1].legend()

plt.show()