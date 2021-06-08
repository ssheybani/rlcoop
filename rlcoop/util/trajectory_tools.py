import numpy as np


def rms(*args):
    # args is an arbitrary number of vectors such as error (across time), force1, force2.
    # For the case of one vector (signal), it returns the RMS, i.e. the root of signal power.
    # For the case of several vectors, it returns sqrt(average power of the signals).
    # Works as: output= sqrt( mean_over_elements( ( {V_1}^2 +{V_2}^2 +...+ {V_n}^2 ) /n ) )
    v2sum =0*args[0]
    n_v = 0; # n_v keeps the number of vectors passed.
    for vec in args:
        v2sum += vec**2
        n_v +=1
    return np.sqrt(np.mean(v2sum/n_v))


class Trajectory():
    # A traj_spec is the specification of a trajectory.
    # A trajectory is a combination of several sinusoids, each described by the
    # three factors: amplitude, frequency (Hz) and phase. 
    # 
    # traj = \Sigma_k A_k cos(2\pi f_k t +\phi_k)
    
    def __init__(self, tstep, seed_=None):
        self.tstep = tstep
        if seed_ is not None:
            np.random.seed(seed_)
#         self.duration = duration            
            
    
    def generate_traj_spec(self, amps, max_amp, traj_max_f, amp_std=0.03):
        # Generate a traj_spec randomly
        # amps is a vector of numbers within [0,1], which "roughly" determines 
            # the relative amplitude of the sinosid components.

        # Returns a list of lists, each sublist being the specifications of a sinusoid.

        
        amps = [0]+amps
        frq_bins, fstep = np.linspace(0, traj_max_f, len(amps), retstep=True)

        traj_spec = []
        for i in range(1, len(amps)):
            frq = fstep*np.random.rand() +frq_bins[i-1]
            amp = max_amp* (amps[i]+amp_std*(np.random.rand()-0.5))#(1-i/(3*n_snsd))* np.random.rand()
            phi = 2*np.pi*np.random.rand()
            traj_spec.append([amp, frq, phi])
        
        return traj_spec
    
    # @staticmethod
    def _traj_derivative(self, traj_spec):
        # Calculate the derivative of a traj_spec. Return the result in the form of another traj_spec.
                
        traj_spec_d=[]
        for snsd in traj_spec:
            traj_spec_d.append([2*np.pi*snsd[1]*snsd[0], snsd[1], snsd[2]+np.pi/2])
        return traj_spec_d
    

    # @staticmethod
    def generate(self, traj_spec, duration):#, normalize=True):
        # traj_specs is a list of traj_spec. 
        # Each traj_spec is a list of sinosoid descriptors. 
        # Each descriptor comprises 3 scalars: amplitude, frequency, phase.
        # Hence, traj_specs is a 3 dimensional list.
        
        n = np.arange(0, duration, self.tstep)
        x =np.zeros_like(n)
        
    # Check traj_spec
        for snsd in traj_spec:
            if len(snsd) !=3:
                raise ValueError 
        
        # Generate the trajectory
        sum_amp =0
        for i in range(len(traj_spec)):
#             x += traj_spec[i][0]* np.sin(2*np.pi*n *traj_spec[i][1] +traj_spec[i][2]) #@@@@@@@@@ Feb 2020
            x += traj_spec[i][0]* np.cos(2*np.pi*n *traj_spec[i][1] +traj_spec[i][2])
            sum_amp += traj_spec[i][0]
        
        return n, x

    
    # @classmethod
    def _weakest_net_force(self, r_spec, obj_mass, obj_fric, duration):
                
        # Check traj_spec
        for snsd in r_spec:
            if len(snsd) !=3:
                raise ValueError
                
        r_d_spec = self._traj_derivative(r_spec)
        n, r_d = self.generate(r_d_spec, duration)
        r_dd_spec = self._traj_derivative(r_d_spec)
        _, r_dd = self.generate(r_dd_spec, duration)
        
        f_net = obj_mass*r_dd + obj_fric*r_d

        return f_net
    
    # @staticmethod
    def _get_traj_rms(self, r_spec, obj_mass, obj_fric):
        net_f = self._weakest_net_force(r_spec, obj_mass, obj_fric, 10.)
        f_rms = rms(net_f)
        return f_rms
    
    
    # @staticmethod
    def generate_random(self, duration, n_traj=1, max_amp=0.45, traj_max_f=0.5, rel_amps=None, fixed_effort=True, obj_mass=0.5, obj_fric=1, n_deriv=1, ret_specs=True):
        # For time step, self.tstep is used.
        # n_traj: the number of trajectories generated.
        # max_amp: maximum amplitude of the trajectory.
        # rel_amps: a vector of numbers within [0,1], which "roughly" determines 
            # the relative amplitude of the sinosid components.
        # fixed_effort: If true, normalizes the trajectory such that it requires an f_rms of 1 to track.
        # n_deriv: the number of the derivatives of the trajectory to be returned as additional time series.
        # ret_specs: If true, returns the specifications of the sinusoidal conponents used to generate the time series. 



        traj_specs = []; trajs =[]
        
        if rel_amps is None:
            rel_amps = [1., 1./2, 1./3, 1./4, 1./5]; 
#             rel_amps = [0.3, 0.6, 0.6, 0.4, 0.4]; 
        
        amp_sum = np.sum(rel_amps)
        rel_amps = [item/amp_sum for item in rel_amps]


        for i in range(n_traj):
            
            traj_spec = self.generate_traj_spec(rel_amps, max_amp, traj_max_f, amp_std=0.03)
            # Determine the scaling factor of (the amplitude of) the traj_spec so that they require the same energy
            if fixed_effort is True:
                sc_fac = self._get_traj_rms(traj_spec, obj_mass, obj_fric)
                for j in range(len(traj_spec)):
                    traj_spec[j][0] = traj_spec[j][0]/sc_fac
            else:
                # Make sure the amplitude of the whole signal remains less than max_amp
                sum_amp = 0.
                for j in range(len(traj_spec)):
                    sum_amp += traj_spec[j][0]
                for j in range(len(traj_spec)):
                    traj_spec[j][0]*= max_amp/sum_amp
            
            traj_specs.append(traj_spec) 
            time1, traj = self.generate(traj_spec, duration)
            
            traj_dx_spec = traj_spec
            traj = np.expand_dims(traj, axis=0)
            for j in range(n_deriv):
                traj_dx_spec = self._traj_derivative(traj_dx_spec)
                _, traj_dx = self.generate(traj_dx_spec, duration)
                traj = np.concatenate((traj, traj_dx[np.newaxis,:]), axis=0)
            
            trajs.append(traj)
        
        if n_traj==1:
            trajs = trajs[0]
        if ret_specs is True:
            return time1, trajs, traj_specs
        else:
            return time1, trajs
        
    
    def generate_sysid_sine_traj(freqs, duration, max_amp=0.45, fixed_effort=True, sys_mass=2.5, sys_fric=1, n_deriv=2, ret_specs=True):
        # For time step, self.tstep is used.
        # n_traj: the number of trajectories generated.
        # max_amp: maximum amplitude of the trajectory.
        # fixed_effort: If true, normalizes the trajectory such that it requires an f_rms of 1 to track.
        # n_deriv: the number of the derivatives of the trajectory to be returned as additional time series.
        # ret_specs: If true, returns the specifications of the sinusoidal conponents used to generate the time series. 


        freqs = np.asarray(freqs)
        traj_specs = []; traj =[] #shall be returned

        rel_amps = 1./freqs

        rel_amp_sum = np.sum(rel_amps)
        amps = [item*max_amp/rel_amp_sum for item in rel_amps]

        traj_spec = [[freqs[i], amps[i], 2*np.pi*np.random.rand()] for i in range(amps)]

        # Determine the scaling factor of (the amplitude of) the traj_spec so that they require the same energy
        if fixed_effort is True:
            sc_fac = self._get_traj_rms(traj_spec, sys_mass, sys_fric)
            for j in range(len(traj_spec)):
                traj_spec[j][0] = traj_spec[j][0]/sc_fac
        else:
            # Make sure the amplitude of the whole signal remains less than max_amp
            sum_amp = 0.
            for j in range(len(traj_spec)):
                sum_amp += traj_spec[j][0]
            for j in range(len(traj_spec)):
                traj_spec[j][0]*= max_amp/sum_amp

        traj_specs.append(traj_spec) 
        time1, traj = self.generate(traj_spec, duration)

        traj_dx_spec = traj_spec
        traj = np.expand_dims(traj, axis=0)
        for j in range(n_deriv):
            traj_dx_spec = self._traj_derivative(traj_dx_spec)
            _, traj_dx = self.generate(traj_dx_spec, duration)
            traj = np.concatenate((traj, traj_dx[np.newaxis,:]), axis=0)


        if ret_specs is True:
            return time1, traj, traj_specs
        else:
            return time1, traj

    
    
class StepFunction():
    
    def __init__(self, tstep, seed_=None):
        self.tstep = tstep
        if seed_ is not None:
            np.random.seed(seed_)
        
    def generate(self, duration, max_amp=0.45, t_pulse=1., n_deriv=1):
        time1 = np.arange(0, duration, self.tstep)
        traj =np.zeros((1,time1.shape[0])); traj[0,int(t_pulse/self.tstep):] = max_amp
        traj_dx = np.zeros_like(traj); traj_dx[0, int(t_pulse/self.tstep)]=max_amp
        if n_deriv==1:
            traj = np.concatenate((traj, traj_dx), axis=0)
        elif n_deriv==2:
            traj_ddx = np.zeros_like(traj); 
            traj_ddx[0, int(t_pulse/self.tstep)]=max_amp;
            traj_ddx[0, int(t_pulse/self.tstep)+1]=-max_amp
            traj = np.concatenate((traj, traj_dx, traj_ddx), axis=0)
        else:
            raise ValueError
            
        return time1, traj
    
    def generate_random(self, duration, max_amp=0.45, n_deriv=1):
        t_pulse = np.random.rand()*duration/2
        return self.generate(duration, max_amp=max_amp, t_pulse=t_pulse, n_deriv=n_deriv)
        
        