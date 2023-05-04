import numpy as np
import matplotlib as mpl
from scipy.special import erf
import matplotlib.pyplot as plt
from scipy.signal import convolve as convolve
# params = {'axes.labelsize': 30,'axes.titlesize':30, 'font.size': 30, 'legend.fontsize': 30, 'xtick.labelsize': 30, 'ytick.labelsize': 30}
# mpl.rcParams.update(params)

def Gaussian(t,A,mu,sigma):
    return A/np.sqrt(2*np.pi*sigma**2)*np.exp(-(t-mu)**2/(2.*sigma**2))
def Decay(t,tau,t0):
    ''' Decay expoential and step function '''
    return 1./tau*np.exp(-t/tau) * 0.5*(np.sign(t-t0)+1.0)
def ModifiedGaussian(t,A,mu,sigma,tau):
        ''' Modified Gaussian function, meaning the result of convolving a gaussian with an expoential decay that starts at t=0'''
        x = 1./(2.*tau) * np.exp(.5*(sigma/tau)**2) * np.exp(- (t-mu)/tau)
        s = A*x*( 1. + erf(   (t-mu-sigma**2/tau)/np.sqrt(2*sigma**2)   ) )
        return s

### Input signal, response function, analytic solution
A,mu,sigma,tau,t0 = 1,0,2/2.344,2,0  # Choose some parameters for decay and gaussian
t = np.linspace(-10,10,1000)         # Time domain of signal
t_response = np.linspace(-5,10,1000)# Time domain of response function

### Populate input, response, and analyitic results
s = Gaussian(t,A,mu,sigma)
r = Decay(t_response,tau,t0)
m = ModifiedGaussian(t,A,mu,sigma,tau)

### Convolve
m_full = convolve(s,r,mode='full')
m_same = convolve(s,r,mode='same')
# m_valid = convolve(s,r,mode='valid')
### Define time of convolved data
t_full = np.linspace(t[0]+t_response[0],t[-1]+t_response[-1],len(m_full))
t_same = t
# t_valid = t

### Normalize the discrete convolutions
m_full /= np.trapz(m_full,x=t_full)
m_same /= np.trapz(m_same,x=t_same)
# m_valid /= np.trapz(m_valid,x=t_valid)

### Plot the input, repsonse function, and analytic result
f1,(ax1,ax2,ax3) = plt.subplots(nrows=3,ncols=1,num='Analytic')
ax1.plot(t,s,label='Input'),ax1.set_xlabel('Time'),ax1.set_ylabel('Signal'),ax1.legend()
ax2.plot(t_response,r,label='Response'),ax2.set_xlabel('Time'),ax2.set_ylabel('Signal'),ax2.legend()
ax3.plot(t_response,m,label='Output'),ax3.set_xlabel('Time'),ax3.set_ylabel('Signal'),ax3.legend()

### Plot the discrete  convolution against analytic
f2,ax4 = plt.subplots(nrows=1)
ax4.scatter(t_same[::2],m_same[::2],label='Discrete Convolution (Same)')
ax4.scatter(t_full[::2],m_full[::2],label='Discrete Convolution (Full)',facecolors='none',edgecolors='k')
# ax4.scatter(t_valid[::2],m_valid[::2],label='Discrete Convolution (Valid)',facecolors='none',edgecolors='r')
ax4.plot(t,m,label='Analytic Solution'),ax4.set_xlabel('Time'),ax4.set_ylabel('Signal'),ax4.legend()
plt.show()