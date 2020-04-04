import numpy as np
import matplotlib.pyplot as plt
import tricoupler as tri
import fringe_functions as ff

"""
Simluates the end output of the interferometer (using the tricoupler matrix)
with an incorrect delay, then tries to calculate the group delay from phasors.
"""

#Effective index difference from RSOFT
del_n = 1.50551 - 1.50494
#Average refractive index of waveguide
n_0 = 1.54

#List of wavelength channels, with spacing 20nm.
bandpass = 10e-9
start_wavelength = 600e-9
end_wavelength = 750e-9
wavelengths = np.arange(start_wavelength,end_wavelength,bandpass)[:-1] + 0.5*bandpass

n = len(wavelengths) #Number of wavelengths

#Delay to try and recover (pretend it's caused by the atmosphere)
bad_delay = 1.5e-5
#Signal to noise ratio in order to add noise to the intensities
SNR = 100000

#Create a list of input vectors
b0 = 1/np.sqrt(2)*np.array([np.ones(n),np.zeros(n),np.exp(1j*2*np.pi/wavelengths*bad_delay)])

#Calculate the intensity at the output of the coupler
fluxes = np.abs(tri.calc_bz(b0,del_n,n_0))**2

#Add noise based on the SNR (sigma = intensity/SNR)
for i in range(3):
    fluxes[i] = np.random.normal(fluxes[i],np.abs(fluxes[i]/SNR))

#Calculate the complex coherence
gamma = (3*fluxes[0] + np.sqrt(3)*1j*(fluxes[2]-fluxes[1]))/np.sum(fluxes,axis=0)-1

#List of trial delays to scan
trial_delays = np.linspace(-5e-5,5e-5,10000)

#Estimate the delay through phasor rotation (and plot it)
fix_delay = ff.find_delay(gamma,trial_delays,wavelengths,plot=True)
print(f"Delay estimate = {fix_delay}")
print(f"Off by: {np.abs(fix_delay)-np.abs(bad_delay)}")
