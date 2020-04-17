import numpy as np
import matplotlib.pyplot as plt
import fringe_functions as ff

"""
Simluates the end output of the interferometer (using intensity equation)
with an incorrect delay, tries to calculate the group delay from phasors,
applies the delay correction, and then calculates the estimated visibility^2
"""

#R band constants:
R_flux = 2.19e-11 #W/m^2/nm
R_bandpass = 133 #nm
nu = 4.28e14
h = 6.62607015e-34 #Js

#Telescope details:
D = 0.1#m
int_time = 0.01#s

#Fake Data:
Rmag_star = 5
f_star = R_flux*10**(-0.4*Rmag_star)*R_bandpass #W/m^2
E_star = np.pi*(D/2)**2*int_time*f_star #J
F_0 = E_star/(h*nu) #(photons per telescope per integration)

coh_phase = np.pi/6
vis = 0.5
true_params = (F_0,vis,coh_phase)

#List of wavelength channels, with spacing 20nm.
bandpass = 15e-9
start_wavelength = 600e-9
end_wavelength = 750e-9
wavelengths = np.arange(start_wavelength,end_wavelength,bandpass)[:-1] + 0.5*bandpass

#Throughput (tricoupler with throughput eta = 0.5)
throughput = 1/3*0.5*1/len(wavelengths)
#Delay to try and recover (pretend it's caused by the atmosphere)
bad_delay = -1.5e-5

#Calculate Vis bias
gamma = ff.cal_coherence(bad_delay,0,throughput,wavelengths,bandpass,(F_0,0,np.pi/3))

#Estimate the visibility based on the corrected coherence and append to list
vis_bias = np.mean(np.abs(gamma)**2)


#Find complex coherence
gamma = ff.cal_coherence(bad_delay,0,throughput,wavelengths,bandpass,true_params)

#List of trial delays to scan
trial_delays = np.linspace(-5e-5,5e-5,10000)

#Estimate the delay through phasor rotation (and plot it)
fix_delay = ff.find_delay(gamma,trial_delays,wavelengths,plot=True)
print(f"Delay estimate = {fix_delay}")
print(f"Off by: {np.abs(fix_delay)-np.abs(bad_delay)}")

#Adjust the delay and calculate the new coherence
new_gamma = gamma/np.sinc(fix_delay*bandpass/wavelengths**2)*np.exp(1j*2*np.pi*fix_delay/wavelengths)
print(f"Visibility^2 estimate = {np.mean(np.abs(new_gamma)**2) - vis_bias}")
