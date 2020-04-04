import numpy as np
import matplotlib.pyplot as plt
import fringe_functions as ff

"""
Simluates the end output of the interferometer (using intensity equation)
with an incorrect delay, tries to calculate the group delay from phasors,
applies the delay correction, and then calculates the estimated visibility^2 
"""

#Fake Data:
F_0 = 1.5
coh_phase = np.pi/6
vis = 0.5
true_params = (F_0,vis,coh_phase)

#List of wavelength channels, with spacing 20nm.
bandpass = 10e-9
start_wavelength = 600e-9
end_wavelength = 750e-9
wavelengths = np.arange(start_wavelength,end_wavelength,bandpass)[:-1] + 0.5*bandpass

#Delay to try and recover (pretend it's caused by the atmosphere)
bad_delay = 1.5e-5
#Signal to noise ratio in order to add noise to the intensities
SNR = 100000

#Find complex coherence
gamma = ff.cal_coherence(bad_delay,0,SNR,wavelengths,bandpass,true_params)

#List of trial delays to scan
trial_delays = np.linspace(-5e-5,5e-5,10000)

#Estimate the delay through phasor rotation (and plot it)
fix_delay = ff.find_delay(gamma,trial_delays,wavelengths,plot=True)
print(f"Delay estimate = {fix_delay}")
print(f"Off by: {np.abs(fix_delay)-np.abs(bad_delay)}")

#Calculate the new coherence by subtracting the estimated delay from the unknown
new_gamma = ff.cal_coherence(bad_delay,fix_delay,SNR,wavelengths,bandpass,true_params)
print(f"Visibility^2 estimate = {np.mean(np.abs(new_gamma)**2)}")
