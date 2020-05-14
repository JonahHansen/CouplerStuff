import numpy as np
import matplotlib.pyplot as plt
import fringe_functions as ff

"""
Simluates the end output of the interferometer (using intensity equation)
with an incorrect delay, tries to calculate the group delay from phasors,
applies the delay correction, and then calculates the estimated visibility^2
"""

#List of wavelength channels, with spacing 20nm (in m).
bandpass = 15e-9
start_wavelength = 600e-9
end_wavelength = 750e-9
wavelengths = np.arange(start_wavelength,end_wavelength,bandpass)[:-1] + 0.5*bandpass

#Throughput (tricoupler with instrumental throughput eta)
eta = 0.5
throughput = 1/3*eta*1/len(wavelengths)

#Turbulence Data
seeing = 1 #arcsec
r0 = 0.98*start_wavelength*180*3600/np.pi/seeing #m
v = 20 #m/s
t0 = 0.31*(r0/v) #s

#Telescope details:
D = 0.1#m
int_time = 1.6*t0 #s

#Star Flux and visibility
Rmag_star = 5
F_0 = ff.star_flux(Rmag_star,int_time,D,throughput)
vis = 0.5

#Delay to try and recover (pretend it's caused by the atmosphere)
bad_delay = -1.56474575e-7 #m

#Calculate Vis bias
fluxes = ff.cal_tri_output(bad_delay,wavelengths,bandpass,F_0,0)
gamma_bias = (3*fluxes[0] + np.sqrt(3)*1j*(fluxes[2]-fluxes[1]))/np.sum(fluxes,axis=0)-1

#Estimate the visibility based on the corrected coherence and append to list
vis_bias = np.mean(np.abs(gamma_bias)**2)

#Find complex coherence
fluxes = ff.cal_tri_output(bad_delay,wavelengths,bandpass,F_0,vis)
gamma = (3*fluxes[0] + np.sqrt(3)*1j*(fluxes[2]-fluxes[1]))/np.sum(fluxes,axis=0)-1

#List of trial delays to scan
Num_delays = 2000
scale = 0.005
wavenumber_bandpass = 1/start_wavelength - 1/end_wavelength
trial_delays = scale*np.arange(-Num_delays/2+1,Num_delays/2)/wavenumber_bandpass

#Estimate the delay through phasor rotation (and plot it)
delay_envelope = ff.tri_group_delay_envelope(gamma,trial_delays,wavelengths,plot=True)
fix_delay = ff.find_tri_delay(delay_envelope,trial_delays)

print(f"Delay estimate = {fix_delay}")
print(f"Off by: {np.abs(fix_delay)-np.abs(bad_delay)}")

#Adjust the delay and calculate the new coherence.
#NOTE: MAY NEED TO FIX THIS!!!!
new_gamma = gamma/np.sinc(fix_delay*bandpass/wavelengths**2)*np.exp(1j*2*np.pi*fix_delay/wavelengths)
print(f"Visibility^2 estimate = {np.mean(np.abs(new_gamma)**2) - vis_bias}")
