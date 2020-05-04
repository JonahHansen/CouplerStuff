import numpy as np
import matplotlib.pyplot as plt
import fringe_functions as ff

"""
Simluates the end output of the interferometer (using intensity equation)
with an incorrect delay, tries to calculate the group delay from phasors,
applies the delay correction, and then calculates the estimated visibility^2
"""

#List of wavelength channels, with spacing 20nm.
bandpass = 15e-9
start_wavelength = 600e-9
end_wavelength = 750e-9
wavelengths = np.arange(start_wavelength,end_wavelength,bandpass)[:-1] + 0.5*bandpass

#Throughput (tricoupler with instrumental throughput eta)
eta = 0.5
throughput = 1/3*eta*1/len(wavelengths)

#R band constants:
R_flux = 2.19e-11 #W/m^2/nm
R_bandpass = 133 #nm
nu = 4.28e14
h = 6.62607015e-34 #Js

#Turbulence Data
seeing = 1 #arcsec
r0 = 0.98*start_wavelength*180*3600/np.pi/seeing #m
v = 20 #m/s
t0 = 0.31*(r0/v) #s

#Telescope details:
D = 0.1#m
int_time = 1.6*t0

#Fake Data:
Rmag_star = 5
f_star = R_flux*10**(-0.4*Rmag_star)*R_bandpass #W/m^2
E_star = np.pi*(D/2)**2*int_time*f_star #J
F_0 = E_star/(h*nu)*throughput #(photons per pixel per integration)

print(f"Number of photons per pixel: {F_0}")

coh_phase = np.pi/6
vis = 0.5
true_params = (F_0,vis,coh_phase)

#Delay to try and recover (pretend it's caused by the atmosphere)
bad_delay = -1.56474575e-5

#Calculate Vis bias
fluxes = ff.cal_tri_output(bad_delay,wavelengths,bandpass,(F_0,0,np.pi/5))
gamma_bias = (3*fluxes[0] + np.sqrt(3)*1j*(fluxes[2]-fluxes[1]))/np.sum(fluxes,axis=0)-1

#Estimate the visibility based on the corrected coherence and append to list
vis_bias = np.mean(np.abs(gamma_bias)**2)

#Find complex coherence
fluxes = ff.cal_tri_output(bad_delay,wavelengths,bandpass,true_params)
gamma = (3*fluxes[0] + np.sqrt(3)*1j*(fluxes[2]-fluxes[1]))/np.sum(fluxes,axis=0)-1

#List of trial delays to scan
Num_delays = 200
scale = 0.1
wavenumber_bandpass = 1/start_wavelength - 1/end_wavelength
trial_delays = scale*np.arange(-Num_delays/2+1,Num_delays/2)/wavenumber_bandpass

#Estimate the delay through phasor rotation (and plot it)
delay_envelope = ff.group_delay_envelope(gamma,trial_delays,wavelengths,plot=True)
fix_delay = ff.find_delay(delay_envelope,trial_delays)

print(f"Delay estimate = {fix_delay}")
print(f"Off by: {np.abs(fix_delay)-np.abs(bad_delay)}")

#Adjust the delay and calculate the new coherence.
#NOTE: MAY NEED TO FIX THIS!!!!
new_gamma = gamma/np.sinc(fix_delay*bandpass/wavelengths**2)*np.exp(1j*2*np.pi*fix_delay/wavelengths)
print(f"Visibility^2 estimate = {np.mean(np.abs(new_gamma)**2) - vis_bias}")
