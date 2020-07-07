import numpy as np
import matplotlib.pyplot as plt
import fringe_functions as ff

"""
Simluates the end output of the interferometer (using intensity equation)
with an incorrect delay, tries to calculate the group delay from phasors,
applies the delay correction, and then calculates the estimated visibility^2
"""

#Set up interferometer
pyxis = ff.tri_interferometer(bandpass = 15e-9, #m
                              start_wavelength = 600e-9, #m
                              end_wavelength = 750e-9, #m
                              eta = 0.2,
                              seeing = 1, #arcsec
                              v = 20, #m/s
                              incoh_scaling = 30,
                              num_delays = 1000,
                              scale_delay = 0.005)

#Star Flux and visibility
Rmag_star = 0
F_0 = pyxis.star_flux(Rmag_star)
vis = 0.5

#Delay to try and recover (pretend it's caused by the atmosphere)
bad_delay = -1.56474575e-7 #m

#Calculate bias with visibility zero
gamma_bias = pyxis.calc_gamma(bad_delay,F_0,0)
vis_bias = np.mean(np.abs(gamma_bias)**2)

#Find complex coherence
gamma = pyxis.calc_gamma(bad_delay,F_0,vis)

#Estimate the delay through phasor rotation (and plot it)
delay_envelope = pyxis.calc_group_delay_envelope(gamma,plot=True)
fix_delay = pyxis.find_delay(delay_envelope)

print(f"Delay estimate = {fix_delay}")
print(f"Off by: {np.abs(fix_delay)-np.abs(bad_delay)}")

#Adjust the delay and calculate the new coherence.
#NOTE: MAY NEED TO FIX THIS!!!!
new_gamma = gamma/np.sinc(fix_delay*pyxis.bandpass/pyxis.wavelengths**2)*np.exp(1j*2*np.pi*fix_delay/pyxis.wavelengths)
print(f"Visibility^2 estimate = {np.mean(np.abs(new_gamma)**2) - vis_bias}")
