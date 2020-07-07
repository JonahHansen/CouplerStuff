import numpy as np
import matplotlib.pyplot as plt
import fringe_functions as ff

"""
Simluates the end output of the interferometer (using intensity equation)
with an incorrect delay, tries to calculate the group delay from phasors,
applies the delay correction, and then calculates the estimated visibility^2
"""

#Set up interferometer
pyxis = ff.AC_interferometer(bandpass = 15e-9, #m
                              start_wavelength = 600e-9, #m
                              end_wavelength = 750e-9, #m
                              eta = 0.15,
                              seeing = 1, #arcsec
                              v = 20, #m/s
                              incoh_scaling = 30,
                              num_delays = 1000,
                              scale_delay = 0.005,
                              disp_length = 1e-3, #m
                              disp_lam_0 = 675e-9) #m


#Plot intensity against wavelength fro two different delays
def plot_intensity(vis,delay1,delay2):
    I1 = pyxis.calc_output(delay1,F_0,vis)
    I2 = pyxis.calc_output(delay2,F_0,vis)
    plt.plot(I1[0],'r-')
    plt.plot(I2[0],'b-')
    plt.show()

#Plot likelihood against delay, for different magnitudes and excess lengths
def plot_delay(bad_delay,vis,mag,length,fmt):
    pyxis.disp_length = length
    F_0 = pyxis.star_flux(mag)
    #Find output
    fluxes = pyxis.calc_output(bad_delay,F_0,vis)
    #Find coherence
    gamma_r = (fluxes[0] - fluxes[1])/(fluxes[0]+fluxes[1])
    #Chi^2 fit
    chi_2 = pyxis.calc_group_delay_envelope(gamma_r)
    print(pyxis.find_delay(chi_2))

    plt.plot(pyxis.trial_delays,np.exp(-chi_2**2/2),fmt)
    return

bad_delay = 8.34e-7
vis = 0.5
plot_delay(bad_delay,vis,0,0.002,'r-')
plot_delay(bad_delay,vis,6,0.002,'b-')
plt.vlines(bad_delay,0,1)
plt.show()
