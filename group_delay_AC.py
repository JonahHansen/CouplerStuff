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
throughput = 1/3*eta/len(wavelengths)

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
D = 0.1 #m
coh_int_time = 1.6*t0

#Fake Data:
Rmag_star = 6
f_star = R_flux*10**(-0.4*Rmag_star)*R_bandpass #W/m^2
E_star = np.pi*(D/2)**2*coh_int_time*f_star #J
F_0 = E_star/(h*nu)*throughput #(photons per pixel per integration)

print(f"Number of photons per pixel: {F_0}")

coh_phase = 0*np.pi/6
vis = 0.2343
true_params = (F_0,vis,coh_phase)

#Dispersion parameters
lam_0 = 675e-9
length = 10 #Length of extended bit of glass

#Delay to try and recover (pretend it's caused by the atmosphere)
bad_delay = 4.2346e-7
plt.vlines(bad_delay,0,14)

#Find real part of the complex coherence
fluxes = ff.cal_AC_output(bad_delay,wavelengths,bandpass,length,lam_0,true_params)
gamma_r = (fluxes[0] - fluxes[1])/(fluxes[0]+fluxes[1])

#List of trial delays to scan
trial_delays = np.linspace(-1e-6,1e-6,5000)
trial_vis = np.linspace(0,1,1000)

#Estimate the delay through phasor rotation (and plot it)
(vis_estimate,fix_delay) = ff.fit_vis_delay_AC(gamma_r,trial_delays,trial_vis,wavelengths,bandpass,length,lam_0)
print(f"Delay estimate = {fix_delay}")
print(f"Off by: {np.abs(fix_delay)-np.abs(bad_delay)}")
print(f"Visibility estimate = {vis_estimate}")
print(f"Off by: {np.abs(vis_estimate)-np.abs(vis)}")

"""
fix_delay = ff.find_delay_AC(gamma_r,trial_delays,trial_vis,wavelengths,bandpass,length,lam_0,True)
print(f"Delay estimate = {fix_delay}")
print(f"Off by: {np.abs(fix_delay)-np.abs(bad_delay)}")
"""

#Plot the fringes of the AC coupler
def print_fringes():
    delays = np.linspace(-1e-5,1e-5,1e4)
    x = []
    for d in delays:
        x.append(ff.cal_AC_output(d,wavelengths,bandpass,length,lam_0,true_params))

    xA = np.array(x)[:,0,:]
    xC = np.array(x)[:,1,:]

    plt.figure(5)
    plt.plot(delays,xA)
    plt.title("A Output")

    plt.figure(6)
    plt.plot(delays,xC)
    plt.title("C Output")

    plt.figure(7)
    plt.plot(delays,(xA-xC)/(xA+xC))
    plt.title("Re(Gamma)")

    plt.show()


#TO CALCULATE VISIBILITY, WOULD HAVE TO PUT INTO A LOOP AND AVERAGE A AND C:
#V ~ <(A-C)^2>/<A+C>^2 - Bias
