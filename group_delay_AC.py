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
eta = 0.15
throughput = 1/2*eta/len(wavelengths)

#Turbulence Data
seeing = 1 #arcsec
r0 = 0.98*start_wavelength*180*3600/np.pi/seeing #m
v = 20 #m/s
t0 = 0.31*(r0/v) #s

#Telescope details:
D = 0.1 #m
coh_int_time = 1.6*t0

#Star Flux and visibility
Rmag_star = 7
F_0 = ff.star_flux(Rmag_star,coh_int_time,D,throughput)
vis = 0.2343
plt.hlines(vis,-10,10)

#Dispersion parameters
lam_0 = 675e-9 #in m
length = 1e-3 #Length of extended bit of glass

#Delay to try and recover (pretend it's caused by the atmosphere)
bad_delay = 4.2346e-7 #in m
plt.vlines(bad_delay*1e6,0,14)

#Find real part of the complex coherence
fluxes = ff.cal_AC_output(bad_delay,wavelengths,bandpass,length,lam_0,F_0,vis)
gamma_r = (fluxes[0] - fluxes[1])/(fluxes[0]+fluxes[1])

#List of trial delays to scan
trial_delays = np.linspace(-1e-6,1e-6,5000)
trial_vis = np.linspace(0,1,1000)

#Estimate the delay through chi^2 fitting (and plot it)
chi_2 = ff.AC_group_delay_envelope(gamma_r,trial_delays,trial_vis,wavelengths,bandpass,length,lam_0)
fix_delay,vis_estimate = ff.find_AC_delay_vis(chi_2,trial_delays,trial_vis)

print(f"Delay estimate = {fix_delay}")
print(f"Off by: {np.abs(fix_delay)-np.abs(bad_delay)}")
print(f"Visibility estimate = {vis_estimate}")
print(f"Off by: {np.abs(vis_estimate)-np.abs(vis)}")

plt.figure(1)
plt.imshow(chi_2,aspect="auto",origin="lower",extent=[min(trial_delays)*1e6,max(trial_delays)*1e6,min(trial_vis),max(trial_vis)])
plt.xlabel("Delay (microns)")
plt.ylabel("Visibility")
plt.show()

#Plot the fringes of the AC coupler
def print_fringes():
    delays = np.linspace(-1e-5,1e-5,1e4)
    x = []
    for d in delays:
        x.append(ff.cal_AC_output(d,wavelengths,bandpass,length,lam_0,F_0,vis))

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
