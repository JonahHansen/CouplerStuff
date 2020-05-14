import numpy as np
import matplotlib.pyplot as plt
import fringe_functions as ff

"""
Plots a fringe packet (with some glass dispersion) at a variety of wavelength
channels.
"""

#Fake Data:
F_0 = 1.5
vis = 0.5

#List of wavelength channels, with spacing 20nm.
bandpass = 10e-9
start_wavelength = 600e-9
end_wavelength = 750e-9
wavelengths = np.arange(start_wavelength,end_wavelength,bandpass)[:-1] + 0.5*bandpass

n = len(wavelengths) #Number of wavelengths

#Dispersion length of glass
width = 5e-3
central_wavelength = 0.5*(start_wavelength+end_wavelength)
#Dispersional phase shift of glass, with 650nm being the reference wavelength
glass_phase = ff.phaseshift_glass(wavelengths,width,central_wavelength)

#List of delays to plot against
delays = np.linspace(-1e-4,1e-4,100000)
colors = plt.cm.jet(np.linspace(0,1,n))
#For each channel,
for i in range(n):
    #Plot fringe pattern
    j = n-i-1
    plt.plot(delays*1000000,ff.fringe_flux(delays,wavelengths[j],bandpass=bandpass,
                         F_0=F_0,vis=vis,
                         disp_phase=glass_phase[j]),color=colors[j],alpha=0.2)
plt.xlabel("Delay (um)")
plt.show()
