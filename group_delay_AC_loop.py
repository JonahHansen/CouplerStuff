import numpy as np
import matplotlib.pyplot as plt
import fringe_functions as ff

"""
Simluates the end output of the interferometer (using intensity equation)
with an incorrect delay, tries to calculate the group delay from phasors,
applies the delay correction, and then calculates the estimated visibility^2.

It performs this in a loop with changing incorrect delays, and then plots the
changing visibilities. Also calculates the time taken for each loop of fringe
tracking and science.

"""

#List of wavelength channels, with spacing 20nm (in m).
bandpass = 15e-9
start_wavelength = 600e-9
end_wavelength = 750e-9
wavelengths = np.arange(start_wavelength,end_wavelength,bandpass)[:-1] + 0.5*bandpass

#Throughput (tricoupler with instrumental throughput eta)
eta = 0.15
throughput = 1/2*eta*1/len(wavelengths)

#Turbulence Data
seeing = 1 #arcsec
r0 = 0.98*start_wavelength*180*3600/np.pi/seeing # Fried parameter (m)
v = 20 # Wind speed (m/s)
t0 = 0.31*(r0/v) #Coherent time (s)

#Telescope details:
B = 20 #Baseline (m)
D = 0.09 #Aperture (m)

#Integration times
coh_int_time = 1.6*t0 # (in s)
incoh_int_time = 10*t0 # (in s)

#Scale factor for group envelope averaging:
a = 1 - np.exp(-coh_int_time/incoh_int_time)

#Star Flux and visibility
Rmag_star = 7
F_0 = ff.star_flux(Rmag_star,coh_int_time,D,throughput)
vis = 0.5

#List of trial delays to scan
Num_delays = 1000 #Number of delays
scale = 0.005 #How fine? Smaller = Finer
wavenumber_bandpass = 1/start_wavelength - 1/end_wavelength
trial_delays = scale*np.arange(-Num_delays/2+1,Num_delays/2)/wavenumber_bandpass

#Dispersion parameters
lam_0 = 675e-9 # (in m)
length = 1e-3 #Length of extended bit of glass (in m)

#Number of integrations
n_iter = 1000

#Number of wavefront "cells" between the two apertures
num_r0s = int(np.ceil(B/r0))

#Calculate the number of cells required for the turbulence sim
num_cells = ff.nearest_two_power(n_iter+num_r0s+1)

####################### Find Visibility Bias ##################################

#Atmospheric wavefront (will move across the interferometer)
atm_phases = ff.km1d(num_cells)

gamma_r_num_array=[]
gamma_r_den_array=[]

print("Begin Bias")

#Calc Bias in visibility
for j in range(n_iter):

    #Calculate the phase error difference between the two apertures
    bad_phase = (atm_phases[j]-atm_phases[j+num_r0s])
    #Convert to an OPD, based on the middle wavelength???
    bad_delay = bad_phase*(0.5*(end_wavelength-start_wavelength))/(2*np.pi)

    #Calculate the fluxes for each outputs
    fluxes = ff.cal_AC_output(bad_delay,wavelengths,bandpass,length,lam_0,F_0,0)

    #Add the square of the numerator and the denominator of the real part of
    #the coherence.
    #From V^2 ~ 2<(A-C)^2>/<A+C>^2 - Bias
    gamma_r_num_array.append((fluxes[0] - fluxes[1])**2)
    gamma_r_den_array.append((fluxes[0] + fluxes[1]))

#From V^2 ~ 2<(A-C)^2>/<A+C>^2 - bias
#Takes the mean over wavelengths
vis_bias = 2*np.mean(np.mean(gamma_r_num_array)/np.mean(gamma_r_den_array)**2)
print("End Bias")

###################### Science and Delay Loop #################################

#Atmospheric wavefront (will move across the interferometer)
atm_phases = ff.km1d(num_cells)

#Setup arrays
gamma_r_num_array=[] #Numerator squared of the real part of the coherence
gamma_r_den_array=[] #Denominator of the real part of the coherence
bad_delay_array=[] #Array of atmospheric errors
fix_delay_array=[] #Position of the delay line
error_delay_array=[] #Residuals
ave_delay_envelope = np.zeros(len(trial_delays))

#Start delay line at 0
fix_delay = 0

a = 0

#Gain of servo loop
gain = 0.0002

vis_array = []

#If using a moving rectangular average, initialise the array
rolling_average_array = np.zeros((int(incoh_int_time/coh_int_time),1,len(trial_delays)))

#Simulate a loop of fringe tracking and science
for j in range(n_iter):

    #Calculate the phase error difference between the two apertures
    bad_phase = (atm_phases[j]-atm_phases[j+num_r0s])
    #Convert to an OPD, based on the middle wavelength???
    bad_delay = bad_phase*(0.5*(end_wavelength-start_wavelength))/(2*np.pi)

    #Add the incorrect delay to the array
    bad_delay_array.append(bad_delay*1e6)

    #Calculate the effective (residual) delay
    eff_delay = bad_delay - fix_delay
    print(f"eff delay = {eff_delay}")

    #Calculate the output complex coherence
    fluxes = ff.cal_AC_output(eff_delay,wavelengths,bandpass,length,lam_0,F_0,vis)

    gamma_r = (fluxes[0] - fluxes[1])/(fluxes[0]+fluxes[1])
    gamma_r_num_array.append((fluxes[0] - fluxes[1])**2)
    gamma_r_den_array.append((fluxes[0]+fluxes[1]))

    #Estimate the squared visibility V^2 ~ 2<(A-C)^2>/<A+C>^2 - Bias
    #Takes the mean over wavelengths
    vis_est = 2*np.mean(np.mean(gamma_r_num_array,axis=0)/np.mean(gamma_r_den_array,axis=0)**2) - vis_bias
    print(vis_est)

    #Append the running average of visibilities
    vis_array.append(vis_est)

    trial_vis = [np.sqrt(vis_est)]

    trial_vis = [1]

    #Estimate the current delay envelope
    delay_envelope = ff.AC_group_delay_envelope(gamma_r,trial_delays,trial_vis,wavelengths,bandpass,length,lam_0)

    #Add to running average
    #If using the fading memory delay
    #ave_delay_envelope = a*delay_envelope + (1-a)*ave_delay_envelope
    #If using the rectangular moving average
    rolling_average_array = ff.shift_array(rolling_average_array,1,fill_value=delay_envelope)
    ave_delay_envelope = np.average(rolling_average_array,axis=0)

    #Find estimate the residual delay for adjustment
    adj_delay,fit_vis = ff.find_AC_delay_vis(ave_delay_envelope,trial_delays,trial_vis)
    print(f"eff delay estimate = {adj_delay}")

    #How close was the estimate
    print(f"Off by: {eff_delay-adj_delay}")

    #Adjust the delay line
    fix_delay += adj_delay*gain
    #Add to array
    fix_delay_array.append(fix_delay*1e6)

    #How close was the estimate?
    error_delay_array.append((fix_delay-bad_delay)*1e6)

#Print the average of the estimated visibilities
print(vis_est)
vis_array = np.array(vis_array)

#Plot the estimated visibilities as a function of time
plt.figure(1)
plt.plot(vis_array,marker=".",ls="",label="Data points",zorder=0)
plt.hlines(0.25,0,n_iter,label="True value",zorder=5)
plt.hlines(vis_est,0,n_iter,color="r",label="Mean value",zorder=9)
plt.xlabel("Time")
plt.ylabel("V^2")
plt.legend()

plt.figure(2)
plt.plot(bad_delay_array,label="Bad delay")
plt.plot(fix_delay_array,label="Delay estimate")
plt.plot(error_delay_array,label="Residuals")
plt.legend()
plt.xlabel("Time")
plt.ylabel("OPD (um)")
plt.show()
