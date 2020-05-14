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
throughput = 1/3*eta*1/len(wavelengths)

#Turbulence Data
seeing = 1 #arcsec
r0 = 0.98*start_wavelength*180*3600/np.pi/seeing # Fried parameter (m)
v = 20 # Wind speed (m/s)
t0 = 0.31*(r0/v) #Coherent time (s)

#Telescope details:
B = 20 #Baseline (m)
D = 0.09 #Aperture (m)

#Integration times
coh_int_time = 1.6*t0 #s
incoh_int_time = 30*t0 #s

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

#Number of integrations
n_iter = 2000

#Number of wavefront "cells" between the two apertures
num_r0s = int(np.ceil(B/r0))

#Calculate the number of cells required for the turbulence sim
num_cells = ff.nearest_two_power(n_iter+num_r0s+1)

####################### Find Visibility Bias ##################################

#Atmospheric wavefront (will move across the interferometer)
atm_phases = ff.km1d(num_cells)

vis_array_num=[]
vis_array_den=[]

print("Begin Bias")

#Calc Bias in visibility
for j in range(n_iter):

    #Calculate the phase error difference between the two apertures
    bad_phase = (atm_phases[j]-atm_phases[j+num_r0s])
    #Convert to an OPD, based on the middle wavelength???
    bad_delay = bad_phase*(0.5*(end_wavelength-start_wavelength))/(2*np.pi)

    #Calculate the output complex coherence
    fluxes = ff.cal_tri_output(bad_delay,wavelengths,bandpass,F_0,0)
    gamma_bias_num = 2*fluxes[0] - (fluxes[2]+fluxes[1]) + np.sqrt(3)*1j*(fluxes[2]-fluxes[1])
    gamma_bias_den = np.sum(fluxes,axis=0)

    #Estimate the visibility based on the corrected coherence and append to list
    #Takes the mean over wavelengths
    vis_array_num.append(np.mean(np.abs(gamma_bias_num)**2))
    vis_array_den.append(np.mean(np.abs(gamma_bias_den)**2))

#Divide the numerator of the vis by the denominator.
vis_bias = np.mean(vis_array_num)/np.mean(vis_array_den)

print("End Bias")

###################### Science and Delay Loop #################################

#Atmospheric wavefront (will move across the interferometer)
atm_phases = ff.km1d(num_cells)

#Setup arrays
vis_array_num=[] #Numerator of the visibility
vis_array_den=[] #Denominator of the visibility
bad_delay_array=[] #Array of atmospheric errors
fix_delay_array=[] #Position of the delay line
error_delay_array=[] #Residuals
ave_delay_envelope = np.zeros(len(trial_delays))

#Start delay line at 0
fix_delay = 0

#Gain of the servo loop
gain = 1*a

#If using a moving rectangular average, initialise the array
rolling_average_array = np.zeros((int(incoh_int_time/coh_int_time),len(trial_delays)))

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
    fluxes = ff.cal_tri_output(eff_delay,wavelengths,bandpass,F_0,vis)
    gamma_num = 2*fluxes[0] - (fluxes[2]+fluxes[1]) + np.sqrt(3)*1j*(fluxes[2]-fluxes[1])
    gamma_den = np.sum(fluxes,axis=0)

    #Estimate the current delay envelope
    delay_envelope = ff.tri_group_delay_envelope(gamma_num,trial_delays,wavelengths)

    #Add to running average
    #If using the fading memory delay
    #ave_delay_envelope = a*delay_envelope + (1-a)*ave_delay_envelope
    #If using the rectangular moving average
    rolling_average_array = ff.shift_array(rolling_average_array,1,fill_value=delay_envelope)
    ave_delay_envelope = np.average(rolling_average_array,axis=0)

    #Find estimate the residual delay for adjustment
    adj_delay = ff.find_tri_delay(ave_delay_envelope,trial_delays)
    print(f"eff delay estimate = {adj_delay}")

    #How close was the estimate?
    print(f"Off by: {eff_delay-adj_delay}")

    #Adjust the delay line
    fix_delay += adj_delay*gain
    #Add to array
    fix_delay_array.append(fix_delay*1e6)

    #Add the residual delay errors to the array
    error_delay_array.append((fix_delay-bad_delay)*1e6)

    #Adjust the delay and calculate the new coherence????
    #new_gamma = gamma/np.sinc(adj_delay*bandpass/wavelengths**2)*np.exp(1j*2*np.pi*fix_delay/wavelengths)

    #Estimate the visibility based on the corrected coherence and append to list
    #Takes the mean over wavelengths
    vis_array_num.append(np.mean(np.abs(gamma_num)**2))
    vis_array_den.append(np.mean(np.abs(gamma_den)**2))

#Print the average of the estimated visibilities
print(np.mean(vis_array_num)/np.mean(vis_array_den)-vis_bias)

#Plot the estimated visibilities as a function of time
plt.figure(1)
plt.plot(vis_array_num/np.mean(vis_array_den)-vis_bias,marker=".",ls="",label="Data points",zorder=0)
plt.hlines(0.25,0,n_iter,label="True value",zorder=5)
plt.hlines(np.mean(vis_array_num)/np.mean(vis_array_den)-vis_bias,0,n_iter,color="r",label="Mean value",zorder=9)
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