import numpy as np
import matplotlib.pyplot as plt
import fringe_functions as ff
import time

"""
Simluates the end output of the interferometer (using intensity equation)
with an incorrect delay, tries to calculate the group delay from phasors,
applies the delay correction, and then calculates the estimated visibility^2.

It performs this in a loop with changing incorrect delays, and then plots the
changing visibilities. Also calculates the time taken for each loop of fringe
tracking and science.

"""
#R band constants:
R_flux = 2.19e-11 #W/m^2/nm
R_bandpass = 133 #nm
nu = 4.28e14
h = 6.62607015e-34 #Js

#Telescope details:
D = 0.1#m
int_time = 0.01#s

#Fake Data:
Rmag_star = 5
f_star = R_flux*10**(-0.4*Rmag_star)*R_bandpass #W/m^2
E_star = np.pi*(D/2)**2*int_time*f_star #J
F_0 = E_star/(h*nu) #(photons per telescope per integration)

coh_phase = np.pi/6
vis = 0.5
true_params = (F_0,vis,coh_phase)

#List of wavelength channels, with spacing 20nm.
bandpass = 15e-9
start_wavelength = 600e-9
end_wavelength = 750e-9
wavelengths = np.arange(start_wavelength,end_wavelength,bandpass)[:-1] + 0.5*bandpass

#Throughput (tricoupler with throughput eta = 0.5)
throughput = 1/3*0.5*1/len(wavelengths)

#Maximum and rms error expected in delay space
error_rms = 2e-5
error_max = 5e-5

#List of trial delays to scan
trial_delays = np.linspace(-error_max,error_max,1000)

fix_delay=0
vis_array=[]

#Number of integrations
n_iter = 500

#Calc Bias in visibility
for j in range(n_iter):

    #Generate a random delay based on the error rms
    bad_delay = 2*error_rms*np.random.random_sample() - error_rms

    #Calculate the output complex coherence
    gamma = ff.cal_coherence(bad_delay,0,throughput,wavelengths,bandpass,(F_0,0,np.pi/3))

    #Estimate the visibility based on the corrected coherence and append to list
    vis_array.append(np.mean(np.abs(gamma)**2))

bias_vis = np.median(vis_array)

vis_array=[]

#Simulate a loop of fringe tracking and science
for j in range(n_iter):

    time_start = time.time()

    #Generate a random delay based on the error rms
    bad_delay = 2*error_rms*np.random.random_sample() - error_rms

    #Calculate the output complex coherence
    gamma = ff.cal_coherence(bad_delay,0,throughput,wavelengths,bandpass,true_params)

    #Estimate the group delay
    fix_delay = ff.find_delay(gamma,trial_delays,wavelengths)

    #Adjust the delay and calculate the new coherence
    new_gamma = gamma/np.sinc(fix_delay*bandpass/wavelengths**2)*np.exp(-1j*2*np.pi*fix_delay/wavelengths)

    #Estimate the visibility based on the corrected coherence and append to list
    vis_array.append(np.mean(np.abs(new_gamma)**2)-bias_vis)

    #Print time it takes to perform fringe tracking and science
    time_end = time.time()
    print(f"Number {j}, Time elapsed = {1000*(time_end-time_start)} ms")

#Print the average of the estimated visibilities
print(np.median(vis_array))

#Plot the estimated visibilities as a function of time
plt.plot(0.01*np.arange(len(vis_array)),vis_array,marker=".",ls="")
plt.xlabel("Time (s)")
plt.ylabel("V^2")
#plt.show()
