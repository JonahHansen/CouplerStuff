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

#Fake Data:
F_0 = 1.5
coh_phase = np.pi/6
vis = 0.5
true_params = (F_0,vis,coh_phase)

#List of wavelength channels, with spacing 20nm.
bandpass = 10e-9
start_wavelength = 600e-9
end_wavelength = 750e-9
wavelengths = np.arange(start_wavelength,end_wavelength,bandpass)[:-1] + 0.5*bandpass

#SNRs for fringe tracking and science
SNR_fringe = 10
SNR_science =10

#Maximum and rms error expected in delay space
error_rms = 2e-5
error_max = 5e-5

#List of trial delays to scan
trial_delays = np.linspace(-error_max,error_max,2000)

fix_delay=0
vis_array=[]

#Number of integrations
n_iter = 1000

#Simulate a loop of fringe tracking and science
for j in range(n_iter):

    time_start = time.time()

    #Generate a random delay based on the error rms
    bad_delay = 2*error_rms*np.random.random_sample() - error_rms

    #Calculate the output complex coherence
    gamma = ff.cal_coherence(bad_delay,0,SNR_fringe,wavelengths,bandpass,true_params)

    #Estimate the group delay
    fix_delay = ff.find_delay(gamma,trial_delays,wavelengths)

    #Adjust the delay and calculate the new coherence
    new_gamma = ff.cal_coherence(bad_delay,fix_delay,SNR_science,wavelengths,bandpass,true_params)

    #Estimate the visibility based on the corrected coherence and append to list
    vis_array.append(np.mean(np.abs(new_gamma)**2))

    #Print time it takes to perform fringe tracking and science
    time_end = time.time()
    print(f"Number {j}, Time elapsed = {1000*(time_end-time_start)} ms")

#Print the average of the estimated visibilities
print(np.mean(vis_array))

#Plot the estimated visibilities as a function of time
plt.plot(0.01*np.arange(len(vis_array)),vis_array,marker=".",ls="")
plt.xlabel("Time (s)")
plt.ylabel("V^2")
#plt.show()
