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

#Set up interferometer
pyxis = ff.tri_interferometer(baseline = 20, #m
                              diameter = 0.07, #m
                              bandpass = 15e-9, #m
                              start_wavelength = 600e-9, #m
                              end_wavelength = 750e-9, #m
                              eta = 0.15,
                              seeing = 1, #arcsec
                              v = 20, #m/s
                              incoh_scaling = 30,
                              num_delays = 1000,
                              scale_delay = 0.005)

#Star Flux and visibility
Rmag_star = 5
F_0 = pyxis.star_flux(Rmag_star)
vis = 0.5

#Number of integrations
n_iter = 1000

####################### Find Visibility Bias ##################################

#Atmospheric wavefront (will move across the interferometer)
atm_phases = pyxis.calc_atmosphere(n_iter)

vis_array_num=[]
vis_array_den=[]

print("Begin Bias")

#Calc Bias in visibility
for j in range(n_iter):

    #Calc bad delay
    bad_delay = pyxis.calc_bad_delay(atm_phases,j)

    #Calculate the output complex coherence
    gamma_bias_num,gamma_bias_den = pyxis.calc_gamma_numden(bad_delay,F_0,0)

    #Estimate the visibility based on the corrected coherence and append to list
    #Takes the mean over wavelengths
    vis_array_num.append(np.mean(np.abs(gamma_bias_num)**2))
    vis_array_den.append(np.mean(np.abs(gamma_bias_den)**2))

vis_bias = np.mean(vis_array_num)/np.mean(vis_array_den)

print("End Bias")

###################### Science and Delay Loop #################################

#Atmospheric wavefront (will move across the interferometer)
atm_phases = pyxis.calc_atmosphere(n_iter)

#Setup arrays
vis_array_num=[] #Numerator of the visibility
vis_array_den=[] #Denominator of the visibility
bad_delay_array=[] #Array of atmospheric errors
fix_delay_array=[] #Position of the delay line
error_delay_array=[] #Residuals
ave_delay_envelope = np.zeros(len(pyxis.trial_delays))

#Start delay line at 0
fix_delay = 0

#Gain of the servo loop
gain = pyxis.a

#If using a moving rectangular average, initialise the array
rolling_average_array = np.zeros((int(pyxis.incoh_int_time/pyxis.coh_int_time),len(pyxis.trial_delays)))

#Simulate a loop of fringe tracking and science
for j in range(n_iter):

    #Calc bad delay
    bad_delay = pyxis.calc_bad_delay(atm_phases,j)

    #Add the incorrect delay to the array
    bad_delay_array.append(bad_delay*1e6)

    #Calculate the effective (residual) delay
    eff_delay = bad_delay - fix_delay
    print(f"eff delay = {eff_delay}")

    #Calculate the output complex coherence
    gamma_num,gamma_den = pyxis.calc_gamma_numden(eff_delay,F_0,vis)

    #Estimate the current delay envelope
    delay_envelope = pyxis.calc_group_delay_envelope(gamma_num)

    #Add to running average

    #If using the fading memory delay
    #ave_delay_envelope = pyxis.a*delay_envelope + (1-pyxis.a)*ave_delay_envelope

    #If using the rectangular moving average
    rolling_average_array = ff.shift_array(rolling_average_array,1,fill_value=delay_envelope)
    ave_delay_envelope = np.average(rolling_average_array,axis=0)

    #Find estimate the residual delay for adjustment
    adj_delay = pyxis.find_delay(ave_delay_envelope)
    print(f"eff delay estimate = {adj_delay}")

    #How close was the estimate?
    print(f"Off by: {eff_delay-adj_delay}")

    #Adjust the delay line
    fix_delay += adj_delay*gain

    #Add to array
    fix_delay_array.append(fix_delay*1e6)

    #Add the residual delay errors to the array
    error_delay_array.append((fix_delay-bad_delay)*1e6)

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
