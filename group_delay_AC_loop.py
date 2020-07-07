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
pyxis = ff.AC_interferometer(bandpass = 15e-9, #m
                              start_wavelength = 600e-9, #m
                              end_wavelength = 750e-9, #m
                              eta = 0.15,
                              seeing = 1, #arcsec
                              v = 20, #m/s
                              incoh_scaling = 30,
                              num_delays = 1000,
                              scale_delay = 0.005,
                              disp_length = 2e-3, #m
                              disp_lam_0 = 675e-9) #m


#Star Flux and visibility
Rmag_star = 5
F_0 = pyxis.star_flux(Rmag_star)
vis = 0.5

#Number of integrations
n_iter = 1000

####################### Find Visibility Bias ##################################

#Atmospheric wavefront (will move across the interferometer)
atm_phases = pyxis.calc_atmosphere(n_iter)

gamma_r_num_array=[]
gamma_r_den_array=[]

print("Begin Bias")

#Calc Bias in visibility
for j in range(n_iter):

    bad_delay = pyxis.calc_bad_delay(atm_phases,j)

    #Calculate the fluxes for each outputs
    fluxes = pyxis.calc_output(bad_delay,F_0,0)

    gamma_r_num_array.append((fluxes[0] - fluxes[1])**2)
    gamma_r_den_array.append((fluxes[0] + fluxes[1]))

#From V^2 ~ 2<(A-C)^2>/<A+C>^2 - bias
#Takes the mean over wavelengths
vis_bias = 2*np.mean(np.mean(gamma_r_num_array)/np.mean(gamma_r_den_array)**2)
print("End Bias")

###################### Science and Delay Loop #################################

#Atmospheric wavefront (will move across the interferometer)
atm_phases = pyxis.calc_atmosphere(n_iter)

#Setup arrays
gamma_r_num_array=[] #Numerator squared of the real part of the coherence
gamma_r_den_array=[] #Denominator of the real part of the coherence
bad_delay_array=[] #Array of atmospheric errors
fix_delay_array=[] #Position of the delay line
error_delay_array=[] #Residuals
ave_delay_envelope = np.zeros(len(pyxis.trial_delays))

#Start delay line at 0
fix_delay = 0

#Gain of servo loop
gain = pyxis.a

vis_array = []

#If using a moving rectangular average, initialise the array
rolling_average_array = np.zeros((int(pyxis.incoh_int_time/pyxis.coh_int_time),1,len(pyxis.trial_delays)))

#Simulate a loop of fringe tracking and science
for j in range(n_iter):

    bad_delay = pyxis.calc_bad_delay(atm_phases,j)

    #Add the incorrect delay to the array
    bad_delay_array.append(bad_delay*1e6)

    #Calculate the effective (residual) delay
    eff_delay = bad_delay - fix_delay
    print(f"eff delay = {eff_delay}")

    #Calculate the output complex coherence
    fluxes = pyxis.calc_output(eff_delay,F_0,vis)

    gamma_r = (fluxes[0] - fluxes[1])/(fluxes[0] + fluxes[1])
    gamma_r_num_array.append((fluxes[0] - fluxes[1])**2)
    gamma_r_den_array.append((fluxes[0] + fluxes[1]))

    #Estimate the squared visibility V^2 ~ 2<(A-C)^2>/<A+C>^2 - Bias
    #Takes the mean over wavelengths
    vis_est = 2*np.mean(np.mean(gamma_r_num_array,axis=0)/np.mean(gamma_r_den_array,axis=0)**2) - vis_bias

    #Append the running average of visibilities
    vis_array.append(vis_est)

    #Estimate the current delay envelope
    delay_envelope = pyxis.calc_group_delay_envelope(gamma_r)

    #Add to running average

    #If using the fading memory delay
    #ave_delay_envelope = a*delay_envelope + (1-a)*ave_delay_envelope

    #If using the rectangular moving average
    rolling_average_array = ff.shift_array(rolling_average_array,1,fill_value=delay_envelope)
    ave_delay_envelope = np.average(rolling_average_array,axis=0)

    #Find estimate the residual delay for adjustment
    adj_delay = pyxis.find_delay(ave_delay_envelope)
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
