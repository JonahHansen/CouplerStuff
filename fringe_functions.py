import numpy as np
import matplotlib.pyplot as plt

"""
Bunch of functions for simulating a fringe packet and group delay calculations
"""

def fringe_flux(x,lam,bandpass,F_0,vis,coh_phase,disp_phase=0,offset=0):
    """
    Calculate the flux of a polychromatic fringe pattern
    for a given delay, wavelength and bandpass

    Inputs:
        BASIC INPUTS:
        x = delay length
        lam = central wavelength of channel
        bandpass = width of channel

        ASTRONOMICAL INPUTS (generate fake data for this...):
        F0 = Total flux of object
        vis = visibility modulus of object
        coh_phase = phase of complex coherence, plus atmospheric turbulence

        EXTRA PHASE STUFF:
        disp_phase = additional phase due to glass dispersion
        offset = optional phase offset (for ABCD/AC combiner stuff)

    Outputs:
        Returns intensity of the fringe pattern

    """

    envelope = np.sinc(x*bandpass/lam**2) #Calculate polychromatic envelope
    i = F_0*(1 + envelope*vis*np.cos(2*np.pi*x/lam - coh_phase
                                     - disp_phase - offset))

    return i


################### Refractive Index and Dispersion Functions #################


def sellmeier_equation(lam):
    """
    Calculate the refractive index of BK7 glass at a given wavelength

    Inputs:
        lam = wavelength
    Outputs:
        refractive index n

    """

    B1 = 1.03961212
    B2 = 0.231792344
    B3 = 1.01046945
    C1 = 6.00069867e-3
    C2 = 2.00179144e-2
    C3 = 103.560653

    n2 = 1 + B1*lam**2/(C1**2-lam**2) + \
             B2*lam**2/(C2**2-lam**2) + \
             B3*lam**2/(C3**2-lam**2)

    return np.sqrt(n2)


def calc_group_index(lam):
    """
    Calculate the group index of BK7 glass at a given wavelength

    Inputs:
        lam = wavelength
    Outputs:
        group refractive index n_group

    """

    B1 = 1.03961212
    B2 = 0.231792344
    B3 = 1.01046945
    C1 = 6.00069867e-3
    C2 = 2.00179144e-2
    C3 = 103.560653
    n = sellmeier_equation(lam)

    a1 = B1*lam**2/(C1**2-lam**2)**2
    a2 = B1/(C1**2-lam**2)
    a3 = B2*lam**2/(C2**2-lam**2)**2
    a4 = B2/(C2**2-lam**2)
    a5 = B3*lam**2/(C3**2-lam**2)**2
    a6 = B3/(C3**2-lam**2)

    adjustment_factor = lam**2*(a1+a2+a3+a4+a5+a6)/n

    return n - adjustment_factor


def phaseshift_glass(lam,length,lam_0):
    """
    Calculate the phase shift due to dispersion from unequal path lengths
    of glass

    Inputs:
        length = extra length of glass
        lam = wavelength to calculate shift for
        lam_0 = reference wavelength
    Outputs:
        dispersional phase shift in radians

    """

    n = sellmeier_equation(lam)
    n_grp = calc_group_index(lam_0)
    OPD = (n-n_grp)*length

    phase_shift = OPD*2*np.pi/lam

    return phase_shift


####################### Tricoupler Functions ##################################


def cal_tri_output(delay,wavelengths,bandpass,true_params):
    """
    Calculates the complex coherence from a simulated tricoupler

    Inputs:
        delay = total delay
        wavelengths = wavelength channels to use
        bandpass = wavelength channel width
        true_params = "fake" true parameters of the source as a tuple of:
                      (Flux (reduced by throughput), Visibility, Phase of coherence (with turbulence))
    Outputs:
        Complex coherence for each wavelength

    """

    F_0,vis,coh_phase = true_params

    fluxes = np.zeros((3,len(wavelengths)))
    i = 0
    for output_offset in 2*np.pi/3*np.array([0,1,2]):
        #Calculate intensity output of each fiber (each output has an offset)
        flux = fringe_flux(delay,wavelengths,bandpass,F_0,vis,
                           coh_phase,offset=output_offset)
        #Make it noisy based on shot noise and read noise
        shot_noise = np.random.poisson(flux)
        fluxes[i] = np.round(shot_noise + np.random.normal(scale=1.6))
        i += 1

    return fluxes

    #Combine the outputs into the coherence
    #gamma = (3*fluxes[0] + np.sqrt(3)*1j*(fluxes[2]-fluxes[1]))/np.sum(fluxes,axis=0)-1

def group_delay_envelope(gamma,trial_delays,wavelengths,plot=False):
    """
    Given a complex coherence and a list of trial delays, calculate the group
    delay envelope

    Inputs:
        gamma = Complex coherence
        trial_delays = list of trial delays
        wavelengths = list of wavelength channels
        plot = Whether to plot the white light fringe intensity against the
               trial delays.
    Output:
        List of white light fringe intensities for each trial delay

    """

    phasors = gamma*np.exp(1j*2*np.pi*np.outer(trial_delays,1/wavelengths))
    F_array = np.abs(np.sum(phasors,axis=1))**2
    if plot:
        plt.plot(trial_delays,F_array) #Plot it
        plt.show()

    return F_array


def find_delay(delay_envelope,trial_delays):
    """
    Given a group delay envelope, find an estimate of the group delay

    Inputs:
        delay_envelope = List of white light fringe intensities for each trial delay
        trial_delays = list of trial delays

    Output:
        Estimation of the group delay

    """

    return trial_delays[np.argmax(delay_envelope)]


########################## AC Functions #######################################

def cal_AC_output(delay,wavelengths,bandpass,length,lam_0,true_params):
    """
    Calculates the flux output from a simulated AC coupler

    Inputs:
        delay = total delay
        wavelengths = wavelength channels to use
        bandpass = wavelength channel width
        length = Length of extra glass for dispersion
        lam_0 = Central wavelength for dispersion
        true_params = "fake" true parameters of the source as a tuple of:
              (Flux (reduced by throughput), Visibility, Phase of coherence (with turbulence))
    Outputs:
        Fluxes A and C for each wavelength

    """

    F_0,vis,coh_phase = true_params

    fluxes = np.zeros((2,len(wavelengths)))
    i = 0
    for output_offset in np.pi*np.array([0,1]):
        #Calculate intensity output of each fiber (each output has an offset)
        flux = fringe_flux(delay,wavelengths,bandpass,F_0,vis,
                           coh_phase,phaseshift_glass(wavelengths,length,lam_0),offset=output_offset)
        #Make it noisy based on shot noise and read noise
        shot_noise = np.random.poisson(flux)
        fluxes[i] = np.round(shot_noise + np.random.normal(scale=1.6))
        i += 1

    return fluxes


def fit_vis_delay_AC(gamma_r,delays,visibilities,wavelengths,bandpass,length,lam_0):

    """
    Given the real part of the coherence, a list of trial delays and a list of trial visibilities,
    find an estimate of the visibility and group delay through chi^2 minimization

    Inputs:
        gamma_r = Real part of the complex coherence
        delays = list of trial delays
        visibilities = list of trial visibilities
        bandpass = bandpass of the wavelength channels
        wavelengths = list of wavelength channels
        length = length of the glass extension
        lam_0 = central wavelength for dispersion
    Output:
        Estimation of the visibility and group delay

    """

    #Calculate the trial fringes (with numpy magic)
    envelope = np.sinc(np.outer(delays*bandpass,1/wavelengths**2))
    sinusoid = np.cos(np.outer(2*np.pi*delays,1/wavelengths) - phaseshift_glass(wavelengths,length,lam_0))
    trial_fringes = np.tensordot(visibilities,envelope*sinusoid,axes=0)

    #Calculate chi^2 element
    chi_lam = (trial_fringes - gamma_r)**2
    #Sum over wavelength
    chi_2 = np.sum(chi_lam,axis=2)
    min_index = np.unravel_index(np.argmin(chi_2),chi_2.shape)

    #Find the best fit visibility and delay
    return (visibilities[min_index[0]],delays[min_index[1]])


############################# OLD FUNCTIONS ####################################

def calc_chi_2_AC(gamma_r,delay,vis,wavelengths,bandpass,length,lam_0):
    """
    Given the real part of the coherence and a trial delay, calculate the chi^2 value

    Inputs:
        gamma_r = Real part of the Complex coherence
        delay = trial delay
        wavelengths = list of wavelength channels
        bandpass = bandpass of the wavelength channels
        length = length of the glass extension
        lam_0 = central wavelength for dispersion
    Outputs:
        Chi^2
    """


    #Calculate each element of the chi^2
    chi_lam = (vis*np.sinc(delay*bandpass/wavelengths**2)*np.cos(2*np.pi*delay/wavelengths - phaseshift_glass(wavelengths,length,lam_0)) - gamma_r)**2

    return np.sum(chi_lam) #Sum them up


def find_delay_AC(gamma_r,trial_delays,trial_vis,wavelengths,bandpass,length,lam_0,plot=False):
    """
    Given the real part of the coherence and a list of trial delays, find an estimate
    of the group delay through chi^2 minimization

    Inputs:
        gamma_r = Real part of the Complex coherence
        trial_delays = list of trial delays
        wavelengths = list of wavelength channels
        bandpass = bandpass of the wavelength channels
        length = length of the glass extension
        lam_0 = central wavelength for dispersion
        plot = Whether to plot the white light fringe intensity against the
               trial delays.
    Output:
        Estimation of the group delay

    """

    chi_2_global_min = 1e10
    if plot:
        chi_2_plot_array = []
        for delay in trial_delays:
            chi_2_local_min = 1e10
            for vis in trial_vis:
                #Find the white light intensity of a given trial delay
                chi_2 = calc_chi_2_AC(gamma_r,delay,vis,wavelengths,bandpass,length,lam_0)
                #Is the intensity bigger than any trial delay that's come before?
                if chi_2 < chi_2_local_min:
                    chi_2_local_min = chi_2
                    delay_local_min = delay
            chi_2_plot_array.append(chi_2_local_min)
            if chi_2_local_min < chi_2_global_min:
                chi_2_global_min = chi_2_local_min
                delay_global_min = delay_local_min
        plt.plot(trial_delays,chi_2_plot_array) #Plot it
        plt.xlabel("Delay")
        plt.ylabel("Chi^2")
        plt.show()

    else:
        for delay in trial_delays:
            #Find the white light intensity of a given trial delay
            chi_2 = calc_chi_2_AC(gamma_r,delay,wavelengths,bandpass,length,lam_0)
            #Is the intensity bigger than any trial delay that's come before?
            if chi_2 < chi_2_min:
                chi_2_min = chi_2
                delay_min = delay

    return delay_global_min
