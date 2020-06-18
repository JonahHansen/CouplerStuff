import numpy as np
import matplotlib.pyplot as plt

"""
Bunch of functions for simulating a fringe packet and group delay calculations
"""

def fringe_flux(x,lam,bandpass,F_0,vis,disp_phase=0,offset=0):
    """
    Calculate the flux of a polychromatic fringe pattern
    for a given delay, wavelength and bandpass

    Inputs:
        BASIC INPUTS:
        x = delay length (in m)
        lam = central wavelength of channel (in m)
        bandpass = width of channel (in m)

        ASTRONOMICAL INPUTS (generate fake data for this...):
        F0 = Total flux of object
        vis = visibility modulus of object

        EXTRA PHASE STUFF:
        disp_phase = additional phase due to glass dispersion
        offset = optional phase offset (for ABCD/AC combiner stuff)

    Outputs:
        Returns intensity of the fringe pattern

    """

    envelope = np.sinc(x*bandpass/lam**2) #Calculate polychromatic envelope
    i = F_0*(1 + envelope*vis*np.cos(2*np.pi*x/lam - disp_phase - offset))

    return i


################### Refractive Index and Dispersion Functions #################

def sellmeier_equation(lam):
    """
    Calculate the refractive index of BK7 glass at a given wavelength

    Inputs:
        lam = wavelength (in microns)
    Outputs:
        refractive index n

    """

    B1 = 1.03961212
    B2 = 0.231792344
    B3 = 1.01046945
    C1 = 6.00069867e-3
    C2 = 2.00179144e-2
    C3 = 103.560653

    n2 = 1 + B1*lam**2/(lam**2-C1) + \
             B2*lam**2/(lam**2-C2) + \
             B3*lam**2/(lam**2-C3)

    return np.sqrt(n2)


def calc_group_index(lam):
    """
    Calculate the group index of BK7 glass at a given wavelength

    Inputs:
        lam = wavelength (in microns)
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

    a1 = -B1*lam**2/(lam**2-C1)**2
    a2 = B1/(lam**2-C1)
    a3 = -B2*lam**2/(lam**2-C2)**2
    a4 = B2/(lam**2-C2)
    a5 = -B3*lam**2/(lam**2-C3)**2
    a6 = B3/(lam**2-C3)

    adjustment_factor = lam**2*(a1+a2+a3+a4+a5+a6)/n

    return n - adjustment_factor


def phaseshift_glass(lam,length,lam_0):
    """
    Calculate the phase shift due to dispersion from unequal path lengths
    of glass

    Inputs:
        length = extra length of glass (in m)
        lam = wavelength to calculate shift for (in m)
        lam_0 = reference wavelength (in m)
    Outputs:
        dispersional phase shift in radians

    """

    n = sellmeier_equation(lam*1e6)
    n_grp = calc_group_index(lam_0*1e6)
    OPD = (n-n_grp)*length
    #import pdb; pdb.set_trace()
    phase_shift = OPD*2*np.pi/lam

    return phase_shift

########################## Other Functions ####################################

def star_flux(Rmag, coh_time, D, throughput):
    """
    Calculates the number of photons per pixel from a star of a certain magnitude

    Inputs:
        Rmag = R magnitude of the star
        coh_time = coherent integration time (in s)
        D = diameter of the aperture (in m)
        throughput = throughput per pixel of the combiner
    Outputs:
        Photons per pixel
    """

    #R band constants:
    R_flux = 2.19e-11 #W/m^2/nm
    R_bandpass = 133 #nm
    nu = 4.56e14
    h = 6.62607015e-34 #Js

    #Fake Data:
    f_star = R_flux*10**(-0.4*Rmag)*R_bandpass #W/m^2
    E_star = np.pi*(D/2)**2*coh_time*f_star #J
    F_0 = E_star/(h*nu)*throughput #(photons per pixel per integration)

    print(f"Number of photons per pixel: {F_0}")
    return F_0


def km1d(sz, r_0_pix=None):
    """
    Algorithm:
        y(midpoint) = ( y(x1) + y(x2) )/2 + 0.4542*Z, where
            0.4542 = sqrt( 1 - 2^(5/3) / 2 )
    """
    if sz != 2**int(np.log2(sz)):
        raise UserWarning("Size must be within a factor of 2")
    #Temporary code.
    wf = kmf(sz, r_0_pix=r_0_pix)
    return wf[0]


def kmf(sz, L_0=np.inf, r_0_pix=None):
    """This function creates a periodic wavefront produced by Kolmogorov turbulence.
    It SHOULD normalised so that the variance at a distance of 1 pixel is 1 radian^2.
    To scale this to an r_0 of r_0_pix, multiply by sqrt(6.88*r_0_pix**(-5/3))

    The value of 1/15.81 in the code is (I think) a numerical approximation for the
    value in e.g. Conan00 of np.sqrt(0.0229/2/np.pi)

    Parameters
    ----------
    sz: int
        Size of the 2D array

    l_0: (optional) float
        The von-Karmann outer scale. If not set, the structure function behaves with
        an outer scale of approximately half (CHECK THIS!) pixels.

    r_0_pix: (optional) float
	The Fried r_0 parameter in units of pixels.

    Returns
    -------
    wavefront: float array (sz,sz)
        2D array wavefront, in units of radians. i.e. a complex electric field based
        on this wavefront is np.exp(1j*kmf(sz))
    """
    xy = np.meshgrid(np.arange(sz/2 + 1)/float(sz), (((np.arange(sz) + sz/2) % sz)-sz/2)/float(sz))
    dist2 = np.maximum( xy[1]**2 + xy[0]**2, 1e-12)
    ft_wf = np.exp(2j * np.pi * np.random.random((sz,sz//2+1)))*dist2**(-11.0/12.0)*sz/15.81
    ft_wf[0,0]=0
    if r_0_pix is None:
        return np.fft.irfft2(ft_wf)
    else:
        return np.fft.irfft2(ft_wf) * np.sqrt(6.88*r_0_pix**(-5/3.))


#Calculate nearest power of two
def nearest_two_power(x):
    return 1<<(x-1).bit_length()


#Shifts every unit in an array down "num" places
def shift_array(arr, num, fill_value=np.nan):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result


####################### Tricoupler Functions ##################################

def cal_tri_output(delay,wavelengths,bandpass,F_0,vis):
    """
    Calculates the complex coherence from a simulated tricoupler

    Inputs:
        delay = total delay (s)
        wavelengths = wavelength channels to use (m)
        bandpass = wavelength channel width (m)
        F_0 = total number of photons reduced by throughput
        vis = visibility of source
    Outputs:
        Complex coherence for each wavelength

    """

    fluxes = np.zeros((3,len(wavelengths)))
    i = 0
    for output_offset in 2*np.pi/3*np.array([0,1,2]):
        #Calculate intensity output of each fiber (each output has an offset)
        flux = fringe_flux(delay,wavelengths,bandpass,F_0,vis,offset=output_offset)
        #Make it noisy based on shot noise and read noise
        shot_noise = np.random.poisson(flux)
        fluxes[i] = np.round(shot_noise + np.random.normal(scale=1.6))
        i += 1

    return fluxes

    #Combine the outputs into the coherence
    #gamma = (3*fluxes[0] + np.sqrt(3)*1j*(fluxes[2]-fluxes[1]))/np.sum(fluxes,axis=0)-1


def tri_group_delay_envelope(gamma,trial_delays,wavelengths,plot=False):
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


def find_tri_delay(delay_envelope,trial_delays):
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

def cal_AC_output(delay,wavelengths,bandpass,length,lam_0,F_0,vis):
    """
    Calculates the flux output from a simulated AC coupler

    Inputs:
        delay = total delay (s)
        wavelengths = wavelength channels to use (m)
        bandpass = wavelength channel width (m)
        length = Length of extra glass for dispersion (m)
        lam_0 = Central wavelength for dispersion (m)
        F_0 = total number of photons reduced by throughput
        vis = visibility of source
    Outputs:
        Fluxes A and C for each wavelength

    """

    fluxes = np.zeros((2,len(wavelengths)))
    i = 0
    for output_offset in np.pi*np.array([0,1]):
        #Calculate intensity output of each fiber (each output has an offset)
        flux = fringe_flux(delay,wavelengths,bandpass,F_0,vis,
                           phaseshift_glass(wavelengths,length,lam_0),offset=output_offset)
        #Make it noisy based on shot noise and read noise
        shot_noise = np.random.poisson(flux)
        fluxes[i] = np.round(shot_noise + np.random.normal(scale=1.6))
        i += 1

    return fluxes


def AC_group_delay_envelope(gamma_r,delays,visibilities,wavelengths,bandpass,length,lam_0):

    """
    Given the real part of the coherence, a list of trial delays and a list of trial visibilities,
    calculate the chi^2 of each combination of delays and visibilities.

    Inputs:
        gamma_r = Real part of the complex coherence
        delays = list of trial delays
        visibilities = list of trial visibilities
        bandpass = bandpass of the wavelength channels
        wavelengths = list of wavelength channels
        length = length of the glass extension
        lam_0 = central wavelength for dispersion
    Output:
        chi^2 array

    """

    #Calculate the trial fringes (with numpy magic)
    envelope = np.sinc(np.outer(delays*bandpass,1/wavelengths**2))
    sinusoid = np.cos(np.outer(2*np.pi*delays,1/wavelengths) - phaseshift_glass(wavelengths,length,lam_0))
    trial_fringes = np.tensordot(visibilities,envelope*sinusoid,axes=0)

    #Calculate chi^2 element
    chi_lam = (trial_fringes - gamma_r)**2
    #Sum over wavelength
    chi_2 = np.sum(chi_lam,axis=2)

    return chi_2


def find_AC_delay_vis(chi_2,trial_delays,trial_vis):
    """
    Given a chi^2 array, plus a list of trial delays and visibilities,
    return the estimate of the delay and visibility

    Inputs:
        chi_2 = chi^2 array
        trial_delays = list of trial delays
        trial_vis = list of trial visibilities

    Output:
        Estimation of the group delay and visibility

    """
    min_index = np.unravel_index(np.argmin(chi_2),chi_2.shape)

    #Find the best fit visibility and delay
    return trial_delays[min_index[1]],trial_vis[min_index[0]]
