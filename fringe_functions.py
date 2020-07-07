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


def nearest_two_power(x):
    """
    Calculates the nearest power of two above input value

    """
    return 1<<(x-1).bit_length()


def shift_array(arr, num, fill_value=np.nan):
    """
    Shifts every unit in an array down "num" places

    Inputs:
        arr = array to perform shift_array
        num = number of places to shift the numbers down
        fill_value = value to put in the (now) empty spaces of the array
    Outputs:
        Shifted array

    """
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

####################### Interferometer Classes ##################################

class interferometer:
    """
    Parent interferometer class that holds relevant parameters, as well as methods
    for doing fringe tracking

    """
    def __init__(self, bandpass, start_wavelength, end_wavelength, n_outputs,
                        eta, seeing, v, incoh_scaling, num_delays, scale_delay):
        """
        Initialisation of class

        Inputs:
            bandpass = bandpass of each spectral channel (m)
            start_wavelength = smallest wavelength of spectrum (m)
            end_wavelength = longest wavelength of spectrum (m)
            n_outputs = number of outputs for the interferometer
            eta = inherent throughput of interferometer
            seeing = seeing of site (arcsec)
            v = windspeed of site (m/s)
            incoh_scaling = how many coherence times to average group delay
            num_delays = number of trial delays to scan
            scale_delay = scale factor for number of trial delays (smaller = finer)

        """

        self.bandpass = bandpass
        self.wavelengths = np.arange(start_wavelength,end_wavelength,bandpass)[:-1] + 0.5*bandpass
        self.wavenumber_bandpass = 1/start_wavelength - 1/end_wavelength

        #Throughput (tricoupler with instrumental throughput eta)
        self.throughput = 1/n_outputs*eta*1/len(self.wavelengths)

        #Turbulence Data
        self.r0 = 0.98*start_wavelength*180*3600/np.pi/seeing # Fried parameter (m)
        self.t0 = 0.31*(self.r0/v) #Coherent time (s)

        #Telescope details:
        self.B = 20 #Baseline (m)
        self.D = 0.07 #Aperture (m)

        #Integration times
        self.coh_int_time = 1.6*self.t0 #s
        self.incoh_int_time = incoh_scaling*self.t0 #s

        #Scale factor for group envelope averaging (and gain of servo loop):
        self.a = 1 - np.exp(-self.coh_int_time/self.incoh_int_time)

        #List of trial delays to scan for group delay
        self.trial_delays = scale_delay*np.arange(-num_delays/2+1,num_delays/2)/self.wavenumber_bandpass


    def star_flux(self, Rmag):
        """
        Given a star R magnitude, find the number of photons per pixel on the
        detector

        Inputs:
            Rmag = Star's R magnitude
        Outputs:
            Number of photons per pixel on the detector

        """

        #R band constants:
        R_flux = 2.19e-11 #W/m^2/nm
        R_bandpass = 133 #nm
        nu = 4.56e14
        h = 6.62607015e-34 #Js

        #Fake Data:
        f_star = R_flux*10**(-0.4*Rmag)*R_bandpass #W/m^2
        E_star = np.pi*(self.D/2)**2*self.coh_int_time*f_star #J
        F_0 = E_star/(h*nu)*self.throughput #(photons per pixel per integration)

        print(f"Number of photons per pixel: {F_0}")

        return F_0


    def calc_atmosphere(self,n_iter):
        """
        Given an number of iterations, create an array of turbulent atmospheric
        phases based on Kolmogorov Turbulence

        Inputs:
            n_iter = number of iterations for simulation
        Output:
            array of atmospheric phases

        """

        #Number of wavefront "cells" between the two apertures
        self.num_r0s = int(np.ceil(self.B/self.r0))

        #Calculate the number of cells required for the turbulence sim
        num_cells = nearest_two_power(n_iter+self.num_r0s+1)

        #Create the atmosphere
        atm_phases = km1d(num_cells)

        return atm_phases


    def calc_bad_delay(self,atm_phases,index):
        """
        Calculate the bad delay induced by the turbulent atmosphere

        Inputs:
            atm_phases = List of turbulent atmospheric phases
            index = index of iteration of simulation
        Outputs:
            bad delay from atmosphere

        """

        #Calculate the phase error difference between the two apertures
        bad_phase = (atm_phases[index]-atm_phases[index+self.num_r0s])
        #Convert to an OPD, based on the middle wavelength???
        bad_delay = bad_phase*(0.5*(self.wavelengths[-1]-self.wavelengths[0]))/(2*np.pi)

        return bad_delay



class tri_interferometer(interferometer):
    """
    Class for the tricoupler interferometer with 3 outputs
    """
    def __init__(self, bandpass, start_wavelength, end_wavelength,
                        eta, seeing, v, incoh_scaling, num_delays, scale_delay):
        """
        Initialisation of class

        Inputs:
            bandpass = bandpass of each spectral channel (m)
            start_wavelength = smallest wavelength of spectrum (m)
            end_wavelength = longest wavelength of spectrum (m)
            eta = inherent throughput of interferometer
            seeing = seeing of site (arcsec)
            v = windspeed of site (m/s)
            incoh_scaling = how many coherence times to average group delay
            num_delays = number of trial delays to scan
            scale_delay = scale factor for number of trial delays (smaller = finer)

        """

        interferometer.__init__(self, bandpass, start_wavelength,
                                      end_wavelength, 3, eta, seeing, v,
                                      incoh_scaling, num_delays, scale_delay)


    def calc_output(self,delay,F_0,vis):
        """
        Calculates the output fluxes from a simulated tricoupler

        Inputs:
            delay = total delay (s)
            F_0 = total number of photons reduced by throughput
            vis = visibility of source
        Outputs:
            Complex coherence for each wavelength

        """

        fluxes = np.zeros((3,len(self.wavelengths)))
        i = 0
        for output_offset in 2*np.pi/3*np.array([0,1,2]):
            #Calculate intensity output of each fiber (each output has an offset)
            flux = fringe_flux(delay,self.wavelengths,self.bandpass,F_0,vis,offset=output_offset)
            #Make it noisy based on shot noise and read noise
            shot_noise = np.random.poisson(flux)
            fluxes[i] = np.round(shot_noise + np.random.normal(scale=1.6))
            i += 1

        return fluxes


    def calc_gamma_numden(self,bad_delay,F_0,vis):
        """
        Returns the numerator and denominator of the complex coherence from a tricoupler

        Inputs:
            delay = total delay (s)
            F_0 = total number of photons reduced by throughput
            vis = visibility of source
        Outputs:
            numerator and denominator of complex coherence

        """

        fluxes = self.calc_output(bad_delay,F_0,vis)
        gamma_num = 2*fluxes[0] - (fluxes[2]+fluxes[1]) + np.sqrt(3)*1j*(fluxes[2]-fluxes[1])
        gamma_den = np.sum(fluxes,axis=0)

        return gamma_num, gamma_den


    def calc_gamma(self,bad_delay,F_0,vis):
        """
        Returns the complex coherence from a tricoupler

        Inputs:
            delay = total delay (s)
            F_0 = total number of photons reduced by throughput
            vis = visibility of source
        Outputs:
            numerator and denominator of complex coherence

        """

        fluxes = self.calc_output(bad_delay,F_0,vis)

        return (3*fluxes[0] + np.sqrt(3)*1j*(fluxes[2]-fluxes[1]))/np.sum(fluxes,axis=0)-1


    def calc_group_delay_envelope(self,gamma,plot=False):
        """
        Given a complex coherence, calculate the group delay envelope for the
        list of trial delays

        Inputs:
            gamma = Complex coherence
            plot = Whether to plot the white light fringe intensity against the
                   trial delays.
        Output:
            List of white light fringe intensities for each trial delay

        """

        phasors = gamma*np.exp(1j*2*np.pi*np.outer(self.trial_delays,1/self.wavelengths))
        F_array = np.abs(np.sum(phasors,axis=1))**2
        if plot:
            plt.plot(self.trial_delays,F_array) #Plot it
            plt.show()

        return F_array


    def find_delay(self,delay_envelope):
        """
        Given a group delay envelope, find an estimate of the group delay

        Inputs:
            delay_envelope = List of white light fringe intensities for each trial delay
        Output:
            Estimation of the group delay

        """

        return self.trial_delays[np.argmax(delay_envelope)]



class AC_interferometer(interferometer):
    """
    Class for the AC interferometer with only 2 outputs
    """

    def __init__(self, bandpass, start_wavelength, end_wavelength,
                       eta, seeing, v, incoh_scaling, num_delays,
                       scale_delay, disp_length, disp_lam_0):
        """
        Initialisation of class

        Inputs:
            bandpass = bandpass of each spectral channel (m)
            start_wavelength = smallest wavelength of spectrum (m)
            end_wavelength = longest wavelength of spectrum (m)
            eta = inherent throughput of interferometer
            seeing = seeing of site (arcsec)
            v = windspeed of site (m/s)
            incoh_scaling = how many coherence times to average group delay
            num_delays = number of trial delays to scan
            scale_delay = scale factor for number of trial delays (smaller = finer)
            disp_length = length of extra bit of glass in coupler to provide dispersion (m)
            disp_lam_0 = central wavelength for dispersion (m)

        """

        interferometer.__init__(self, bandpass, start_wavelength,
                                      end_wavelength, 2, eta, seeing, v,
                                      incoh_scaling, num_delays, scale_delay)
        self.disp_length = disp_length
        self.disp_lam_0 = disp_lam_0


    def calc_output(self,delay,F_0,vis):
        """
        Calculates the flux output from a simulated AC coupler

        Inputs:
            delay = total delay (s)
            F_0 = total number of photons reduced by throughput
            vis = visibility of source
        Outputs:
            Fluxes A and C for each wavelength

        """

        fluxes = np.zeros((2,len(self.wavelengths)))
        i = 0
        for output_offset in np.pi*np.array([0,1]):
            #Calculate intensity output of each fiber (each output has an offset)
            flux = fringe_flux(delay,self.wavelengths,self.bandpass,F_0,vis,
                               phaseshift_glass(self.wavelengths,self.disp_length,
                               self.disp_lam_0),offset=output_offset)
            #Make it noisy based on shot noise and read noise
            shot_noise = np.random.poisson(flux)
            fluxes[i] = np.round(shot_noise + np.random.normal(scale=1.6))
            i += 1

        return fluxes


    def calc_group_delay_envelope(self,gamma_r,plot=False):
        """
        Given the real part of the coherence, calculate the chi^2 of each
        trial delay.

        Inputs:
            gamma_r = Real part of the complex coherence
            plot = Whether to plot the chi^2 likelihood against the
                   trial delays.
        Output:
            chi^2 array

        """

        #Calculate the trial fringes
        envelope = np.sinc(np.outer(self.trial_delays*self.bandpass,1/self.wavelengths**2))
        sinusoid = np.cos(np.outer(2*np.pi*self.trial_delays,1/self.wavelengths) -
                   phaseshift_glass(self.wavelengths,self.disp_length,self.disp_lam_0))
        trial_fringes = envelope*sinusoid

        #Calculate chi^2 element (normalising to remove visibility dependence)
        chi_lam = (trial_fringes/np.sum(np.abs(trial_fringes),axis=1)[:,None] - gamma_r/(np.sum(np.abs(gamma_r))))**2

        #Sum over wavelength
        chi_2 = np.sum(chi_lam,axis=1)

        if plot:
            plt.plot(pyxis.trial_delays,np.exp(-chi_2**2/2)) #Plot the likelihood
            plt.show()

        return chi_2


    def find_delay(self,delay_envelope):
        """
        Given a chi^2 envelope, find an estimate of the group delay

        Inputs:
            delay_envelope = List of chi^2 values for each trial delay
        Output:
            Estimation of the group delay

        """

        return self.trial_delays[np.argmin(delay_envelope)]
