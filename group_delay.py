import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def flux(x,kappa,bandpass,F0,vis,coh_phase,disp_phase,offset=0):
    """
    Calculate the flux of a polychromatic fringe pattern
    for a given delay, wavelength and bandpass

    Inputs:
        BASIC INPUTS:
        x = delay length
        kappa = central angular wavenumber of channel (2pi/lambda)
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

    envelope = np.sinc(np.pi*x/bandpass) #Calculate polychromatic envelope
    i = F_0*(1+envelope*vis*np.cos(kappa*x - coh_phase - disp_phase - offset))

    return i


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


#Fake Data:
F_0 = 1.5
coh_phase = np.pi/6
vis = 0.5

#List of wavelength channels, with spacing 20nm.
lam_channels = np.array([610e-9,630e-9,650e-9,670e-9,690e-9,710e-9,730e-9])
#Convert to angular wavenumber:
wavenumbers = 2*np.pi/lam_channels

#Dispersion length of glass
width = 5e-3
#Dispersional phase shift of glass, with 650nm being the reference wavelength
glass_phase = phaseshift_glass(lam_channels,width ,650e-9)

#List of delays to plot against
delays = np.linspace(-5e-7,5e-7,100000)

#For each channel,
for i in range(len(lam_channels)):
    #Plot fringe pattern
    plt.plot(delays,flux(delays,wavenumbers[i],bandpass=20e-9,
                         F0=F_0,vis=vis,coh_phase=coh_phase,
                         disp_phase=glass_phase[i]))

plt.show()
