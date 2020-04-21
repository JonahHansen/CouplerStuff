""" kmf Functions from Opticstools """
import numpy as np

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
