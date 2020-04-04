import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

plt.ion()
######################### b CALCULATION FUNCTIONS #############################

def calc_b(vectors, values, coeffs, n_0):
    """
    Calculate field amplude equation as a function of z for a given
    coupler system

    Inputs:
        vectors = eigenvectors of the given system
        values = eigenvalues of the given system
        coeffs = coefficients based on an initial condition
        n_0 = Average refractive index of waveguide
    Outputs:
        Function that returns |b| as a function of z

    """

    def b(z):
        """
        Calculate |b| as a function of z

        Inputs:
            z = length along coupler
        Outputs:
            |b| - modal amplitude at length z

        """

        z = np.atleast_1d(z)
        #Propagation constant at 600nm
        beta = 2*np.pi/600e-9*n_0

        result = np.zeros((len(z),3),dtype='complex128')
        for i in range(3):
            result += coeffs[i]*np.outer(np.exp(1j*beta*values[i]*z),vectors[i])
        res = np.abs(result)

        return res

    return b


def calc_b_lin(del_n, n_0):
    """
    Calculate the field amplitude function for a planar tricoupler.
    Assumes b0 = [1,0,0]

    Inputs:
        del_n = change in effective refractive index between modes
        n_0 = Average refractive index of waveguide
    Outputs:
        function to calculate |b| as a function of z

    """

    #Coupling coefficient
    delta = del_n/(2*n_0)
    #Eigenvectors
    vectors_lin = np.array([[1/2,1/np.sqrt(2),1/2],
                            [1/np.sqrt(2),0,-1/np.sqrt(2)],
                            [1/2,-1/np.sqrt(2),1/2]])
    #Eigenvalues
    values_lin = np.array([1+np.sqrt(2)*delta,1,1-np.sqrt(2)*delta])
    #Coefficients
    coeffs_lin = 1/2*np.array([1,np.sqrt(2),1])

    return calc_b(vectors_lin, values_lin, coeffs_lin, n_0)


def calc_coeffs_tri(b_0):
    """
    Calculate coefficients for the triangular tricoupler for a given initial
    condition

    Inputs:
        b_0 = Injection vector
    Outputs:
        array of coefficients

    """

    a1 = 1/2*(-np.sqrt(2)*b_0[0]+np.sqrt(2)*b_0[1])
    a2 = 1/3*(np.sqrt(3)*b_0[0]+np.sqrt(3)*b_0[1]+np.sqrt(3)*b_0[2])
    a3 = 1/6*(-np.sqrt(6)*b_0[0]-np.sqrt(6)*b_0[1]+2*np.sqrt(6)*b_0[2])

    return np.array([a1,a2,a3])


def calc_b_tri(b_0, del_n, n_0):
    """
    Calculate the field amplitude function for a triangular tricoupler.

    Inputs:
        b_0 = Injection vector
        del_n = change in effective refractive index between modes
        n_0 = average refractive index of waveguide
    Outputs:
        function to calculate |b| as a function of z

    """

    #Coupling coefficient
    delta = del_n/(2*n_0)
    #Eigenvectors
    vectors_tri = np.array([[-1/np.sqrt(2),1/np.sqrt(2),0],
                        [1/np.sqrt(3),1/np.sqrt(3),1/np.sqrt(3)],
                        [-1/np.sqrt(6),-1/np.sqrt(6),2/np.sqrt(6)]])
    #Eigenvalues
    values_tri = np.array([1-delta,1+2*delta,1-delta])
    #Coefficients
    a = calc_coeffs_tri(b_0)

    return calc_b(vectors_tri,values_tri,a,n_0)


###############################################################################

def find_length(lam, del_n):
    """
    Calculate the ideal length of the coupler (see maths)

    Inputs:
        lam = Wavelength
        del_n = change in effective refractive index between modes
    Outputs:
        Length of the coupler
    """

    return 2/9*lam/del_n

def calc_bz(b_0, del_n, n_0):
    """
    Calculate the output of the tricoupler at the ideal length from an input

    Inputs:
        b_0 = Injection vector
        del_n = change in effective refractive index between modes
        n_0 = average refractive index of waveguide
    Outputs:
        b_z_res = The Output modal vector at the ideal length z_res
    """

    delta = del_n/(2*n_0)

    #Transfer matrix
    M = np.array([[1,np.exp(1j*2*np.pi/3),np.exp(1j*2*np.pi/3)],
                 [np.exp(1j*2*np.pi/3),1,np.exp(1j*2*np.pi/3)],
                 [np.exp(1j*2*np.pi/3),np.exp(1j*2*np.pi/3),1]])
    phase_factor = np.exp(1j*(4-delta)*np.pi/(18*delta))

    return 1/np.sqrt(3)*phase_factor*np.dot(M,b_0)


###################### PLOTTING/CALCULATING FUNCTIONS #########################

def z_plot(b_0, del_n, n_0):
    """
    Plot the modal amplitude of a triangular tricoupler as a function of z

    Inputs:
        b_0 = Initial condition (Injection vector)
        del_n = change in effective refractive index between modes
        n_0 = average refractive index of waveguide

    """

    b = calc_b_tri(b_0,del_n,n_0)
    #Plot |b| as a function of z, up to the beat length of the coupler
    z = np.linspace(0,600e-9/del_n,1000)
    plt.plot(z,b(z))
    plt.xlabel("Length of coupler (z)")
    plt.ylabel("Modal amplitude (b)")
    plt.show()


def phi_z_plot(del_n, n_0):
    """
    Plot modal amplitude at the ideal coupler length z_res (see above)
    as a function of the input phase of light, phi.

    Assumes an injection vector of 1/sqrt(2)*[1,0,e^(i*phi)]

    Inputs:
        del_n = change in effective refractive index between modes
        n_0 = average refractive index of waveguide
    """

    #Number of phis
    n = 1000
    b_array = np.zeros((n,3))
    phis = np.linspace(-np.pi,np.pi,n)
    for i in range(n):
        b_0 = 1/np.sqrt(2)*np.array([1,0,np.exp(1j*phis[i])])
        b_array[i] = np.abs(calc_bz(b_0, del_n, n_0))
    plt.plot(phis,b_array)
    plt.xlabel("Phase of input beam 2 (phi)")
    plt.ylabel("Modal amplitude (b)")
    plt.show()


if __name__ == "__main__":

    #Effective index difference from RSOFT
    del_n = 1.50551 - 1.50494
    #Average refractive index of waveguide
    n_0 = 1.54

    plt.figure(1)
    z_plot([1,0,0],del_n,n_0,)

    plt.figure(2)
    phi_z_plot(del_n,n_0)
