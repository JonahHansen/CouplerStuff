import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

#Effective index difference from RSOFT
del_n = 1.50551 - 1.50494

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


def calc_b_lin(n_0):
    """
    Calculate the field amplitude function for a planar tricoupler.
    Assumes b0 = [1,0,0]

    Inputs:
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

    return calc_b(vectors_lin, values_lin, coeffs_lin)


def calc_coeffs_tri(b_0):
    """
    Calculate coefficients for the triangular tricoupler for a given initial
    condition

    Inputs:
        b_0 - Initial modal amplitude vectors
    Outputs:
        array of coefficients

    """

    a1 = 1/2*(-np.sqrt(2)*b_0[0]+np.sqrt(2)*b_0[1])
    a2 = 1/3*(np.sqrt(3)*b_0[0]+np.sqrt(3)*b_0[1]+np.sqrt(3)*b_0[2])
    a3 = 1/6*(-np.sqrt(6)*b_0[0]-np.sqrt(6)*b_0[1]+2*np.sqrt(6)*b_0[2])

    return np.array([a1,a2,a3])


def calc_b_tri(n_0,b_0):
    """
    Calculate the field amplitude function for a triangular tricoupler.

    Inputs:
        n_0 = average refractive index of waveguide
        b_0 = initial condition
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

    return calc_b(vectors_tri,values_tri,a)

###################### PLOTTING/CALCULATING FUNCTIONS #########################

def z_plot(n_0,b_0):
    """
    Plot the modal amplitude of a triangular tricoupler as a function of z

    Inputs:
        n_0 = average refractive index of waveguide
        b_0 = Initial condition (Injection vector)

    """

    b = calc_b_tri(n_0,b_0)
    #Plot |b| as a function of z, up to the beat length of the coupler
    z = np.linspace(0,600e-9/del_n,1000)
    plt.plot(z,b(z))
    plt.show()


def calc_length(n_0):
    """
    Find the length at which if light is injected into one input of the coupler,
    the output is spread equally through each of the three fibers.

    !!! NOTE: THIS IS VERY DIRTY... NEED TO FIND A WAY TO FIND THE FIRST
    !!! MINIMUM OF A PERIODIC FUNCTION...

    Inputs:
        n_0 = Average refractive index of the waveguide
    Outputs:
        z_res = Coupler length

    """

    b = calc_b_tri(n_0,[1,0,0])
    def func(z):
        return np.linalg.norm(b(z) - 1/np.sqrt(3)*np.array([1,1,1]))
    z_res = minimize_scalar(func,bounds=(0,4e-4),method="bounded")

    return z_res


def phi_z_plot(n_0):
    """
    Plot modal amplitude at the ideal coupler length z_res (see above)
    as a function of the input phase of light, phi.

    Assumes an injection vector of 1/sqrt(2)*[1,0,e^(i*phi)]

    Inputs:
        n_0 = Average refractive index of the waveguide

    """

    #Number of phis
    n = 1000
    #Find optimal coupler length (not great at the moment...)
    z = calc_length(n_0).x
    b_array = np.zeros((n,3))
    phis = np.linspace(-np.pi,np.pi,n)
    for i in range(n):
        b_0 = 1/np.sqrt(2)*np.array([1,0,np.exp(1j*phis[i])])
        b_func = calc_b_tri(n_0,b_0)
        b_array[i] = b_func(z)
    plt.plot(phis,b_array)
    plt.ylabel("Modal amplitude (b)")
    plt.xlabel("Phase of input beam 2 (phi)")
    plt.show()
