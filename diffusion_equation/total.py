import numpy as np

from diffusion_equation.coefficients import D, z

def RT(rho,t,s,m,mua,musp,n1,n2,mua_independent):
    """
    Computes equations (36) and (39)
    
    :param rho: radial position of the detector at distance s
    :param t: time
    :param s: slab thickness
    :param m: maximum number of positive or negative sources
    :param mua: absorption coefficient
    :param musp: reduced scattering coefficient
    :param n1: external medium
    :param n2: diffusing medium
    """
    rho = np.array(rho,dtype=float)
    t = np.array(t)
    s = float(s)
    m = int(m)
    mua = float(mua)
    n2 = float(n2)

    R = np.zeros((len(rho),len(t)))
    T = np.zeros((len(rho),len(t)))
    for i in range(0,R.shape[0]):
        c=299792458

        v=c/n2

        d=D(mua,musp,mua_independent)

        m = range(-m,m) if m != 0 else [0]
        Z = np.array([z(s,i+1,mua,musp,n1,n2,mua_independent) for i in m])

        R[i] = -np.exp(-mua*v*t-rho**2/(4*d*v*t))/(2*(4*np.pi*d*v)**(3/2)*t**(5/2))
        R[i]*= sum([zz[2]*np.exp(-zz[2]**2/(4*d*v*t)) - zz[3]*np.exp(-zz[3]**2/(4*d*v*t)) for zz in Z])
        R[i]*= 1e-6*1e-12
        T[i] =  np.exp(-mua*v*t-rho**2/(4*d*v*t))/(2*(4*np.pi*d*v)**(3/2)*t**(5/2))\
                * sum([zz[0]*np.exp(-zz[0]**2/(4*d*v*t)) - zz[1]*np.exp(-zz[1]**2/(4*d*v*t)) for zz in Z])\
                * 1e-6*1e-12

        R[i][t<=0]=0
        T[i][t<=0]=0
        
    T=T if m != [0] else np.array([])

    return (R,T)

