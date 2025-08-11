from .diffusion import D
from .A import A

def z(s,m,mua,musp,n1,n2,mua_independent):
    """
    Equation 37
    :param s: slab thickness
    :param m: maximum number of positive or negative sources
    :param mua: absorption coefficient
    :param musp: reduced scattering coefficient
    :param n1: external medium
    :param n2: diffusing medium
    """
    z0 = 1/float(musp)
    a = A(n1,n2)
    d = D(mua, musp, mua_independent)

    # Page 4592
    ze = 2*a*d

    z1 = float(s)*(1-2*int(m)) - 4*int(m)*ze - z0;
    z2 = float(s)*(1-2*int(m)) - (4*int(m)-2)*ze + z0;
    z3 = -2*int(m)*float(s) -4*int(m)*ze - z0;
    z4 = -2*int(m)*float(s) -(4*int(m)-2)*ze + z0;

    return tuple(float(f) for f in [z1,z2,z3,z4])

