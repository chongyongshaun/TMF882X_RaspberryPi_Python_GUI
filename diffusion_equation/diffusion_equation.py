import logging
import numpy as np

from enum import Enum

from diffusion_equation import total
Phantom = Enum('Phantom',[('SLAB','SLAB'),('SEMIINF','SEMIINF')])

def Contini1997(rho,t,s,mua,musp,n1,n2,phantom,mua_independent,m):
    """
    Implementation of the diffusion equation from:

    | *Contini D, Martelli F, Zaccanti G.*
    | *Photon migration through a turbid slab described by a model based diffusion approximation.*
    | *I. Theory, *Applied Optics Vol 36, No 19, 1997, pp 4587-4599*

    Page numbers in below documentation are from the above journal article.

    :param phantom: Either semi-infinite or slab ('semiinf' or 'slab'). If the former, then `s` will be set to 'inf' and if the latter then s must be specified.
    :param rho: radial position of the detector (mm)
    :param t: time (ns)
    :param s: slab thickness (mm)
    :param m: maximum number of positive or negative sources. This defaults to 200 if not specified for a slab phantom and is set to 0 for a semi-infinite medium.
    :param mua: absorption coefficient (:math:`mm^{-1}`)
    :param musp: reduced scattering coefficient (:math:`mm^{-1}`)
    :param n1: external medium refractive index
    :param n2: diffusing medium refractive index

    :param mua_independent: If true, the mua independent form of the diffusion coefficient equation (p 4588)
        Otherwise, the mua dependent form given in equation 19 will be used.

    :returns RTtotal: A dictionary whose only entry 'total' is the time resolved reflectance and transmittance (eqns (36), (39)) [mm^(-2) ps^(-1)]. Tuple of two N x M matrices (N rho values and M t values)

    Can also return various other values, but these have been commented out as their computation is not yet necessary.

    Original MATLAB code by:
    Tiziano BINZONI (University of Geneva)
    Fabrizio MARTELLI (University of Firenze)
    Alessandro TORRICELLI (Politecnico di Milano)
    """
    # FIXME: time might be in ps, not ns
    # Nah, pretty sure tis in ns
    # Conversion of quantities into SI units and conversion to known types
    rho  = np.array([float(r)*1e-3 for r in rho]) if isinstance(rho, list) else np.array([float(rho)*1e-3])
    t    = np.array([float(tt)*1e-9 for tt in t])
    s    = float(s)   *1e-3
    mua  = float(mua) *1e+3
    musp = float(musp)*1e+3
    n1   = float(n1)
    n2   = float(n2)
    mua_independent = bool(mua_independent)
    try:
        m = int(m) if str(m) != 'inf' else 0
    except ValueError:
        m = int(float(m))

    try:
        phantom = Phantom(str(phantom).upper())
    except ValueError as e:
        raise ValueError(f'Phantom type not specified as one of "slab" or "semiinf"')
    if phantom == Phantom.SLAB:
        if not m:
            m=200
    elif phantom == Phantom.SEMIINF:
        m=0
    # Max acceptable error on computed data
    error=1e-6

    # ADDED TO FIX SEMIINF PHANTOM BEHAVIOUR ISSUE
    if phantom == Phantom.SLAB:
        RT_total                       = total.RT                      (rho,t,s,m,mua,musp,n1,n2,mua_independent)
    if phantom == Phantom.SEMIINF:
        RT_total                       = total.RT                      (rho,t,s,m+1,mua,musp,n1,n2,mua_independent)
    # ADDED TO FIX SEMIINF PHANTOM BEHAVIOUR ISSUE
    #### BELOW IS THE ORIGINAL CODE FOR THE RETURN VALUE OF RT_TOTAL
        
    #RT_cw_source                   = cw_source.RT                  (rho  ,s,m,mua,musp,n1,n2,mua_independent)
    #RT_infinite_beam               = infinite_beam.RT              (    t,s,m,mua,musp,n1,n2,mua_independent)
    #RT_infinite_beam_time_integral = infinite_beam_time_integral.RT(      s,m,mua,musp,n1,n2,mua_independent)
    #RT_mean_path_length            = mean_path_length.RT           (rho  ,s,m,mua,musp,n1,n2,mua_independent)
    if not phantom == Phantom.SEMIINF:
        RT1 = [
                 total.RT                      (rho,t,s,m+1,mua,musp,n1,n2,mua_independent)
    #            ,cw_source.RT                  (rho  ,s,m+1,mua,musp,n1,n2,mua_independent)
    #            ,infinite_beam.RT              (    t,s,m+1,mua,musp,n1,n2,mua_independent)
    #            ,infinite_beam_time_integral.RT(      s,m+1,mua,musp,n1,n2,mua_independent)
    #            ,mean_path_length.RT           (rho  ,s,m+1,mua,musp,n1,n2,mua_independent)
        ]
    #    for x,y,l in zip([ RT_total ,RT_cw_source ,RT_infinite_beam ,RT_infinite_beam_time_integral ,RT_mean_path_length ],RT1,['total','cw_source','infinite_beam','infinite_beam_time_integral','mean_path_length']):
    #        if all([a.size > 0 for a in [x[0], y[0], x[1],y[1]]]):
    #            e = [max(abs((x[0].flatten()-y[0].flatten())/x[0].flatten()*100)),max(abs((x[1].flatten()-y[1].flatten())/x[1].flatten()*100))]
    #            for ee,rt in zip(e,['Reflectance','Transmittance']):
    #                if ee > error:
    #                    logging.warning(f"Increase m [{m}] (error [{ee}] larger than [{error}] for {l}:{rt}")

    #a = A(n1,n2)
    #M = range(-m,m) if m != 0 else [0]
    #Z = np.array([z(s,i,mua,musp,n1,n2,mua_independent) for i in M])
    return {
            'total' : RT_total
    #        , 'cw_source' : RT_cw_source
    #        , 'infinite_beam' : RT_infinite_beam
    #        , 'mean_path_length' : RT_mean_path_length
    #        , 'infinite_beam_time_integral' : RT_infinite_beam_time_integral
    #        , 'A' : a
    #        , 'z' : Z
            }

if __name__ == '__main__':
    import sys
    from time_of_flight.parsing import Data # type: ignore
    import matplotlib.pyplot as plot

    irf  = Data.from_file(sys.argv[1]).data[:,0,0,0]
    meas = Data.from_file(sys.argv[2]).data[:,0,0,0]
    time = np.linspace(0,4096*2.02,4096)
    theory_body = Contini1997([0],time,18*1e3,0.1030505*1e-4,5.433029*1e-4,1,1.4,'slab',True,400)['total'][1][0]
    
    def slidingavg(vector, N):
        # TODO: consider uniform filter 1D version
        cumsum = np.cumsum(np.insert(vector, 0, 0))
        return (cumsum[N:] - cumsum[:-N]) / float(N)

    noise_window=(400,570)
    bg = (
            np.mean(irf[noise_window[0]:noise_window[1]]),
            np.mean(meas[noise_window[0]:noise_window[1]])
        )

    average_width = 3

    meas_bg = meas - bg[1]
    meas_avg_bg = slidingavg(meas_bg, average_width)
    meas_max = max(meas_avg_bg)
    meas_bg_corrected = meas_bg/meas_max

    irf_bg = irf - bg[0]
    irf_avg = slidingavg(irf_bg, average_width)
    irf_bg_corrected = irf_bg/max(irf_avg)

    theory = np.convolve(theory_body,irf_bg_corrected)
    theory = theory/max(theory)

    plot.plot(time, meas_bg_corrected[0:len(time)], '.', color='orange', label='Measured output')
    plot.plot(time+40, theory[0:len(time)], color='blue', label=f'Theoretical output')
    plot.legend()
    plot.xlabel(f'Time [ps]')
    plot.ylabel(f'Magnitude')
    plot.show()
