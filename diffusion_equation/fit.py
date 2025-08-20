"""
FIT
===

All code related to fitting curves to data should exist here.
For code related to displaying fitted data, see `time_of_flight.gui.windows.plot`.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import least_squares
import numpy as np
# from .parsing import Data
# from .diffusion_equation import Contini1997
from diffusion_equation.diffusion_equation import Contini1997
import time as t

import logging
from enum import IntEnum

GEOMETRY = IntEnum('GEOMETRY', {'REFLECTANCE':0, 'TRANSMITTANCE':1})

def model(irf,rho,time,s,mua,musp,n1,n2,phantom,mua_independent,m, geometry=GEOMETRY.REFLECTANCE, offset=0):
    """
    This function returns the convolution of the theoretical model given by `diffusion_equation.Contini1997` with the measured irf

    All parameters that are input to `Contini1997` are described there.

    :param geometry: The particular geometry in use, i.e. transmittance or reflectance. Values are enumerated in `GEOMETRY`
    :param offset: Number of cells to offset the result by to most closely line it up with the measured data
    """
    theoretical = Contini1997(rho,time,s,mua,musp,n1,n2,phantom,mua_independent,m)

    # ret = np.pad(np.convolve(theoretical['total'][int(geometry)][0], irf), (offset,0)) 
    ret = np.pad(scipy.signal.fftconvolve(theoretical['total'][int(geometry)][0], irf), (offset,0))
    ret = ret[:len(irf)]
    max_val = np.max(ret)
    if max_val > 0:
        return ret / max_val
    else:
        return np.zeros_like(ret)  # Avoid division by zero
    
def convolve_irf_with_model(irf, model, geometry=GEOMETRY.REFLECTANCE, offset=0, normalize_irf=True, normalize_model=True, denest_contini_output=False):
    """
    irf: 1D float array with n values where n = number of time bins
    model: 1D float array with n values if denest_contini_output is False, otherwise a dict with the key 'total' which contains a 2D array with the first dimension being the geometry and the second dimension being the reflectance and transmittance values
    geometry: The particular geometry in use, i.e. transmittance or reflectance. Values are enumerated in `GEOMETRY`, only useful if denest_contini_output is True
    offset: Number of cells to offset the result by to most closely line it up with the measured data
    normalize_irf: If True, normalize the irf to its maximum value before convolution
    normalize_model: If True, normalize the model to its maximum value before convolution
    denest_contini_output: If True, denest the output of Contini1997() which is a dict with the only key 'total', described in model param
    """
    if denest_contini_output:
        model = model['total'][int(geometry)][0]
    # check if irf and model have the same dimensions
    if len(irf) != len(model):
        raise ValueError(f"IRF and model must have the same length. IRF length: {len(irf)}, model length: {len(model)}")
    
    # % area normalization (below)
    # ci_arnorm = ci_arnorm(1:length(irf_bg_corrected));
    # ci_offset_n = ci_arnorm./sum(ci_arnorm(xleft:xright));
    # ci_offset_n = ci_offset_n.*sum(meas_bg_corrected(xleft:xright)); 

    if normalize_irf:
        irf = irf / np.max(irf)
    if normalize_model:
        model = model / np.max(model)
    res = np.pad(scipy.signal.fftconvolve(model, irf), (offset,0))
    #truncate the result to the length of the model/irf
    res = res[:len(irf)]
    return res

#TODO: add an option to only take the main curve, meaning, assuming normalized, peak of the curve for model and measured data should be 1, to the left of peak, cut off when y = 0.8, to the right of the peak, cut off when y=0.01 before taking the residual 
def fun_residual(x, time, irf, measured, rho=0, n1=1,n2=1.4, fit_start=0, fit_end=-1, mua_independent=True, phantom='semiinf',s=0, m=0, offset=0, geometry=GEOMETRY.TRANSMITTANCE, smart_crop=False, optimize_shift = True):
    """
    :param x: [mua, musp, shift] to be calculated
    :param irf: input response function
    :param time: time
    :param measured: measured quantity
    :param fit_start: initial index of fit range
    :param fit_end: final index of fit range
    :param rho: source detector separation
    :param optimize_shift: If True, the shift parameter is optimized, otherwise it is set to 0
    """
    # Time convolution of irfs with DE performed within model fn
    # TODO:Find a way to do this more efficiently
    """
    x[0] = mua
    x[1] = musp
    x[2] = time offset (in bins)
    """
    mua, musp, shift = x[0], x[1], x[2]

    model_vals = model(
        irf, rho, time, s,
        mua, musp, n1, n2,
        phantom, mua_independent, m,
        offset=0, geometry=geometry
    )


    if optimize_shift:
        # measured = apply_offset(measured, shift)
        model_vals = apply_offset(model_vals, shift)

    # Match lengths
    model_vals = model_vals[:len(measured)]
    measured = measured[:len(model_vals)]

    if smart_crop:
        peak_idx = measured.argmax()
        peak_val = measured[peak_idx]

        left_idx = peak_idx
        while left_idx > 0 and measured[left_idx] > 0.8 * peak_val:
            left_idx -= 1

        right_idx = peak_idx
        while right_idx < len(measured) and measured[right_idx] > 0.01 * peak_val:
            right_idx += 1

        fit_start, fit_end = left_idx, right_idx

    return (model_vals[fit_start:fit_end] - measured[fit_start:fit_end])



#def fit(x0, u, y, irf, fit_start=0, fit_end=-1):
#    """
#    Fit measurement values `y` and independent variable `u` to parameter vector `x`
#
#    :param x0: parameter vector containing initial estimates.
#    :param u: values of the independent variable.
#    :param y: measurement values.
#
#    Scipy's lm algorithm uses the implementation in J. J. More, “The Levenberg-Marquardt Algorithm: Implementation and Theory,” Numerical Analysis, ed. G. A. Watson, Lecture Notes in Mathematics 630, Springer Verlag, pp. 105-116, 1977.
#    """
#    res = least_squares(lambda x, u, y: fun_residual(x, u, irf, y, n1=1, n2=1.41, rho=0, s=18e3, fit_start=fit_start, fit_end=fit_end, phantom='slab', mua_independent=True, m=400), x0, method='lm', args=(u, y), verbose=1)
#    return res

#def fit_files(irf_files, data_files):
#    if not (isinstance(irf_files, tuple) and isinstance(data_files)):
#        raise ValueError(f'irf_files and data_files arguments must be tuples.\n')
#    # Get data from irf and data files
#    irfs = [Data.from_file(fn).data for fn in irf_files]
#    measurements = [Data.from_file(fn).data for fn in data_files]
#
#    # Fit measured data (data_files) to time convolution of an irf and the parameterised DE
#    data = zip(irfs,measurements)
#    #for i,m in data:
#
#    # Return mu_a and mu_s' values that get the closest fit

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return (idx, array[idx])

def slidingavg(vector, N):
    # TODO: consider uniform filter 1D version
    cumsum = np.cumsum(np.insert(vector, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

# TODO: preprocess measured data separately. reuse preprocessed irf curves
def preprocess(
    measured, irf,
    meas_noise_win=(1,1), irf_noise_win=(1,1),
    meas_roi=(1,1), irf_roi=(1,1),
    meas_avg_w=1, irf_avg_w=1,
    normalization="none"  # "none", "peak", "area"
):
    """
    Preprocess measured and irf curves:
    - background subtraction
    - ROI masking
    - optional normalization (none, peak, area)
    """
    bg = (
        np.mean(irf[irf_noise_win[0]:irf_noise_win[1]]),
        np.mean(measured[meas_noise_win[0]:meas_noise_win[1]])
    )

    # Remove background
    irf_bg = irf - bg[0]
    meas_bg = measured - bg[1]

    # # perform sliding average
    # irf_avg = slidingavg(irf_bg, irf_avg_w)
    # irf_bg_corrected = irf_avg/max(irf_avg)
    # meas_avg_bg = slidingavg(meas_bg, meas_avg_w)
    # meas_bg_corrected = meas_avg_bg/max(meas_avg_bg) #Get rid of sliding average for now

    # ROI mask
    irf_bg[:irf_roi[0]] = 0
    irf_bg[irf_roi[1]:] = 0
    meas_bg[:meas_roi[0]] = 0
    meas_bg[meas_roi[1]:] = 0

    # ----- Normalization logic -----
    EPS = 1e-12
    if normalization.lower() == "peak":
        irf_bg = irf_bg / max(np.max(irf_bg), EPS)
        meas_bg = meas_bg / max(np.max(meas_bg), EPS)

    elif normalization.lower() == "area":
        irf_area = max(np.sum(irf_bg[irf_roi[0]:irf_roi[1]]), EPS)
        meas_area = max(np.sum(meas_bg[meas_roi[0]:meas_roi[1]]), EPS)
        irf_bg = irf_bg / irf_area * meas_area
        meas_bg = meas_bg / meas_area * meas_area  # → normalize to 1 area

    # else "none" → just return bg-subtracted

    return meas_bg, irf_bg


def fit_least_squares(
    x0,
    meas,
    irf,
    time_arr,
    rho=0,
    s=1,
    n1=1,
    n2=1.41,
    phantom='semiinf',
    mua_independent=True,
    m=200,
    geometry=GEOMETRY.TRANSMITTANCE,
    offset=0,
    meas_noise_win=None,
    irf_noise_win=None,
    meas_roi=None,
    irf_roi=None,
    meas_avg_w=3,
    irf_avg_w=3,
    fit_start=None,
    fit_end=None,
    verbose=1,
    smart_crop=False,
    normalization="none",
    optimize_shift=True,
    interp_factor=1
):
    """
    Fit measurement values to a convolutional diffusion model using least squares.

    :param x0: Initial guess for [mua, musp]
    :param meas: Measured histogram (1D list or array)
    :param irf: Instrument response function (1D list or array)
    :param time_step_ns: Time step between histogram bins in nanoseconds
    :param Other params: Passed to `fun_residual` and `model`
    :return: res, to get mua and musp, use res.x[0] and res.x[1]
    """
    # Step 1: Convert to NumPy arrays
    meas = np.array(meas)
    irf = np.array(irf)

    meas = interpolate_curve(meas, interp_factor)
    irf = interpolate_curve(irf, interp_factor)
    # Rescale time axis
    time_arr = np.linspace(0, len(meas) * (time_arr[1]-time_arr[0]) / interp_factor, len(meas))

    if meas_noise_win is None or irf_noise_win is None:
        noise_win_percent = 0.08  # Use last 8% of the curve for noise window
        noise_win_len = int(len(meas) * noise_win_percent)
        meas_noise_win = (len(meas) - noise_win_len, len(meas))
        irf_noise_win = (len(irf) - noise_win_len, len(irf))

    if meas_roi is None or irf_roi is None:
        roi = (0, len(meas))
        meas_roi = roi
        irf_roi = roi

    # Step 2: Preprocess measured and irf
    y, irf = preprocess(
        meas,
        irf=irf,
        meas_noise_win=meas_noise_win,
        irf_noise_win=irf_noise_win,
        meas_roi=meas_roi,
        irf_roi=irf_roi,
        meas_avg_w=meas_avg_w,
        irf_avg_w=irf_avg_w,
        normalization=normalization
    )

    # Remove negative values by setting them to zero
    y[y < 0] = 0
    irf[irf < 0] = 0

    # Step 3: Define time axis
    u = time_arr

    # Auto-detect fit range if not provided
    if fit_start is None or fit_end is None:
        fit_start = int(len(y) * 0.10)
        fit_end = int(len(y) * 0.90)

    # Step 4: Perform fitting using least squares
    fit = least_squares(
        lambda x, u, y: fun_residual(
            x, u, irf, y,
            rho=rho, n1=n1, n2=n2,
            fit_start=fit_start,
            fit_end=fit_end,
            mua_independent=mua_independent,
            phantom=phantom,
            s=s, m=m,
            geometry=geometry,
            offset=offset,
            smart_crop=smart_crop,
            optimize_shift= optimize_shift
        ),
        x0,
        method='trf',
        # method='lm',
        args=(u, y),
        verbose=verbose,
        bounds=([0, 0.1, -50*interp_factor], [1000, 1000, 50*interp_factor])
    )

    print(f"Fit result: mua = {fit.x[0]:.5e}, musp = {fit.x[1]:.5e}, shift = {fit.x[2]:.2f} bins")
    return fit

from scipy.interpolate import interp1d
def apply_offset(model_vals, offset):
    """
    Apply a continuous shift to model_vals by interpolating.
    Positive offset -> shift right (later in time)
    Negative offset -> shift left (earlier in time)
    """
    x = np.arange(len(model_vals))
    f = interp1d(x, model_vals, bounds_error=False, fill_value=0.0)
    shifted = f(x - offset)   # shift curve by "offset" bins
    return shifted

from scipy.interpolate import interp1d

def interpolate_curve(curve, factor=4):
    """
    Interpolate a curve by a given factor (default=4).
    len=128 → len=128*factor.
    """
    if factor <= 1 or curve is None or len(curve) == 0:
        return curve
    x = np.arange(len(curve))
    f = interp1d(x, curve, kind="linear")
    new_x = np.linspace(0, len(curve)-1, len(curve)*factor)
    return f(new_x)


# #commenting this out as idk how Data works ngl
# def fit_from_variables(vs):
#     """
#     Do the fit process once using the values in variable store vs
#     """
#     irf_curve_idx = int(vs['COMMANDER']['Code']['Current']) % int(vs['COMMANDER']['Files']['System']['Codes Per File'])
#     meas_curve_idx = int(vs['COMMANDER']['Code']['Current']) % int(vs['COMMANDER']['Files']['Data']['Codes Per File'])

#     irfpath  = Path(str(vs['COMMANDER']['Files']['System']['Root Directory'])) / ( str(vs['COMMANDER']['Files']['System']['Name']) + str(vs['COMMANDER']['Files']['System']['Suffix']) )
#     # TODO: add capacity for multiple file formats
#     irf = Data.from_file(irfpath).data[:,irf_curve_idx,0,0]

#     datapath = Path(str(vs['COMMANDER']['Files']['Data']['Root Directory'])) / ( str(vs['COMMANDER']['Files']['Data']['Name']) + str(vs['COMMANDER']['Files']['Data']['Suffix']))
#     y = Data.from_file(datapath).data[:,meas_curve_idx,0,0]
#     x0 = np.array([float(vs['VARIABLES']['FIT']['Initial']['mu_a'])*1e-4,float(vs['VARIABLES']['FIT']['Initial']['mu_s\''])*1e-4])
#     u = np.linspace(0, len(y)*float(vs['COMMANDER']['Time']['Channel Factor']), len(y))

#     y, irf = preprocess(y, irf
#             , (int(vs['COMMANDER']['Background']['Data']['First Channel'])  , int(vs['COMMANDER']['Background']['Data']['Last Channel']))
#             , (int(vs['COMMANDER']['Background']['System']['First Channel']), int(vs['COMMANDER']['Background']['System']['Last Channel']))
#             , (int(vs['COMMANDER']['Region of Interest']['Data']['First Channel'])  , int(vs['COMMANDER']['Region of Interest']['Data']['Last Channel']))
#             , (int(vs['COMMANDER']['Region of Interest']['System']['First Channel']), int(vs['COMMANDER']['Region of Interest']['System']['Last Channel']))
#             ,  int(vs['COMMANDER']['Background']['Data']['Average Width'])  , int(vs['COMMANDER']['Background']['System']['Average Width']))
    
#     # Finds indices within the measured data corresponding to fractions of the peak
#     meas_peak_idx = y.argmax()
#     meas_peak_val = y.max()
#     meas_left_idx, meas_left_val  = find_nearest(y[:meas_peak_idx], float(vs['COMMANDER']['Fitting Range']['First Fraction'])*meas_peak_val)
#     meas_right_idx, meas_right_val = find_nearest(y[meas_peak_idx:], float(vs['COMMANDER']['Fitting Range']['Last Fraction'])*meas_peak_val)
#     meas_right_idx += meas_peak_idx

#     if int(vs['COMMANDER']['Fitting Range']['First Channel']) < int(vs['COMMANDER']['Fitting Range']['Last Channel']):
#         fit_first = int(vs['COMMANDER']['Fitting Range']['First Channel']) 
#         fit_last  = int(vs['COMMANDER']['Fitting Range']['Last Channel']) 
#         logging.info(f"Using COMMANDER/Fitting Range/First&Last Channels as fit range of {fit_first}:{fit_last}")
#     else:
#         fit_first = meas_left_idx
#         fit_last = meas_right_idx
#         logging.info(f"Using COMMANDER/Fitting Range/First&Last Fractions to calculate fit range of {fit_first}:{fit_last}")

#     res = least_squares(
#             lambda x, u, y: fun_residual(x, u, irf, y
#                 , n1=vs['PARAMETERS']['Constants']['n0'], n2=vs['PARAMETERS']['Constants']['n_ext'], rho=vs['PARAMETERS']['Geometry']['Radial Position']*1e3
#                 , s=vs['PARAMETERS']['Geometry']['Slab Thickness']*1e3
#                 , fit_start=fit_first, fit_end=fit_last
#                 , phantom=vs['PARAMETERS']['General']['Medium'], mua_independent=bool(vs['PARAMETERS']['General']['Diffusion Coeff. Mu_a independent'])
#                 , m=int(vs['PARAMETERS']['General']['No. Imaginary Sources'])
#                 , geometry=GEOMETRY[str(vs['PARAMETERS']['General']['Geometry'])]
#                 , offset=int(vs['VARIABLES']['FIT']['Initial']['t_0 Channels']))
#             , x0, method='lm', args=(u, y), verbose=1)
#     return (res, meas_left_idx, meas_right_idx)


if __name__ == '__main__':
    import sys
    # irf = Data.from_file(sys.argv[1]) 
    # meas = Data.from_file(sys.argv[2])

    # Generate synthetic Gaussian-like data for irf and meas
    ### CHANGED
    x = np.linspace(0, 127, 128)
    irf = np.random.normal(loc=64, scale=15, size=128)
    irf = np.exp(-0.5 * ((x - 64) / 15) ** 2) * 1000  # Gaussian shape
    irf += np.random.normal(0, 30, 128)  # Add some noise
    irf = irf.astype(int)
    irf = np.array(irf)
    print(f"IRF: {irf}") 

    meas = np.exp(-0.5 * ((x - 70) / 18) ** 2) * 1200  # Slightly shifted and wider
    meas += np.random.normal(0, 40, 128)
    meas = meas.astype(int)
    meas = np.array(meas)
    print(f"Measured: {meas}")
    ### CHANGED


    expected = np.array([0.1030505*1e-4,5.433029*1e-4])   # Expected final value
    x0 = np.array([0.1*1e-4,5*1e-4])   # Initial estimate of variables changing during fit


    # y = meas.data[:,0,0,0]              # Dependent variable (measured data)
    # u = np.linspace(0,4096*2.03,4096)   # Independent variable (time)

    time_step_ns = 0.19  # based on your earlier deduction
    u = np.linspace(0, len(meas) * time_step_ns, len(meas)) ### CHANGED


    # y, irf = preprocess(y, irf.data[:,0,0,0], (400, 570), (400, 570), 3, 3)
    # y, irf = preprocess(meas, irf, (400, 570), (400, 570), 3, 3)

    y, irf = preprocess(
        meas, irf,
        meas_noise_win=(0, 10),     # early bins for background noise estimate
        irf_noise_win=(0, 10),
        meas_roi=(0, 128),          # full range of bins
        irf_roi=(0, 128),
        meas_avg_w=3,
        irf_avg_w=3
    ) ### CHANGED


    # Find peak of measured data ∴ indices of 80% and 1% values
    meas100idx = y.argmax()
    meas100 = y.max()
    meas80idx, _ = find_nearest(y[:meas100idx], 0.8*meas100)
    meas1idx, _ = find_nearest(y[meas100idx:], 0.01*meas100)
    #res = fit(x0,u,y,irf.data[:,0,0,0],fit_start=meas80idx,fit_end=meas1idx)
    #res = fit(x0,u,y,irf,fit_start=550,fit_end=3250)

    # res = least_squares(lambda x, u, y: fun_residual(x, u, irf, y, n1=1, n2=1.41, rho=0, s=18e3, fit_start=550, fit_end=3250, phantom='slab', mua_independent=True, m=400, geometry=GEOMETRY.TRANSMITTANCE, offset=20), x0, method='lm', args=(u, y), verbose=1)
    res = least_squares(lambda x, u, y: fun_residual(x, u, irf, y, n1=1, n2=1.41, rho=0, s=18e3, fit_start=10, fit_end=100, phantom='slab', mua_independent=True, m=400, geometry=GEOMETRY.TRANSMITTANCE, offset=20), x0, method='lm', args=(u, y), verbose=1) ### CHANGED
    print(f"Res: {res.x}")

    #theoretical = Contini1997([0],u,18e3,expected[0],expected[1],1,1.41,'slab',True,400)['total'][1][0]
    #plt.plot(u, theoretical/max(theoretical), 'k-', label='Transmittance (normalised)')
    #import scipy.io
    #scipy.io.savemat('theoretical_T.mat', dict(x=u, y=theoretical))


    given_params = model(irf,[0],u,18*1e3,expected[0],expected[1],1,1.41,'slab',True,400)
    plt.semilogy(u,given_params[0:len(u)], 'r-', label=f"Expected: (mu_a, mu_s') {expected*1e4} cm^-1")
    #plt.plot(u,fun_residual(expected, u, irf, y, [0], 1, 1.41, s=18e3, fit_start=0, fit_end=len(u), phantom='slab'))
    modelled = model(irf,[0],u,18*1e3,res.x[0],res.x[1],1,1.41,'slab',True,200)
    plt.semilogy(u,modelled[0:len(u)], 'g-', label=f"Fitted: (mu_a, mu_s') {res.x*1e4} cm^-1")
    plt.semilogy(u,y, '.', label='Measured data')
    plt.title('Wavelength 600 nm; Offset of 21 cells')

    plt.legend()
    plt.xlabel("Time ps")
    plt.ylabel("Counts ph")
    plt.show()
