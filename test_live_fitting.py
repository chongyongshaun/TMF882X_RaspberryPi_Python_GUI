import numpy as np
from diffusion_equation.fit import GEOMETRY, fun_residual, preprocess
from scipy.optimize import least_squares
from scipy.interpolate import interp1d
import csv


# Define a parameter dictionary (can be populated from GUI) THESE ARE DEFAULTS
# for reference, these are the units
# :param phantom: Either semi-infinite or slab ('semiinf' or 'slab'). If the former, then `s` will be set to 'inf' and if the latter then s must be specified.
#     :param rho: radial position of the detector (mm)
#     :param t: time (ns)
#     :param s: slab thickness (mm)

default_non_changing_residual_params = {
    'n1': 1.0,        # Refractive index 1
    'n2': 1.41,       # Refractive index 2
    'rho': 15,       # Source-detector separation (mm)
    's': 1,        # Slab thickness (mm)

    # Dynamically calculate fit_start and fit_end based on the length of measured
    'fit_start': int(0.1 * 1280),  # First 10% of measured bins
    'fit_end': int(0.9 * 1280),    # Last 10% of measured bins

    'phantom': 'semiinf',# Geometry type
    'mua_independent': True,  # Diffusion coefficient flag
    'm': 400,         # Number of imaginary sources
    'geometry': GEOMETRY.REFLECTANCE,  # Measurement geometry
}


def fit_mua_musp_live(irf, measured, time_arr):

    noise_win_percent = 0.08  # Use first 8% of the curve for noise window
    noise_win_len = int(len(measured) * noise_win_percent)

    pre_meas, pre_irf = preprocess(
        measured, irf,
        meas_noise_win=(0, noise_win_len),
        irf_noise_win=(0, noise_win_len),
        meas_roi=(0, len(measured)),
        irf_roi=(0, len(irf)),
        meas_avg_w=3,
        irf_avg_w=3
    )

    # Interpolate to increase resolution to 1280 bins
    num_bins_highres = 10*len(time_arr)  # 10x resolution
    time_arr_highres = np.linspace(min(time_arr), max(time_arr), num_bins_highres)

    interp_meas = interp1d(time_arr, pre_meas, kind='cubic', fill_value="extrapolate")
    interp_irf = interp1d(time_arr, pre_irf, kind='cubic', fill_value="extrapolate")

    pre_meas_highres = interp_meas(time_arr_highres)
    pre_irf_highres = interp_irf(time_arr_highres)

    # Remove negative values by setting them to zero
    pre_meas_highres[pre_meas_highres < 0] = 0
    pre_irf_highres[pre_irf_highres < 0] = 0

    x0 = np.array([0.1e-4, 5e-4])  # initial guess
    # Fitting call
    fit = least_squares(
        fun_residual_fixed_params,  # Our parameterized function
        x0,               # Initial guess [mua, musp] 1st param of the fun_residual_fixed_params function
        method='lm',
        args=(time_arr, pre_irf_highres, pre_meas_highres, default_non_changing_residual_params),  # these are the other parameters following the x0 parameter in the fun_residual_fixed_params function
        verbose=1
    )

    return fit.x  # fitted [mua, musp]

def fun_residual_fixed_params(x, time, irf, measured, params):
    """Residual function using parameter dictionary"""
    return fun_residual(
        x, 
        time, 
        irf, 
        measured,
        n1=params['n1'],
        n2=params['n2'],
        rho=params['rho'],
        s=params['s'],
        fit_start=params['fit_start'],
        fit_end=params['fit_end'],
        phantom=params['phantom'],
        mua_independent=params['mua_independent'],
        m=params['m'],
        geometry=params['geometry']
    )

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    """
    Do Fitting once with example data
    """
    # Example usage
    # Load IRF data
    with open('example csv/irf_1_extracted.csv', 'r') as f_irf:
        irf = np.array([float(val) for val in f_irf.read().strip().split(",")])

    # Load measured data
    with open('example csv/B3_2.0_extracted.csv', 'r') as f_meas:
        measured = np.array([float(val) for val in f_meas.read().strip().split(",")])
    # time bin 0 should not be used because it can't divide by zero
    
    time_step = 0.19  # Example time step in nanoseconds
    # time_step = 0.004  # Example time step in nanoseconds
    num_bins = 128
    time_arr = [1e-9 if i == 0 else time_step * i for i in range(num_bins)] #dont use 0 for time bin 0


    fit = fit_mua_musp_live(irf, measured, time_arr)
    print(fit)



    """
    Fit all of the data in master list
    """
    results = []
    with open('example csv/master_measurement_list.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header       
        
        for row in reader:
            file_name = row[0]
            try:
                measured = np.array([float(val) for val in row[1:129]])
                # Load IRF data (assuming same IRF for all, or adjust as needed)
                with open('example csv/irf_1_extracted.csv', 'r') as f_irf:
                    irf = np.array([float(val) for val in f_irf.read().strip().split(",")])
                time_step = 0.19
                # time_step = 0.004
                num_bins = 128
                time_arr = [1e-9 if i == 0 else time_step * i for i in range(num_bins)]
                fit_result = fit_mua_musp_live(irf, measured, time_arr)
                print(f"{file_name}: {fit_result}")
                results.append((file_name, fit_result))
            except Exception as e:
                print(f"{file_name}: Error during fitting - {e}")

    # Save results to results.txt
    with open('results_semiinf.txt', 'w') as f_out:
        f_out.write("File Name, mua, musp\n")
        for file_name, fit_result in results:
            mua, musp = fit_result
            f_out.write(f"{file_name}, {mua}, {musp}\n")


    """
    Plotting the results for debugging
    """
    noise_win_percent = 0.08  # Use first 8% of the curve for noise window
    noise_win_len = int(len(measured) * noise_win_percent)
    
    pre_meas, pre_irf = preprocess(
        measured, irf,
        meas_noise_win=(0, noise_win_len),
        irf_noise_win=(0, noise_win_len),
        meas_roi=(0, len(measured)),
        irf_roi=(0, len(irf)),
        meas_avg_w=3,
        irf_avg_w=3
    )

    # Interpolate to increase resolution to 1280 bins
    num_bins_highres = 1280
    time_arr_highres = np.linspace(min(time_arr), max(time_arr), num_bins_highres)

    interp_meas = interp1d(time_arr, pre_meas, kind='cubic', fill_value="extrapolate")
    interp_irf = interp1d(time_arr, pre_irf, kind='cubic', fill_value="extrapolate")

    pre_meas_highres = interp_meas(time_arr_highres)
    pre_irf_highres = interp_irf(time_arr_highres)

    # Remove negative values by setting them to zero
    pre_meas_highres[pre_meas_highres < 0] = 0
    pre_irf_highres[pre_irf_highres < 0] = 0


    #DEBUG PLOT
    # The Original Data, no processing
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.plot(time_arr, irf, label='IRF')
    plt.plot(time_arr, measured, label='Measured')
    plt.legend()
    plt.title("Raw Data")

    # The Processed Data, after preprocessing and interpolation and removing negatives
    plt.subplot(122)
    plt.plot(time_arr_highres[300:500], pre_irf_highres[300:500], label='Processed IRF', marker='o', linestyle='-')
    plt.plot(time_arr_highres[300:500], pre_meas_highres[300:500], label='Processed Measured', marker='o', linestyle='-')
    plt.legend()
    plt.title("Processed Data")
    plt.show()

    #TODO: test different initial guesses
    initial_guesses = [
        [0.05e-4, 1e-4],   # Low scattering
        [0.1e-4, 5e-4],     # Original
        [0.2e-4, 10e-4],    # Higher scattering
        [0.5e-4, 20e-4]     # Very high scattering
    ]