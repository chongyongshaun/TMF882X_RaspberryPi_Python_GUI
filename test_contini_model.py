from enum import IntEnum

from diffusion_equation.diffusion_equation import Contini1997
from diffusion_equation.fit import GEOMETRY, convolve_irf_with_model, fit_least_squares, model

import pandas as pd
import csv
import os
import numpy as np

if __name__ == "__main__":
    mu_list = [
        (1.24, 188.37),
        (2.67, 92.50),
        (0.61, 241.29),
        (2.03, 289.43),
        (1.78, 66.91),
        (0.83, 138.75),
        (2.48, 271.64),
        (0.94, 210.86),
        (1.35, 59.02),
        # (1.37, 45.01),
        (2.89, 172.18)
    ]


    mu_list_divided_by_100 = [(round(mua / 100, 4), round(musp / 100, 4)) for mua, musp in mu_list]
    print("mu_list_divided_by_100:", mu_list_divided_by_100)


    phantom_type = ["slab", "semiinf"]

    params = {
        'rho': '15',
        'time_step (ns)': '0.004', #4e-12
        'num_bins': '4096',
        's': '1',
        'mua': '0.01', # mm^{-1}
        'musp': '1', # mm^{-1}
        'n1': '1',
        'n2': '1.41',
        'phantom': 'semiinf',
        'mua_independent': 'True',
        'm': '200',
        'geometry': GEOMETRY.REFLECTANCE,  # Measurement geometry
        't' : None, # this is calculated from time_step and num_bins
    }

    time_step = float(params['time_step (ns)'])
    num_bins = int(params['num_bins'])
    # t = [1e-9 if i == 0 else time_step * i for i in range(num_bins)] # don't use 0 for time bin 0 otherwise it would cause error
    t = [time_step * i for i in range(num_bins)] # don't use 0 for time bin 0 otherwise it would cause error
    params['t'] = t

    rho = float(params['rho'])
    t = params['t']
    s = float(params['s'])
    mua = float(params['mua'])
    musp = float(params['musp'])
    n1 = float(params['n1'])
    n2 = float(params['n2'])
    phantom = params['phantom']
    mua_independent = params['mua_independent'].lower() == 'true'
    m = int(params['m'])

    irf_file_path_4096  = "example csv/4096_Real_Time_Data/IRF/irf_800nm.csv"
    meas_file_path_4096_1  = "example csv/4096_Real_Time_Data/MEASUREMENT/B5_15mm_800nm.csv"
    meas_file_path_4096_2  = "example csv/4096_Real_Time_Data/MEASUREMENT/C5_15mm_800nm.csv"

    dir_name = "C:/Users/USER/Desktop/work related/Tyndall Intern/FLIM_Keela/output/"

    # Load IRF and measurement data from CSV files (single column, 4096 rows, no headers, newline-separated)
    irf = np.loadtxt(irf_file_path_4096)
    meas1 = np.loadtxt(meas_file_path_4096_1)
    meas2 = np.loadtxt(meas_file_path_4096_2)
    irf = irf.astype(int)
    meas1 = meas1.astype(int)
    meas2 = meas2.astype(int)
    print("IRF shape:", irf.shape)
    print("Measurement 1 shape:", meas1.shape)
    print("Measurement 2 shape:", meas2.shape)


    """ run the fitting algo, same set of mua and musp, all semiinf, different measurement data"""
    x0 = np.array([0.025, 2])
    res1 = fit_least_squares(
        x0,
        meas1,
        irf,
        time_arr=t,
        rho=rho,
        s=s,
        n1=n1,
        n2=n2,
        phantom=phantom,
        mua_independent=mua_independent,
        m=m,
        geometry=GEOMETRY.REFLECTANCE,
        fit_start=None,
        fit_end=None,
        verbose=1,
        # smart_crop=True
    )

    res2 = fit_least_squares(
        x0,
        meas2,
        irf,
        time_arr=t,
        rho=rho,
        s=s,
        n1=n1,
        n2=n2,
        phantom=phantom,
        mua_independent=mua_independent,
        m=m,
        geometry=GEOMETRY.REFLECTANCE,
        fit_start=None,
        fit_end=None,
        verbose=1
        # smart_crop=True
    )

    print('initial guess:', x0)
    print("Fitting result for measurement 1:", res1)
    print("Fitting result for measurement 2:", res2)

    """ compare convolved semiinf with """
    mua1, musp1 = res1[0], res1[1]
    output = Contini1997([rho], t, s, mua1, musp1, n1, n2, phantom, mua_independent, m)["total"][0][0]
    model_conv_res1 = convolve_irf_with_model(irf, output, geometry=GEOMETRY.REFLECTANCE, offset=0, normalize_irf=True, normalize_model=True, denest_contini_output=False)
    mua2, musp2 = res2[0], res2[1]
    output = Contini1997([rho], t, s, mua2, musp2, n1, n2, phantom, mua_independent, m)["total"][0][0]
    model_conv_res2 = convolve_irf_with_model(irf, output, geometry=GEOMETRY.REFLECTANCE, offset=0, normalize_irf=True, normalize_model=True, denest_contini_output=False)


    # normalize irf and mode
    norm_model_1 = model_conv_res1 / model_conv_res1.max()
    norm_model_2 = model_conv_res2 / model_conv_res2.max()
    norm_irf = irf / irf.max()
    norm_meas1 = meas1 / meas1.max()
    norm_meas2 = meas2 / meas2.max()

    # Calculate the difference between normalized model and measurement for both datasets
    diff_1 = norm_model_1 - norm_meas1
    diff_2 = norm_model_2 - norm_meas2

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # Subplot 1: Model 1 vs Measurement 1

    axs[0].plot(norm_model_1, label='Normalized Model (Convolved) 1')
    axs[0].plot(norm_meas1, label='Normalized Measurement B5_15mm_800nm')
    axs[0].plot(diff_1, label='difference')
    axs[0].set_ylabel('Normalized Intensity')
    axs[0].set_title(f'Normalized Model vs Measurement B5_15mm_800nm mua={mua1:.4f}, musp={musp1:.4f}')
    axs[0].legend()

    # Subplot 2: Model 2 vs Measurement 2
    axs[1].plot(norm_model_2, label='Normalized Model (Convolved) 2')
    axs[1].plot(norm_meas2, label='Normalized Measurement C5_15mm_800nm')
    axs[1].plot(diff_2, label='difference')
    axs[1].set_xlabel('Time Bin')
    axs[1].set_ylabel('Normalized Intensity')
    axs[1].set_title(f'Normalized Model vs Measurement C5_15mm_800nm mua={mua2:.4f}, musp={musp2:.4f}')
    axs[1].legend()

    plt.tight_layout()
    plt.show()


    """ run python contini with 10 different set of mua and musp values, and save the results to a csv file """
    # file_name = "python_contini_master_3.csv"

    # for mua, musp in mu_list_divided_by_100:
    #     for phantom in phantom_type:
    #         print(f"Fitting for mua: {mua}, musp: {musp}, phantom: {phantom}")
    #         output = Contini1997([rho], t, s, mua, musp, n1, n2, phantom, mua_independent, m) 
    #         result = output["total"][0][0] #just get the TRANSMITTENCE for the first rho
    #         csv_file = os.path.join(dir_name, file_name)
    #         write_header = not os.path.exists(csv_file)

    #         with open(csv_file, mode='a', newline='') as file:
    #             writer = csv.writer(file)
    #             if write_header:
    #                 header = ['mua', 'musp', 'phantom'] + [f'bin_{i}' for i in range(len(result))]
    #                 writer.writerow(header)
    #             row = [mua, musp, phantom] + list(result)
    #             writer.writerow(row)




    """ run python convolution with 10 different set"""
    # for mua, musp in mu_list_divided_by_100:
    #     output = Contini1997([rho], t, s, mua, musp, n1, n2, "slab", mua_independent, m) 
    #     convolved = convolve_irf_with_model(
    #         irf,
    #         output,
    #         geometry=GEOMETRY.REFLECTANCE,
    #         offset=0,
    #         normalize_irf=True,
    #         normalize_model=False,
    #         denest_contini_output=True
    #     )
        
    #     csv_file = os.path.join(dir_name, "python_convolved_master_norm_irf.csv")
    #     write_header = not os.path.exists(csv_file)
    #     with open(csv_file, mode='a', newline='') as file:
    #         writer = csv.writer(file)
    #         if write_header:
    #             header = ['mua', 'musp', "phantom"] + [f'bin_{i}' for i in range(len(convolved))]
    #             writer.writerow(header)
    #         row = [mua, musp, "slab"] + list(convolved)
    #         writer.writerow(row)





    """ run matlab convolution with 10 different set"""
    # matlab_contini_df = pd.read_csv("example csv/matlab_contini_master_2.csv")
    # # Loop through each row of the DataFrame
    # for _, row in matlab_contini_df.iterrows():
    #     mua = row[0]
    #     musp = row[1]
    #     phantom = row[2]
    #     output = row[3:].to_numpy()  # histogram data (4096 bins)

    #     # Perform the convolution
    #     convolved = convolve_irf_with_model(
    #         irf,
    #         output,
    #         geometry=GEOMETRY.REFLECTANCE,
    #         offset=0,
    #         normalize_irf=True,
    #         normalize_model=True,
    #         denest_contini_output=False # we use the matlab contini directly, no need to denest
    #     )

    #     # Prepare to save the output
    #     csv_file = os.path.join(dir_name, "matlab_convolved_master_norm_irf_model.csv")
    #     write_header = not os.path.exists(csv_file)

    #     with open(csv_file, mode='a', newline='') as file:
    #         writer = csv.writer(file)
    #         if write_header:
    #             header = ['mua', 'musp', "phantom"] + [f'bin_{i}' for i in range(len(convolved))]
    #             writer.writerow(header)
    #         row_to_write = [mua, musp, phantom] + list(convolved)
    #         writer.writerow(row_to_write)

