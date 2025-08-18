import numpy as np
from diffusion_equation.diffusion_equation import Contini1997
from diffusion_equation.fit import GEOMETRY


params = {
            'rho': '5',
            # 'time_step (ns)': '0.004', #4e-12
            # 'num_bins': '4096',
            'time_step (ns)': '0.38',
            'num_bins': '128',
            's': '1000',
            'mua': '0.01', # mm^{-1}
            'musp': '1', # mm^{-1}
            'n1': '1',
            'n2': '1.41',
            'phantom': 'semiinf',
            'mua_independent': 'True',
            'm': '200',
            'geometry': GEOMETRY.REFLECTANCE,  # Measurement geometry
            't' : None, # this is calculated from time_step and num_bins
            'fit_start': '10',  # Start bin for fitting
            'fit_end': '30',   # End bin for fitting These should be dynamically calculated based on the length of the time array, 10-100 assumes 128 bins
            'smart_crop': 'False',  # Smart crop option 80% of y to the left of peak, 1% of y to the right of peak
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


output = Contini1997([rho], t, s, mua, musp, n1, n2, phantom, mua_independent, m)["total"][0][0]
import matplotlib.pyplot as plt

plt.plot(t, output)
plt.xlabel('Time (ns)')
plt.ylabel('Output')
plt.yscale('log')  # Set y-axis to logarithmic scale
plt.plot(t, output, marker='o', linestyle='-')
plt.grid(True)
plt.show()
