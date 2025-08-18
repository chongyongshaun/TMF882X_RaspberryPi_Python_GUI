import csv
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter import filedialog
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from diffusion_equation.diffusion_equation import Contini1997
from diffusion_equation.fit import convolve_irf_with_model, model, GEOMETRY

# add log scale to both figures
class ContiniModelPanel:
    """
    A class to represent the Contini model panel for diffusion equations.
    Provides a Tkinter GUI frame to input parameters and compute results.
    """
    def __init__(self, root):
        self.root = root
        self.main_frame = ttk.Frame(root)

        # Create left panel for inputs
        self.input_frame = ttk.Frame(self.main_frame)
        self.input_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        # Create right panel for plots
        self.plot_frame = ttk.Frame(self.main_frame)
        self.plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.irf = None
        self.contini = None
        self.convolved = None
        # Create 3 labeled plots
        self.irf_ax, self.irf_canvas = self._create_plot_frame(self.plot_frame, "IRF")
        self.contini_ax, self.contini_canvas = self._create_plot_frame(self.plot_frame, "Contini")
        self.convolved_ax, self.convolved_canvas = self._create_plot_frame(self.plot_frame, "Convolved")

        self.params = {
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
            'meas_noise_win': '(100,120)', 
            'irf_noise_win': '(100,120)',  
        }

        self.entries = {} # dictionary to hold entry widgets, to access their values just call self.entries['label'].get() e.g label 'rho' will be self.entries['rho'].get()
        
        self.create_widgets() # create the widgets in the user entry panel (label + entry)

        self.apply_settings() # do it first time so it can calculate t, fit_start, and fit_end

        self.contini_button = ttk.Button(self.input_frame, text="Contini", command=self.compute_contini)
        self.contini_button.grid(row=len(self.params), column=0, columnspan=1, pady=10)

        self.compute_convolution_button = ttk.Button(self.input_frame, text="Contini & Convolve IRF", command=self.compute_convolution_with_irf)
        self.compute_convolution_button.grid(row=len(self.params), column=1, columnspan=1, pady=10)

        load_irf_button = ttk.Button(self.input_frame, text="Load IRF from CSV", command=self.load_irf)
        load_irf_button.grid(row=len(self.params)+1, column=0, columnspan=1, pady=10)

        self.y_scale = "linear"
        self.toggle_scale_button = ttk.Button(self.input_frame, text="Toggle Y Scale (Linear/Log)", command=self.toggle_y_scale)
        self.toggle_scale_button.grid(row=len(self.params)+1, column=1, columnspan=1, pady=10)

        self.apply_settings_button = ttk.Button(self.input_frame, text="Apply Settings", command=self.apply_settings)
        self.apply_settings_button.grid(row=len(self.params)+2, column=0, columnspan=1, pady=10)

    def create_widgets(self):
        # rho
        ttk.Label(self.input_frame, text="Source-Detector Separation (rho, mm)").grid(row=0, column=0, sticky='w', padx=5, pady=5)
        entry_rho = ttk.Entry(self.input_frame)
        entry_rho.insert(0, self.params['rho'])
        entry_rho.grid(row=0, column=1, padx=5, pady=5)
        self.entries['rho'] = entry_rho

        # time_step (ns)
        ttk.Label(self.input_frame, text="Time Step (ns)").grid(row=1, column=0, sticky='w', padx=5, pady=5)
        entry_time_step = ttk.Entry(self.input_frame)
        entry_time_step.insert(0, self.params['time_step (ns)'])
        entry_time_step.grid(row=1, column=1, padx=5, pady=5)
        self.entries['time_step (ns)'] = entry_time_step

        # num_bins
        ttk.Label(self.input_frame, text="Number of Time Bins").grid(row=2, column=0, sticky='w', padx=5, pady=5)
        entry_num_bins = ttk.Entry(self.input_frame)
        entry_num_bins.insert(0, self.params['num_bins'])
        entry_num_bins.grid(row=2, column=1, padx=5, pady=5)
        self.entries['num_bins'] = entry_num_bins

        # s (slab thickness mm)
        ttk.Label(self.input_frame, text="Slab Thickness (s, mm)").grid(row=3, column=0, sticky='w', padx=5, pady=5)
        entry_s = ttk.Entry(self.input_frame)
        entry_s.insert(0, self.params['s'])
        entry_s.grid(row=3, column=1, padx=5, pady=5)
        self.entries['s (slab thickness mm)'] = entry_s

        # mua
        ttk.Label(self.input_frame, text="Absorption Coefficient (mua, mm⁻¹)").grid(row=4, column=0, sticky='w', padx=5, pady=5)
        entry_mua = ttk.Entry(self.input_frame)
        entry_mua.insert(0, self.params['mua'])
        entry_mua.grid(row=4, column=1, padx=5, pady=5)
        self.entries['mua'] = entry_mua

        # musp
        ttk.Label(self.input_frame, text="Reduced Scattering Coefficient (musp, mm⁻¹)").grid(row=5, column=0, sticky='w', padx=5, pady=5)
        entry_musp = ttk.Entry(self.input_frame)
        entry_musp.insert(0, self.params['musp'])
        entry_musp.grid(row=5, column=1, padx=5, pady=5)
        self.entries['musp'] = entry_musp

        # n1 (external n)
        ttk.Label(self.input_frame, text="External Refractive Index (n1)").grid(row=6, column=0, sticky='w', padx=5, pady=5)
        entry_n1 = ttk.Entry(self.input_frame)
        entry_n1.insert(0, self.params['n1'])
        entry_n1.grid(row=6, column=1, padx=5, pady=5)
        self.entries['n1 (external n)'] = entry_n1

        # n2 (diffusing n)
        ttk.Label(self.input_frame, text="Diffusing Refractive Index (n2)").grid(row=7, column=0, sticky='w', padx=5, pady=5)
        entry_n2 = ttk.Entry(self.input_frame)
        entry_n2.insert(0, self.params['n2'])
        entry_n2.grid(row=7, column=1, padx=5, pady=5)
        self.entries['n2 (diffusing n)'] = entry_n2

        # phantom
        ttk.Label(self.input_frame, text="Phantom Type").grid(row=8, column=0, sticky='w', padx=5, pady=5)
        phantom_combo = ttk.Combobox(self.input_frame, values=["slab", "semiinf"], state="readonly")
        phantom_combo.set(self.params['phantom'])  # default
        phantom_combo.grid(row=8, column=1, padx=5, pady=5)
        self.entries['phantom'] = phantom_combo

        # mua_independent
        ttk.Label(self.input_frame, text="Mua Independent? (True or False)").grid(row=9, column=0, sticky='w', padx=5, pady=5)
        entry_mua_independent = ttk.Entry(self.input_frame)
        entry_mua_independent.insert(0, self.params['mua_independent'])
        entry_mua_independent.grid(row=9, column=1, padx=5, pady=5)
        self.entries['mua_independent (True/False)'] = entry_mua_independent

        # m (num imaginary sources)
        ttk.Label(self.input_frame, text="Number of Imaginary Sources (m)").grid(row=10, column=0, sticky='w', padx=5, pady=5)
        entry_m = ttk.Entry(self.input_frame)
        entry_m.insert(0, self.params['m'])
        entry_m.grid(row=10, column=1, padx=5, pady=5)
        self.entries['m (num imaginary sources)'] = entry_m

        # geometry
        ttk.Label(self.input_frame, text="Measurement Geometry").grid(row=11, column=0, sticky='w', padx=5, pady=5)
        geom_combo = ttk.Combobox(self.input_frame, values=["REFLECTANCE", "TRANSMITTANCE"], state="readonly")
        # Store it as string, you can later map it back to GEOMETRY enum
        geom_combo.set("REFLECTANCE" if self.params['geometry'] == GEOMETRY.REFLECTANCE else "TRANSMITTANCE")
        geom_combo.grid(row=11, column=1, padx=5, pady=5)
        self.entries['geometry'] = geom_combo

        # fit_start
        ttk.Label(self.input_frame, text="Fit Start Bin (auto, or insert number)").grid(row=12, column=0, sticky='w', padx=5, pady=5)
        entry_fit_start = ttk.Entry(self.input_frame)
        entry_fit_start.insert(0, self.params['fit_start'])
        entry_fit_start.grid(row=12, column=1, padx=5, pady=5)
        self.entries['fit_start'] = entry_fit_start

        # fit_end
        ttk.Label(self.input_frame, text="Fit End Bin (auto, or insert number)").grid(row=13, column=0, sticky='w', padx=5, pady=5)
        entry_fit_end = ttk.Entry(self.input_frame)
        entry_fit_end.insert(0, self.params['fit_end'])
        entry_fit_end.grid(row=13, column=1, padx=5, pady=5)
        self.entries['fit_end'] = entry_fit_end

        # smart crop as a checkbox
        # self.smart_crop_var = tk.BooleanVar(value=self.params['smart_crop'].lower() == "true")
        # def on_smart_crop_toggle():
        #     self.params['smart_crop'] = "True" if self.smart_crop_var.get() else "False"
        # smart_crop_checkbox = ttk.Checkbutton(
        #     self.input_frame,
        #     text="Smart Crop",
        #     variable=self.smart_crop_var,
        #     command=on_smart_crop_toggle
        # )
        # smart_crop_checkbox.grid(row=14, column=0, columnspan=2, sticky='w', padx=5, pady=5)
        ttk.Label(self.input_frame, text="Smart Crop (True or False)").grid(row=14, column=0, sticky='w', padx=5, pady=5)
        entry_smart_crop = ttk.Entry(self.input_frame)
        entry_smart_crop.insert(0, self.params['smart_crop'])
        entry_smart_crop.grid(row=14, column=1, padx=5, pady=5)
        self.entries['smart_crop'] = entry_smart_crop

        # irf noise window
        ttk.Label(self.input_frame, text="IRF Noise Window (start,end)").grid(row=15, column=0, sticky='w', padx=5, pady=5)
        entry_irf_noise = ttk.Entry(self.input_frame)
        entry_irf_noise.insert(0, self.params['irf_noise_win'])
        entry_irf_noise.grid(row=15, column=1, padx=5, pady=5)
        self.entries['irf_noise_win'] = entry_irf_noise

        # meas noise window
        ttk.Label(self.input_frame, text="Measurement Noise Window (start,end)").grid(row=16, column=0, sticky='w', padx=5, pady=5)
        entry_meas_noise = ttk.Entry(self.input_frame)
        entry_meas_noise.insert(0, self.params['meas_noise_win'])
        entry_meas_noise.grid(row=16, column=1, padx=5, pady=5)
        self.entries['meas_noise_win'] = entry_meas_noise

        # --- add phantom type callback ---
        def on_phantom_change(event=None):
            phantom = phantom_combo.get().lower()
            if phantom == "semiinf":
                entry_s.config(state="disabled")
                entry_m.config(state="disabled")
            else:
                entry_s.config(state="normal")
                entry_m.config(state="normal")

        phantom_combo.bind("<<ComboboxSelected>>", on_phantom_change)
        on_phantom_change()  # call once to apply at startup

        # Add a checkbox for "Save to CSV"
        self.save_to_csv = tk.BooleanVar(value=False)
        save_csv_checkbox = ttk.Checkbutton(
            self.input_frame,
            text="Save to CSV",
            variable=self.save_to_csv
        )
        save_csv_checkbox.grid(row=17, column=1, columnspan=2, sticky='w', padx=5, pady=5)


    def _create_plot_frame(self, parent, title):
        """Create a labeled frame with an embedded empty matplotlib plot."""
        frame = ttk.LabelFrame(parent, text=title)
        frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        fig = Figure(figsize=(3, 1.7), dpi=100)
        ax = fig.add_subplot(111)
        ax.set_xlabel("Time Bin")
        ax.set_ylabel("Value")
        ax.text(0.5, 0.5, 'No data yet', transform=ax.transAxes,
                ha='center', va='center', fontsize=12, color='gray')

        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        return ax, canvas

    def toggle_y_scale(self):
        """Toggle y-axis between linear and log for all 3 plots."""
        self.y_scale = "log" if self.y_scale == "linear" else "linear"

        # Apply to all three axes
        for ax, canvas in [(self.irf_ax, self.irf_canvas),
                        (self.contini_ax, self.contini_canvas),
                        (self.convolved_ax, self.convolved_canvas)]:
            ax.set_yscale(self.y_scale)
            # prevent errors if log and no positive values
            ymin, ymax = ax.get_ylim()
            if self.y_scale == "log":
                if ymin <= 0:
                    ymin = 1e-4
                ax.set_ylim(bottom=ymin, top=ymax)
            canvas.draw()

    #getters for settings and the computed results as well as irf
    def get_settings(self):
        return self.params
    def get_irf(self):
        return self.irf
    def get_contini(self):
        return self.contini
    def get_convolved(self):
        return self.convolved

    def apply_settings(self):
        """
        Applies the settings from the entries to the params dictionary.
        This is useful if you want to change the default values of the parameters.
        """
        for key in self.params.keys():
            if key in self.entries:
                self.params[key] = self.entries[key].get()

        # Dynamically calculate 't', 'fit_start', and 'fit_end'
        try:
            time_step = float(self.params['time_step (ns)'])
            num_bins = int(self.params['num_bins'])
            # t = [1e-9 if i == 0 else time_step * i for i in range(num_bins)] # don't use 0 for time bin 0 otherwise it would cause error
            t = [time_step * i for i in range(num_bins)] # don't use 0 for time bin 0 otherwise it would cause error
            self.params['t'] = t

            smart_crop_entry = self.entries['smart_crop'].get().strip().lower()
            if smart_crop_entry == "true":
                self.params['fit_start'] = '0'
                self.params['fit_end'] = str(num_bins - 1)
            else: 
                # Handle fit_start
                fit_start_entry = self.entries['fit_start'].get().strip().lower()
                # If "auto" is not in the entry, we try to parse it as an integer, limit it to (0, num_bins)
                if "auto" not in fit_start_entry:
                    try:
                        fit_start_val = int(fit_start_entry)
                        self.params['fit_start'] = str(max(0, min(fit_start_val, num_bins))) 
                    except Exception:
                        self.params['fit_start'] = str(max(0, int(0.08 * num_bins)))
                else:
                    self.params['fit_start'] = str(max(0, int(0.08 * num_bins)))

                # Handle fit_end
                fit_end_entry = self.entries['fit_end'].get().strip().lower()
                if "auto" not in fit_end_entry:
                    try:
                        fit_end_val = int(fit_end_entry)
                        self.params['fit_end'] = str(max(0, min(fit_end_val, num_bins)))
                    except Exception:
                        self.params['fit_end'] = str(min(num_bins - 1, int(0.78 * num_bins)))
                else:
                    self.params['fit_end'] = str(min(num_bins - 1, int(0.78 * num_bins)))
        except Exception:
            self.params['t'] = None
            self.params['fit_start'] = '10'
            self.params['fit_end'] = '100'
        finally:
            #DEBUG: print the parameters to check
            for k, v in self.params.items():
                print(f"{k}: {v}")

    def compute_contini(self):
        """
        handles contini button, reads parameters from entries, computes the Contini1997 model
        Return Value: A dictionary with only the "total" key, which contains 2 arrays, both array size are (1, num_bins)
        and the first array is the reflectance and the second array is the transmittance.
        """
        try:
            # Get parameters from self.params
            rho = float(self.params['rho'])
            t = self.params['t']
            s = float(self.params['s'])
            mua = float(self.params['mua'])
            musp = float(self.params['musp'])
            n1 = float(self.params['n1'])
            n2 = float(self.params['n2'])
            phantom = self.params['phantom']
            mua_independent = self.params['mua_independent'].lower() == 'true'
            m = int(self.params['m'])

            theoretical_model = Contini1997([rho], t, s, mua, musp, n1, n2, phantom, mua_independent, m)
            self.contini = theoretical_model  # Store the result for later use
            
            result = theoretical_model["total"]

            self.contini_ax.clear()
            self.contini_ax.set_title("Contini - Reflectance")
            self.contini_ax.set_xlabel("Time Bin")
            self.contini_ax.set_ylabel("Value")
            self.contini_ax.plot(result[0][0], label="Reflectance")
            self.contini_ax.legend()
            self.contini_canvas.draw()

            if not self.save_to_csv.get():
                # plot both result[0][0] and result[1][0] as subplots in one figure
                import matplotlib.pyplot as plt
                fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
                axs[0].plot(result[0][0])
                axs[0].set_title("Reflectance AKA result[0][0]")
                axs[0].set_ylabel("Value")
                if phantom == "slab":
                    axs[1].plot(result[1][0])
                    axs[1].set_title("Transmittance AKA result[1][0]")
                    axs[1].set_xlabel("Time Bin")
                    axs[1].set_ylabel("Value")
                plt.tight_layout()
                plt.show()
            else:
                # Save the result to a CSV file
                file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
                if file_path:
                    with open(file_path, 'w', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(['Reflectance', 'Transmittance'])
                        for r, t in zip(result[0][0], result[1][0]):
                            writer.writerow([r, t])
                    messagebox.showinfo("Success", f"Results saved to {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to compute: {e}")

    def compute_convolution_with_irf(self):
        try:
            if self.irf is None:
                messagebox.showerror("Error", "Please load an IRF first.")
                return
            convolved = None
            if self.contini is None:
                # Get parameters from self.params
                rho = float(self.params['rho'])
                t = self.params['t']
                s = float(self.params['s'])
                mua = float(self.params['mua'])
                musp = float(self.params['musp'])
                n1 = float(self.params['n1'])
                n2 = float(self.params['n2'])
                phantom = self.params['phantom']
                mua_independent = self.params['mua_independent'].lower() == 'true'
                m = int(self.params['m'])

                convolved = model(self.irf, [rho], t, s, mua, musp, n1, n2, phantom, mua_independent, m, GEOMETRY.REFLECTANCE, offset=0)
                convolved = convolved[:int(self.params["num_bins"])]  # Truncate to match measurement bins (e.g. output is 255, and input is 128, so we truncate to 128)
            else: # contini is already computed
                convolved = convolve_irf_with_model(self.irf, self.contini, geometry=GEOMETRY.REFLECTANCE, offset=0, normalize_irf=True, normalize_model=True, denest_contini_output=True)
            
            self.convolved = convolved  # Store the convolved result for later use
            #update the convolved plot
            self.convolved_ax.clear()
            self.convolved_ax.set_title("Convolved Result (Reflectance)")
            self.convolved_ax.set_xlabel("Time Bin")
            self.convolved_ax.set_ylabel("Value")
            self.convolved_ax.plot(convolved, color='green')
            self.convolved_canvas.draw()

            if not self.save_to_csv.get():
                #plot the convolved result
                import matplotlib.pyplot as plt
                plt.figure(figsize=(8, 4))
                plt.plot(convolved)
                plt.title("Convolved Result (Reflectance)")
                plt.xlabel("Time Bin")
                plt.ylabel("Value")
                plt.tight_layout()
                plt.show()
            else:
                file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
                if file_path:
                    with open(file_path, 'w', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        for val in convolved:
                            writer.writerow([val])
                    messagebox.showinfo("Success", f"Convolved result saved to {file_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to compute: {e}")
        
    def load_irf(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return

        try:
            with open(file_path, 'r') as file:
                reader = csv.reader(file)
                first_row = next(reader)
                if first_row and first_row[0].startswith("#HLONG"):
                    first_row = first_row[0].split(';')
                    self.irf = [float(value) for value in first_row[1:] if value.strip()]
                else:
                    self.irf = [float(value) for value in first_row]
                print(f"self.irf: {self.irf}")
                # Update the IRF plot
                self.irf_ax.clear()
                self.irf_ax.set_title("IRF")
                self.irf_ax.set_xlabel("Time Bin")
                self.irf_ax.set_ylabel("Value")
                self.irf_ax.plot(range(len(self.irf)), self.irf, color='blue')
                self.irf_canvas.draw()
            print("IRF Loaded", f"IRF loaded with {len(self.irf)} time bins.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load IRF: {e}")

    def get_frame(self):
        return self.main_frame
    


if __name__ == '__main__':
    pass
    root = tk.Tk()
    root.title("Contini1997 Model Panel")
    panel = ContiniModelPanel(root)
    panel.get_frame().pack(padx=10, pady=10)
    root.mainloop()