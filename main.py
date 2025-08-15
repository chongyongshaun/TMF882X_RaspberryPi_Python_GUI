
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 12:22:32 2025

@author: keela
"""

import queue
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os

from TMF8828PiDataReader import DataReader
from contini_model_panel import ContiniModelPanel
from diffusion_equation.diffusion_equation import Contini1997
from diffusion_equation.fit import convolve_irf_with_model
from fitting_worker import FittingWorker
from test_contini_model import GEOMETRY

# -------------- TMF8828 Raspberry Pi Class --------------    
class TMF8828RaspberryPiGUI:
    def __init__(self, root):
        self.root = root
        self.root.winfo_toplevel().title("TMF8828 Raspberry Pi Measurement")

        self.frame = ttk.Frame(root)
        self.frame.pack(fill="both", expand=True)

        self.plot_data_queue = queue.Queue()
        self.selected_channels = set() 
        # self.selected_channels.add(1)  # Default to channel 1
        self.status_queue = queue.Queue()
        self.fit_data_queue = queue.Queue()
        self.reader = DataReader(self.plot_data_queue, self.fit_data_queue, self.selected_channels, self.status_queue)

        self.fitting_worker = None

        # Create notebook inside main frame
        self.notebook = ttk.Notebook(self.frame)
        self.notebook.pack(fill="both", expand=True)

        # Two tabs: left and right content
        self.left_frame = ttk.Frame(self.notebook)
        self.right_frame = ttk.Frame(self.notebook)

        self.notebook.add(self.left_frame, text="Control & Model")
        self.notebook.add(self.right_frame, text="Graph")
        
        #options:
        self.save_fitting = False
        self.normalize_graph_output = tk.BooleanVar(value=True)  # Default to True 
        self.EPS = 1e-6  # small positive number

        # top left corner
        self.model_frame = tk.LabelFrame(self.left_frame, text="Model", padx=5, pady=5, bg="white")
        self.model_frame.pack(fill="both", expand=True, padx=5, pady=5)
        # bottom left corner
        self.control_frame = tk.LabelFrame(self.left_frame, text="Control Panel", padx=5, pady=5, bg="white")
        self.control_frame.pack(fill="both", expand=True, padx=5, pady=5)
        # top right corner
        self.graph_frame = tk.LabelFrame(self.right_frame, text="Graph", padx=5, pady=5, bg="white")
        self.graph_frame.pack(fill="both", expand=True, padx=5, pady=5)


        self.build_control_panel()
        self.build_channel_selector()
        self.build_model_panel()

        self.status_labels = {}
        self.build_status_display()
        self.update_status_display()
        self.build_graph_frame_misc()

        # self.fig, self.ax = plt.subplots()
        # self.bars = self.ax.bar(np.arange(128), np.zeros(128))
        # self.ax.set_ylim(0, 100)

        # self.fig, self.ax = plt.subplots()
        self.fig, (self.ax, self.residual_ax) = plt.subplots(2, 1, figsize=(6, 6), sharex=True, height_ratios=[3, 1])
        self.x_data = np.arange(128)
        self.y_data = np.zeros(128)
        self.line, = self.ax.plot(self.x_data, self.y_data, color='blue', label='Measurement Curve')
        self.model_line, = self.ax.plot(self.x_data, np.zeros_like(self.x_data), color='red', linestyle='--', label='Fitted Model')
        self.ax.legend()
        self.ax.set_ylim(0, 100)
        self.ax.set_xlim(0, 127)
        self.ax.set_xlabel("Time Bins")
        self.ax.set_ylabel("Intensity")
        self.ax.set_title("Time of Flight Graph")
        self.residual_line, = self.residual_ax.plot(self.x_data, np.zeros_like(self.x_data), color='green', label='Residual')
        self.residual_ax.axhline(0, color='gray', linestyle='--')
        self.residual_ax.set_ylabel("Residual")
        self.residual_ax.set_xlabel("Time Bins")
        self.residual_ax.legend()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
        self.canvas.get_tk_widget().grid(row=1, column=0, padx=5, pady=5)

        self.update_plot()

    def start_fitting_worker(self):
        if not self.contini_model_panel.get_irf():
            messagebox.showerror("Error", "No IRF loaded. Please load an IRF before starting live fitting.")
            return
        # Starts the fitting worker if not already running.
        if self.fitting_worker is None:
            self.fitting_worker = FittingWorker(
                data_queue=self.fit_data_queue,
                get_settings_fn=self.contini_model_panel.get_settings,
                irf=self.contini_model_panel.irf,  # Assuming loaded
                result_callback=self.handle_fit_result,
                interval=0.5  # every 1 second
            )
            self.fitting_worker.start()
            print("Fitting worker started.")
            self.filter_meas_checkbox.config(state=tk.NORMAL)  # Initially disabled

    #callback function for FittingWorker, if you want to do something with the fit result this function will be called whenever a fit result is available
    def handle_fit_result(self, fit_result): 

        print("Live Fit Result:", fit_result) 
        # update the fit result label
        self.fit_result_var.set(f"Fit Result: μa: {fit_result['mua']:.4f}, μs': {fit_result['musp']:.4f}")

        # create the fitting curve using the model
        mua, musp = fit_result["mua"], fit_result["musp"]
        settings = self.contini_model_panel.get_settings()
        rho = float(settings['rho'])
        t = settings['t']
        s = float(settings['s'])
        n1 = float(settings['n1'])
        n2 = float(settings['n2'])
        phantom = settings['phantom']
        mua_independent = settings['mua_independent'].lower() == 'true'
        m = int(settings['m'])
        output = Contini1997([rho], t, s, mua, musp, n1, n2, phantom, mua_independent, m)["total"][0][0]
        model_conv = convolve_irf_with_model(self.contini_model_panel.irf, output, geometry=GEOMETRY.REFLECTANCE, offset=0, normalize_irf=True, normalize_model=True, denest_contini_output=False)
        

        if self.save_fitting and self.file_path_var.get():
            #save raw model_conv to a csv file
            file_path = self.file_path_var.get()
            with open(file_path, 'a') as f:
                # Convert each value in model_conv to string and join with commas
                csv_line = "#FIT;" + ';'.join(str(val) for val in model_conv)
                f.write(csv_line + '\n')
        if self.normalize_graph_output.get(): #normalize before plotting
            model_conv = model_conv/max(model_conv) 
        if self.use_log_scale.get():
            EPS = 1e-6
            model_conv = np.clip(model_conv, EPS, None)  # Avoid zero in log scale
        self.model_line.set_ydata(model_conv)

        if len(model_conv) == len(self.y_data):
            residual = self.y_data - model_conv
            self.residual_line.set_ydata(residual)
            self.residual_ax.set_ylim(residual.min() * 1.1, residual.max() * 1.1)
        
        self.canvas.draw()
        print("Model line updated with new fit result.")

    def update_plot(self):
        while not self.plot_data_queue.empty():
            line = self.plot_data_queue.get()
            parts = line.strip().split(";")
            if len(parts) == 129 and parts[0].startswith("#HLONG"):
                try:
                    values = np.array(list(map(int, parts[1:])))
                    # for bar, val in zip(self.bars, values): #this one is for bar graph
                    #     bar.set_height(val)
                    # self.ax.set_ylim(0, values.max() * 1.1)
                    if self.normalize_graph_output.get():  # Normalize if the option is selected
                        values = values / max(values) 
                    values[values <= 0] = self.EPS
                    self.y_data = values
                    self.line.set_ydata(self.y_data)

                    if self.use_log_scale.get():
                        EPS = 1e-6
                        values = np.clip(values, EPS, None)  # Avoid zero in log scale

                    self._safe_adjust_ylim() 
                    self.canvas.draw()
                except Exception as e:
                    print(f"Error: {e}")
        self.root.after(100, self.update_plot)
    
    """
    Status Display Methods
    """
    def build_status_display(self):
        status_frame = tk.LabelFrame(self.graph_frame, text="Sensor Status", padx=5, pady=5, bg="white")
        status_frame.grid(row=2, column=0, padx=5, pady=5, sticky="w")

        status_fields = [
            "Timestamp", "Iterations", "Threshold", "SPAD Map ID",
            "Measurement Range", "Operation Mode", "De-scattering", "Short Range Mode"
        ]

        # Top row: first 5 fields
        for idx, field in enumerate(status_fields[:5]):
            label = tk.Label(status_frame, text=f"{field}: N/A", anchor="w", bg="white")
            label.grid(row=0, column=idx, sticky="w", padx=5)
            self.status_labels[field] = label

        # Bottom row: remaining 4 fields
        for idx, field in enumerate(status_fields[5:]):
            label = tk.Label(status_frame, text=f"{field}: N/A", anchor="w", bg="white")
            label.grid(row=1, column=idx, sticky="w", padx=5)
            self.status_labels[field] = label

    def update_status_display(self):
        while not self.status_queue.empty():
            line = self.status_queue.get()
            if line.startswith("#ITT"):
                self.parse_and_update_status(line)

        # Schedule next check
        self.root.after(1000, self.update_status_display)  # update every 1 second

    def parse_and_update_status(self, line):
        parts = line.strip().split(";")
        if len(parts) != 9:
            return  # Invalid format

        status_values = {
            "Timestamp": parts[1],
            "Iterations": parts[2],
            "Threshold": parts[3],
            "SPAD Map ID": parts[4],
            "Measurement Range": parts[5],
            "Operation Mode": parts[6],
            "De-scattering": "Enabled" if parts[7] == "1" else "Disabled",
            "Short Range Mode": "Enabled" if parts[8] == "1" else "Disabled"
        }

        for key, value in status_values.items():
            self.status_labels[key].config(text=f"{key}: {value}")

    """
    Other options on the graph frame Methods
    """
    def build_graph_frame_misc(self):
        # Create a StringVar to hold the dynamic text
        self.fit_result_var = tk.StringVar()
        self.fit_result_var.set("Fit Result: μa: ---, μs': ---")
        # Label bound to the StringVar
        fit_label = tk.Label(self.graph_frame, textvariable=self.fit_result_var, font=("Arial", 14))
        fit_label.grid(row=3, column=0, padx=5, pady=5, sticky="w")

        plotting_options_frame = tk.LabelFrame(self.graph_frame, text="Plotting Options", padx=5, pady=5, bg="white")
        plotting_options_frame.grid(row=4, column=0, padx=5, pady=5, sticky="w")
        # Normalize output checkbox
        normalize_checkbox = tk.Checkbutton(
            plotting_options_frame,
            text="Normalize Graph Output",
            variable=self.normalize_graph_output,
            bg="white"
        )
        normalize_checkbox.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.filter_meas_checkbox = tk.Checkbutton(
            plotting_options_frame,
            text="Filter Largest Peak and Noise before fitting",
            bg="white",
            command=lambda: self.fitting_worker.toggle_filter_largest_peak() if self.fitting_worker else None,
        )
        self.filter_meas_checkbox.grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.filter_meas_checkbox.config(state=tk.DISABLED)  # Initially disabled
        self.use_log_scale = tk.BooleanVar(value=False)
        log_checkbox = tk.Checkbutton(
            plotting_options_frame,
            text="Use Logarithmic Y-Axis",
            variable=self.use_log_scale,
            bg="white",
            command=self.toggle_log_scale
        )
        log_checkbox.grid(row=2, column=0, padx=5, pady=5, sticky="w")


    def toggle_log_scale(self):
        """Toggle between linear and logarithmic Y-axis."""
        if self.use_log_scale.get():
            self.ax.set_yscale("log")
            self.residual_ax.set_yscale("log")
        else:
            self.ax.set_yscale("linear")
            self.residual_ax.set_yscale("linear")
        
        self._safe_adjust_ylim()
        self.canvas.draw_idle()

    def _safe_adjust_ylim(self):
        EPS = 1e-6  # Small positive number
        ymin = EPS if self.ax.get_yscale() == "log" else 0

        # Measurement curve
        y_max_meas = max(self.y_data.max(), EPS)
        self.ax.set_ylim(ymin, y_max_meas * 1.1)

        # Residual plot
        y_resid = self.residual_line.get_ydata()
        if len(y_resid) > 0:
            y_min_resid = min(y_resid.min(), -EPS)
            y_max_resid = max(y_resid.max(), EPS)
            if self.residual_ax.get_yscale() == "log":
                # Residuals can be negative, so for log scale we take absolute values
                y_max_abs = max(abs(y_min_resid), abs(y_max_resid), EPS)
                self.residual_ax.set_ylim(EPS, y_max_abs * 1.1)
            else:
                self.residual_ax.set_ylim(y_min_resid * 1.1, y_max_resid * 1.1)



    """
    Channel Selector Methods
    These methods handle the creation, destruction, and management of the channel selector frame.
    """
    def build_channel_selector(self):
        self.channel_frame = tk.LabelFrame(self.control_frame, text="TDC Channel", padx=5, pady=5, bg="white")
        self.channel_frame.grid(row=0, column=3, rowspan=6,columnspan=3, padx=5, pady=5)

        # Canvas and scrollbar setup
        canvas = tk.Canvas(self.channel_frame, height=120, bg="white")
        scrollbar = tk.Scrollbar(self.channel_frame, orient="vertical", command=canvas.yview)
        self.inner_frame = tk.Frame(canvas, bg="white")
        self.inner_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.inner_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # apply button to update the selected channels
        apply_btn = tk.Button(self.inner_frame, text="Apply", command=self.apply_selected_channels, bg="white", fg="black")
        apply_btn.pack(pady=(10, 0))
        
        # Checkboxes for channels
        self.channel_vars = []
        for i in range(10):
            var = tk.IntVar()
            chk = tk.Checkbutton(self.inner_frame, text=f"Channel {i}", variable=var, bg="white")
            chk.pack(anchor="w")
            self.channel_vars.append(var)

    def destroy_channel_selector(self):
        """Destroys the channel selector frame and its widgets"""
        for widget in self.channel_frame.winfo_children():
            widget.destroy()
        self.channel_frame.destroy()

    def get_selected_channels(self):
        """Returns a list of selected channel numbers"""
        return [i for i, var in enumerate(self.channel_vars) if var.get()]
    
    def apply_selected_channels(self):
        """Updates the selected channels in data reader with the current selection (channel_vars)"""
        self.selected_channels.clear()
        self.selected_channels.update(self.get_selected_channels())
        print(f"Selected channels updated: {sorted(self.selected_channels)}")

        # self.build_graphs_for_selected_channels()


    """
    Control Frame Methods
    These methods handle the creation, destruction, and management of the control frame.
    """
    def build_control_panel(self):
        meas_settings_frame = tk.LabelFrame(self.control_frame, text="Measurement Settings", padx=5, pady=5, bg="white")
        meas_settings_frame.grid(row=0, column=0, rowspan=6,columnspan=3, padx=5, pady=5)

        # Iterations
        tk.Label(meas_settings_frame, text="Iterations (0-65535):", bg="white").grid(row=0, column=0, sticky="w")
        self.iterations_var = tk.IntVar()
        tk.Entry(meas_settings_frame, textvariable=self.iterations_var, width=10).grid(row=0, column=1)
        tk.Button(meas_settings_frame, text="Set", command=self.set_iterations).grid(row=0, column=2)

        # Threshold
        tk.Label(meas_settings_frame, text="Threshold (0-255):", bg="white").grid(row=1, column=0, sticky="w")
        self.threshold_var = tk.IntVar()
        tk.Entry(meas_settings_frame, textvariable=self.threshold_var, width=10).grid(row=1, column=1)
        tk.Button(meas_settings_frame, text="Set", command=self.set_threshold).grid(row=1, column=2)

        # Short Range Mode
        tk.Label(meas_settings_frame, text="Short Range Mode (0-Off, 1-On):", bg="white").grid(row=2, column=0, sticky="w")
        self.short_range_var = tk.IntVar()
        tk.Entry(meas_settings_frame, textvariable=self.short_range_var, width=10).grid(row=2, column=1)
        tk.Button(meas_settings_frame, text="Set", command=self.set_short_range).grid(row=2, column=2)

        # Operation Mode
        tk.Label(meas_settings_frame, text="Operation Mode (0-3):", bg="white").grid(row=3, column=0, sticky="w")
        self.operation_mode_var = tk.IntVar()
        tk.Entry(meas_settings_frame, textvariable=self.operation_mode_var, width=10).grid(row=3, column=1)
        tk.Button(meas_settings_frame, text="Set", command=self.set_operation_mode).grid(row=3, column=2)

        # Histogram Mode
        tk.Label(meas_settings_frame, text="Histogram Mode (0-3):", bg="white").grid(row=4, column=0, sticky="w")
        self.histogram_mode_var = tk.IntVar()
        tk.Entry(meas_settings_frame, textvariable=self.histogram_mode_var, width=10).grid(row=4, column=1)
        tk.Button(meas_settings_frame, text="Set", command=self.set_histogram_mode).grid(row=4, column=2)
        
        # ROW 6: Start Reader and Toggle Measurement buttons and take one measurement button
        self.start_reader_button = tk.Button(self.control_frame, text="Connect", command=self.start_reader)
        self.start_reader_button.grid(row=6, column=0, padx=5, pady=5)   
        self.toggle_measurement_button = tk.Button(self.control_frame, text="Toggle Measurement", command=self.reader.toggle_measurement, state=tk.DISABLED)
        self.toggle_measurement_button.grid(row=6, column=1, padx=5, pady=5)
        tk.Button(self.control_frame, text="Start Live Fitting", command=self.start_fitting_worker).grid(row=6, column=2, padx=5, pady=5)

        tk.Label(self.control_frame, text="Take N measurements:", bg="white").grid(row=6, column=3, sticky="w")
        self.n_measurements_var = tk.IntVar(value=1)
        tk.Entry(self.control_frame, textvariable=self.n_measurements_var, width=5).grid(row=6, column=4)
        tk.Button(
            self.control_frame,
            text="Take N",
            command=lambda: self.reader.take_n_measurements(self.n_measurements_var.get())
        ).grid(row=6, column=5, padx=5, pady=5)

        # ROW 7: File selection and saving options and start live fitting button
        self.file_path_var = tk.StringVar()
        self.file_path_var.set("No file selected")
        self.file_path_label = tk.Label(self.control_frame, textvariable=self.file_path_var, bg="white")
        self.file_path_label.grid(row=7, column=0, padx=5, pady=5, sticky="w")
        self.file_path_button = tk.Button(self.control_frame, text="Select File", command=self.select_file)
        self.file_path_button.grid(row=7, column=1, padx=5, pady=5)
        self.save_meas_checkbox = tk.Checkbutton(self.control_frame, text="Save Measurement to CSV", bg="white", command=self.reader.toggle_saving_to_csv)
        self.save_meas_checkbox.grid(row=7, column=2, padx=5, pady=5)
        self.save_fit_checkbox = tk.Checkbutton(self.control_frame, text="Save Fitting Curve to CSV", bg="white", command=self.toggle_saving_fitting)
        self.save_fit_checkbox.grid(row=7, column=3, padx=5, pady=5)
    
    def toggle_saving_fitting(self):
        """Toggle the saving of fitting data to a CSV file."""
        self.save_fitting = not self.save_fitting
        if self.save_fitting:
            print("Saving fitting data to CSV is enabled.")
            if not self.file_path_var.get():
                messagebox.showwarning("Warning", "Please select a file to save the fitting data.")
        else:
            self.file_path_var.set("Fitting data will not be saved.")
    
    def start_reader(self):
            self.reader.start()
            self.start_reader_button.config(state=tk.DISABLED)
            self.toggle_measurement_button.config(state=tk.NORMAL)

    def select_file(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.file_path_var.set(os.path.basename(file_path))
            self.reader.set_file_path(os.path.basename(file_path))
        else:
            self.file_path_var.set("No file selected")

    def set_iterations(self):
        value = self.iterations_var.get()
        self.reader.set_number_of_iterations(value)

    def set_threshold(self):
        value = self.threshold_var.get()
        self.reader.set_object_detection_threshold(value)

    def set_short_range(self):
        value = self.short_range_var.get()
        self.reader.set_short_range_mode(value)

    def set_operation_mode(self):
        value = self.operation_mode_var.get()
        self.reader.set_operation_mode(value)

    def set_histogram_mode(self):
        value = self.histogram_mode_var.get()
        self.reader.set_histogram_mode(value)

    """
    Contini Model Panel Frame Methods
    These methods handle the creation, destruction, and management of the model panel frame.
    """
    def build_model_panel(self):
        self.contini_model_panel = ContiniModelPanel(self.model_frame)
        panel_frame = self.contini_model_panel.get_frame()
        panel_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

    
# -------------- Main Application --------------
class MainApp:
    def __init__(self, root):
        self.root = root
        self.root.title("TMF882X Raspberry Pi GUI")

        # notebook = ttk.Notebook(root)
        # self.tmf8828_rasp_tab = ttk.Frame(notebook)

        # notebook.add(self.tmf8828_rasp_tab, text="TMF882X Raspberry Pi Measurement")
        # notebook.pack(expand=True, fill="both")

        # TMF8828RaspberryPiGUI(self.tmf8828_rasp_tab)
        self.tmf8828_rasp_gui = TMF8828RaspberryPiGUI(root)

if __name__ == "__main__":
    #Run app
    root = tk.Tk()
    app = MainApp(root)
    root.mainloop()