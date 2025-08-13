import threading
import time

import numpy as np

from diffusion_equation.fit import fit_least_squares
from test_contini_model import GEOMETRY

class FittingWorker(threading.Thread):
    def __init__(self, data_queue, get_settings_fn, irf=None, result_callback=None, interval=1.0):
        """
        :param data_queue: Queue containing latest measurement data (128 bins per array)
        :param get_settings_fn: Function to retrieve current model settings from GUI (e.g., model_panel.get_settings)
        :param irf: The loaded IRF array to use in the model fitting
        :param result_callback: Function to call with fitting results (optional)
        :param interval: Time interval between fits in seconds
        """
        super().__init__(daemon=True)
        self.data_queue = data_queue
        self.get_settings_fn = get_settings_fn
        self.irf = irf
        self.result_callback = result_callback
        self.interval = interval
        self.filter_largest_peak = True

        self.running = True

    def run(self):
        while self.running:
            try:
                latest_data = None
                while not self.data_queue.empty():
                    latest_data = self.data_queue.get()

                if latest_data is not None:
                    # place holder code for fitting logic
                    settings = self.get_settings_fn()
                    print("[FittingWorker] Performing fitting with settings:", settings)

                    # process latest_data, convert string to numpy array
                    parts = latest_data.strip().split(";")
                    values = np.array(list(map(int, parts[1:])))
                    print(f"[FittingWorker] Latest data values: {values}")

                    if self.filter_largest_peak:
                        from peak_filtering import filter_largest_peak
                        values = filter_largest_peak(values)
                    fit_result = self.perform_fit(values, settings, self.irf)

                    if self.result_callback:
                        self.result_callback(fit_result)

            except Exception as e:
                print(f"[FittingWorker] Error during fitting: {e}")

            time.sleep(self.interval)


    def perform_fit(self, measured_data, settings, irf):
        """
        fitting logic
        """
        x0 = np.array([0.025, 2])
        rho = float(settings['rho'])
        t = settings['t']
        s = float(settings['s'])
        n1 = float(settings['n1'])
        n2 = float(settings['n2'])
        phantom = settings['phantom']
        mua_independent = settings['mua_independent'].lower() == 'true'
        m = int(settings['m'])
        res = fit_least_squares(
            x0,
            measured_data,
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
        mua, musp = res[0], res[1]
        return {"mua": mua, "musp": musp}

    def toggle_filter_largest_peak(self):
        """
        Toggle the filter_largest_peak setting.
        """
        self.filter_largest_peak = not self.filter_largest_peak
        print(f"[FittingWorker] Filter largest peak set to: {self.filter_largest_peak}")
        
    def stop(self):
        self.running = False
