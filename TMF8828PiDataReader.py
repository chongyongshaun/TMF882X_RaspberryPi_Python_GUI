import socket
import threading
import time

class DataReader(threading.Thread):
    def __init__(self, plot_data_queue, fitting_data_queue, selected_channels, status_queue=None):
        super().__init__(daemon=True)
        self.plot_data_queue = plot_data_queue
        self.fitting_data_queue = fitting_data_queue
        self.selected_channels = selected_channels
        self.status_queue = status_queue

        self.running = True
        self.measuring = True
        self.file_path = None
        self.saving_to_csv = False
        self.sock = None

    def setup(self):
        self.sock = socket.create_connection(('127.0.0.1', 39998)) #establish tcp connection
        try:
            # Read initial connection data
            init = read_all(self.sock, timeout=0.5)
            print("Connected. Initial data:")
            print(init)

            print("\nChanging iteration...")
            self.sock.sendall(b'(i275)')  # Set iterations to 275
            time.sleep(0.1)        # Short wait before reading
            print("\nChanging Histogram Mode ")
            self.sock.sendall(b'(H3)')  # H0 to switch off, 1 for reference histogram, 2 for measuremnet histogram, 3 for both
            time.sleep(0.1)        # Short wait before reading
        except socket.error as e:
            print(f"Socket error during setup: {e}")
            self.running = False
        

    def run(self):     
        self.setup()     
        # Loop: trigger m0 and extract #HLONG01
        while self.running:
            if not self.measuring: # wait for user to start measurement
                time.sleep(0.1)
                continue
            print(f"Measuring on channels: {self.selected_channels}")
            self.sock.sendall(b'(m0)')  # Trigger measurement
            
            response = read_all(self.sock)
            for line in response.splitlines():
                if line.startswith('#ITT'): # update the status queue with iteration info adn etc.
                    if self.status_queue:
                        self.status_queue.put(line)
                for channel in self.selected_channels: # check if the channel is in the selected channels and update the data queue
                    target = f'#HLONG{channel:02d}'
                    if line.startswith(target):
                        # if file_path is set and saving is enabled save the data to a csv file
                        if self.file_path and self.saving_to_csv:
                            with open(self.file_path, 'a') as f:
                                f.write(line + '\n')

                        self.plot_data_queue.put(line)
                        self.fitting_data_queue.put(line)  #send the same data to 2 queuees because i dont want it to steal each other's data by sharing the same queue
                        break  # stop checking other channels once matched
            # # Parse the HLONG01 line
            # target = f'#HLONG{self.channel_number:02d}'
            # for line in response.splitlines():
            #     if line.startswith(target):
            #         self.data_queue.put(line)
            #         break

    def stop(self):
        self.running = False
        self.sock.close()
        print("DataReader stopped.")

    def take_n_measurements(self, n):
        """
        Take n measurements and return the data, useful for taking the irf measurement, only runs if measuring is False to prevent it from clashing with live measurements.
        :param n: Number of measurements to take
        :return: None
        """   
        print(f"Taking {n} measurements...")
        if self.measuring:
            print("Live measurements are running. Please stop them before taking manual measurements.")
            return
        for _ in range(n):
            self.sock.sendall(b'(m0)')  # Trigger measurement
            response = read_all(self.sock)
            for line in response.splitlines():
                for channel in self.selected_channels: # check if the channel is in the selected channels and update the data queue
                    target = f'#HLONG{channel:02d}'
                    if line.startswith(target):
                        # if file_path is set and saving is enabled save the data to a csv file
                        if self.file_path and self.saving_to_csv:
                            with open(self.file_path, 'a') as f:
                                f.write(line + '\n')
                            self.plot_data_queue.put(line)
                            self.fitting_data_queue.put(line) 
                            break

    def toggle_measurement(self):
        if self.measuring:
            self.measuring = False
            print("Measurement stopped.")
        else:
            self.measuring = True
            print("Measurement started.")

    def set_file_path(self, file_path):
        """
        Set the file path for saving data.
        This is a placeholder function, as the actual saving logic is not implemented here.
        """
        self.file_path = file_path
        print(f"File path set to: {file_path}")

    def toggle_saving_to_csv(self):
        """
        Toggle saving data to CSV.
        This is a placeholder function, as the actual saving logic is not implemented here.
        """
        self.saving_to_csv = not self.saving_to_csv
        status = "enabled" if self.saving_to_csv else "disabled"
        print(f"Saving to CSV is now {status}.")

    def send_command(self, command_char, command_value):
        if self.sock:
            try:
                command = f'({command_char}{command_value})'.encode('utf-8')
                self.sock.sendall(command)
                response = read_all(self.sock)
                if response:
                    print(f"Command {command_char} with value {command_value} sent. Response: {response}")
            except socket.error as e:
                print(f"Socket error while sending command {command_char}: {e}")
        else:
            print("Socket not initialized. Cannot send command.")

    def set_number_of_iterations(self, iterations):
        if not (0 <= iterations <= 65535):
            print("Iterations must be between 0 and 65535.")
            return
        self.send_command('I', iterations)
        

    def set_object_detection_threshold(self, threshold):
        if not (0 <= threshold <= 255):
            print("Threshold must be between 0 and 255.")
            return
        self.send_command('T', threshold)

    def set_operation_mode(self, mode):
        """
        Toggle the number of pixel rows
        0 - 3 x 3
        1 - 4 x 4
        2 - 3 x 6
        3 - 8 x 8
        """
        if not (0 <= mode <= 3):
            print("Please enter a valid mode\nAvailable modes: 0, 1, 2, 3")
            return
        self.send_command('P', mode)

    def set_histogram_mode(self, mode):
        """
        Set histogram mode
        0 - off
        1 - reference histogram
        2 - measurement histogram
        3 - both
        """
        if not (0 <= mode <= 3):
            print("Please enter a valid mode\nAvailable modes: 0, 1, 2, 3")
            return
        self.send_command('H', mode)

    def set_short_range_mode(self, mode):
        """
        Set short range mode
        0 - off
        1 - on
        """
        if not (0 <= mode <= 1):
            print("Please enter a valid mode\nAvailable modes: 0, 1")
            return
        self.send_command('R', mode)


def read_all(sock, timeout=0.07):
    sock.settimeout(timeout)
    chunks = []
    try:
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            chunks.append(chunk)
    except socket.timeout:
        pass
    return b''.join(chunks).decode('utf-8', errors='ignore')


