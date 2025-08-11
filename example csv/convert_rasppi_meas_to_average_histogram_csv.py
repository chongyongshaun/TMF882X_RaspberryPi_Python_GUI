import numpy as np

filename = 'irf_2.csv'  # replace with your actual file path

with open(filename, 'r') as file:
    # Skip the first 37 lines
    for _ in range(37):
        next(file)
    for line in file:
        if line.startswith('#HLONG'):
            parts = line.strip().split(';')
            # skip the first entry which is the #HLONGXX tag
            average_histogram = list(map(int, parts[1:129]))  # take the first 128 bins
            break  # Stop after the first #HLONG line

# # plot the average histogram for debug
# import matplotlib.pyplot as plt

# plt.figure(figsize=(10, 5))
# plt.plot(average_histogram, marker='o')
# plt.title('Average Histogram')
# plt.xlabel('Bin')
# plt.ylabel('Counts')
# plt.grid(True)
# plt.show()

# Convert to CSV string with integers
average_csv_string = ','.join(str(int(round(val))) for val in average_histogram)

print("output: ", average_csv_string)
extracted_name = filename + '_extracted.csv'
with open(extracted_name, 'w') as out_file:
    out_file.write(average_csv_string + '\n')
