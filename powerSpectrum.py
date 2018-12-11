from scipy import fft
import json
import numpy as np
import matplotlib.pyplot as plt


with open('./results/cover_box_line_reflector.json') as f:
    data = json.load(f)


for scan in range(1,17):
    time = data[str(scan)]["time"]
    amplitude = data[str(scan)]["amplitude"]

    #data was swapped in collection
    time, amplitude = amplitude, time

    plt.plot(time,amplitude)
    plt.show()

    numScans = len(amplitude)
    scanTime = time[-1] - time[0]
    print(numScans)
    print(scanTime)
    frequency = numScans/scanTime

    print("frequency",frequency)


    # Nyquist Sampling Criteria
    T = 1/frequency # inverse of the sampling rate
    x = np.linspace(0.0, 1.0/(2.0*T), int(numScans/2))

    # FFT algorithm
    yr = fft(amplitude) # "raw" FFT with both + and - frequencies
    y = 2/numScans * np.abs(yr[0:np.int(numScans/2)]) # positive freqs only

    print("gna plot")

    plt.plot(x, y)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Vibration (g)')
    plt.title('Frequency Domain')
    plt.show()
    plt.pause(0.2)
    plt.gcf().clear()
