import numpy as np
import matplotlib.pyplot as plt
import fringe_functions as ff
import numpy.fft as fft

#Set up interferometer
pyxis = ff.AC_interferometer(baseline = 20, #m
                             diameter = 0.07, #m
                             bandpass = 15e-9, #m
                             start_wavelength = 600e-9, #m
                             end_wavelength = 750e-9, #m
                             eta = 0.2,
                             seeing = 1, #arcsec
                             v = 20, #m/s
                             incoh_scaling = 30,
                             num_delays = 1000,
                             scale_delay = 0.005,
                             disp_length = 0, #m
                             disp_lam_0 = 675e-9) #m

frame_rate = 30 #Fps
scan_rate = 3e-6 #m/s
start_position = -1e-3
end_position = 1e-3
F_0 = 100
V = 0.5
window_size = 1000
plot_skip = 200000

meters_per_frame = scan_rate/frame_rate

x = np.arange(start_position,end_position,meters_per_frame)

wavelength = pyxis.wavelengths[0]
correct_wavenumber = 1/wavelength




flux = []
for i in x:
    f = pyxis.calc_output(i,F_0,V)
    flux.append(f)

r = np.array(flux)[:,1,0]

arr = np.zeros(window_size)
values = np.arange(int(window_size/2))
period = window_size*meters_per_frame
wavenumbers = (values/period)[1:]

idx = (np.abs(wavenumbers - correct_wavenumber)).argmin()
signal_indices = np.arange(idx-1,idx+2)
noise_indices = np.concatenate((np.arange(0,idx-1),np.arange(idx+2,len(wavenumbers))))


for i,f in enumerate(x):
    arr = ff.shift_array(arr,1,fill_value=r[i])
    ft = fft.fft(arr)/window_size
    ft = ft[range(int(window_size/2))][1:]

    signal = np.median(abs(ft[signal_indices]))
    noise = np.std(abs(ft[noise_indices]))

    snr = signal/noise
    print(snr)

    #if i % plot_skip == 74:
    #    plt.plot(wavenumbers, abs(ft))

    #    plt.show()

    if i > 1000 and snr > 20:
        print(x[i-int(window_size/2)])

        break
