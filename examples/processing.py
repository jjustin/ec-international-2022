import numpy as np
import scipy
from scipy import signal
from ellipse import LsqEllipse
import circle_fit as cf
#import matplotlib.pyplot as plt
from fft import range_fft



def ellipse_fit_correction(st_data):
    X = np.array(list(zip(st_data.real, st_data.imag)))
    reg = LsqEllipse().fit(X)
    center, width, height, phi = reg.as_parameters()

    #print(f'center: {center[0]:.3f}, {center[1]:.3f}')
    #print(f'width: {width:.3f}')
    #print(f'height: {height:.3f}')
    #print(f'phi: {phi:.3f}')


    st_data_corr = st_data.copy()
    st_data_corr.real -= center[0]
    st_data_corr.imag -= center[1]

    A_rot = np.asarray([[np.cos(phi), np.sin(phi)], [-np.sin(phi), np.cos(phi)]])
    st_data_corr = np.matmul(A_rot, np.asarray([st_data.real, st_data.imag]))
    st_data_corr = st_data_corr[0] + 1j*st_data_corr[1]
    st_data_corr.real = st_data_corr.real/width
    st_data_corr.imag = st_data_corr.imag/height

    #fig = plt.figure(figsize=(6, 6))
    #ax = plt.subplot()
    #ax.axis('equal')
    #ax.plot(st_data.real, st_data.imag)
    #ax.plot(st_data_corr.real, st_data_corr.imag)
    #ellipse = Ellipse(
    #    xy=center, width=2*width, height=2*height, angle=np.rad2deg(phi),
    #    edgecolor='r', fc='None', lw=2, label='Fit', zorder=2
    #)
    #ax.add_patch(ellipse)

    #plt.xlabel('$X_1$')
    #plt.ylabel('$X_2$')

    #plt.legend()
    #plt.show()
    
    return st_data_corr


def fit_circle(st_data):
    xc = 0
    yc = 0
    r = 0

    coords = np.asarray([st_data.real, st_data.imag])
    coords = np.transpose(coords, (1, 0))
    xc, yc, r, s = cf.least_squares_circle(coords)


    corrected_data = (st_data.real - xc) + 1j*(st_data.imag - yc)
    corrected_phase = np.unwrap(np.angle(corrected_data))

    #plt.plot(st_data[2].real, st_data[2].imag)
    #plt.show()

    #phases = np.unwrap(np.angle(st_data[2]))
    #plt.plot(phases)
    #plt.plot(np.unwrap(phase_signal))
    #plt.show()

    center = xc + 1j*yc

    return np.unwrap(corrected_phase), np.abs(corrected_data), center, r



def do_processing(st_data, compensateMotion=False):

    st_data = remove_mean(st_data)

    range_idx, range_data = get_range_index_recordwise(st_data)

    st_data = range_data[:, :, range_idx]

    phases = np.zeros((st_data.shape[0], st_data.shape[1]))
    abses = np.zeros((st_data.shape[0], st_data.shape[1]))
    centers = np.zeros(st_data.shape[0], dtype=np.complex64)
    radii = np.zeros(st_data.shape[0])

    for rx in range(3):
        #for i in range(2):
        st_data[rx] = ellipse_fit_correction(st_data[rx])
        phases[rx], abses[rx], centers[rx], radii[rx] = fit_circle(st_data[rx])
        #    if compensateMotion is True:
                #pass
        #        phases[rx] = motion_compensation(phases[rx], th=0.1)
                #abses[rx] = motion_compensation(abses[rx], th=0.001)
        st_data[rx] = abses[rx]*np.exp(1j*phases[rx])


    #abses = np.abs(st_data)

    return phases, abses, centers, radii


def processing_rangeData(st_data, compensateMotion=False):

    st_data = remove_mean(st_data)

    range_idx, range_data = get_range_index_recordwise(st_data)

    st_data = range_data[:, :, range_idx]

    return st_data


def get_range_index_recordwise(raw_data):
    range_window_func = lambda x: scipy.signal.windows.chebwin(x, at=100)
    range_data  = range_fft(raw_data, range_window_func)

    range_tmp = np.sum(np.sum(np.abs(range_data), axis=-2), axis=0)
    range_idx = np.argmax(range_tmp, axis=-1)


    return range_idx, range_data

def remove_mean(raw_data):
    raw_data -= np.mean(raw_data, axis=-1, keepdims=True)
    raw_data -= np.mean(raw_data, axis=-2, keepdims=True)
    return raw_data


