
from fft import range_fft, range_doppler_fft



def processing_rangeDopplerData(st_data, compensateMotion=False):


    range_doppler_data = range_doppler_fft(st_data)

    return range_doppler_data



