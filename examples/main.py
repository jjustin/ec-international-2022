import ifxdaq
import processing
import numpy as np
#print(ifxdaq.__version__)
from ifxdaq.sensor.radar_ifx import RadarIfxAvian

config_file = "radar_configs/RadarIfxBGT60.json"
raw_data    = []

with RadarIfxAvian(config_file) as device:                             # Initialize the radar with configurations
    
    for i_frame, frame in enumerate(device):                           # Loop through the frames coming from the radar
        
        raw_data.append(np.squeeze(frame['radar'].data/(4095.0)))      # Dividing by 4095.0 to scale the data
        
        if(len(raw_data) > 4999 and len(raw_data) % 5000 == 0):        # 5000 is the number of frames. which corresponds to 5seconds
            
            data = np.swapaxes(np.asarray(raw_data), 0, 1)

            phases, abses, _, _ = processing.do_processing(data)       # preprocessing to get the phase information
            
            phases              = np.mean(phases, axis=0)
            break