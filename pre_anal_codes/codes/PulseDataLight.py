from enum import Enum
import numpy as np

class PulseDataLight:
    '''
    Similar data class of codes.PulseData
    Diffrent point
    * recieve (water / norm) range index directly
    * more faster than original class during sum
    * remove I0, ice part

    Data class for save one pulse`s information (1D x:tth y:int graph)
    Since all tth value is common, do not save tth value.
    '''

    laser_is_on = False
    droplet_hit = False
    water_peak_sum = 0
    water_type = None
    is_normalized = False
    neg_delay_laser_on = False
    pair_range_sum = 0
    norm_range_sum = 0
    key = None

    def __init__(self, int_val, data_key, water_start_idx, water_after_idx, norm_start_idx, norm_after_idx,
                 pair_start_idx, pair_after_idx):
        self.intensity_val = int_val
        self.key = data_key
        self.water_start_idx = water_start_idx
        self.water_after_idx = water_after_idx
        self.norm_start_idx = norm_start_idx
        self.norm_after_idx = norm_after_idx
        self.pair_start_idx = pair_start_idx
        self.pair_after_idx = pair_after_idx
        self.calc_range_sum()

    def calc_range_sum(self):
        self.water_peak_sum = sum(self.intensity_val[self.water_start_idx:self.water_after_idx])
        self.norm_range_sum = sum(self.intensity_val[self.norm_start_idx:self.norm_after_idx])
        # TODO :  edit pair range (done?)
        self.pair_range_sum = sum(self.intensity_val[self.pair_start_idx:self.pair_after_idx])


    def check_laser_onoff(self, pulseID):
        """
        when pulseID is divisible by 24, signal is obatianed with laser
        :param pulseID:
        """
        pulseID = int(pulseID)
        if (pulseID % 24) == 0:
            self.laser_is_on = True
        else:
            self.laser_is_on = False
        '''if (pulseID % 24) == 0:
            self.laser_is_on = False
        else:
            self.laser_is_on = True'''
        # print(pulseID, "id -> ", self.laser_is_on)

    def norm_given_range(self):
        self.is_normalized = True
        try:
            self.intensity_val = list((np.array(self.intensity_val) / self.norm_range_sum) * 1E7)
        except:
            print("normalization error ")
            print(self.intensity_val)
            print(self.norm_range_sum)
            self.is_normalized = False