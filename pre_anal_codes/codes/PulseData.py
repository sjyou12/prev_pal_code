import numpy as np
import configparser
from enum import Enum

class IceType(Enum):
    ICE = 1
    GRAY_ICE = 2
    NOT_ICE = 3

class WaterType(Enum):
    WATER = 1
    GRAY_WATER = 2
    NOT_WATER = 3
    OUTLIER = 4

config = configparser.ConfigParser()
config.read('anal_cond.cfg')

# read constant from config file
IcePeakStartQ = float(config['DEFAULT']['IcePeakStartQ'])
IcePeakEndQ = float(config['DEFAULT']['IcePeakEndQ'])
WaterPeakStartQ = float(config['DEFAULT']['WaterPeakStartQ'])
WaterPeakEndQ = float(config['DEFAULT']['WaterPeakEndQ'])
LowerBoundOfIce = float(config['DEFAULT']['LowerBoundOfIce'])
UpperBoundOfNotIce = float(config['DEFAULT']['UpperBoundOfNotIce'])
LowerBoundOfWater = float(config['DEFAULT']['LowerBoundOfWater'])
UpperBoundOfNotWater = float(config['DEFAULT']['UpperBoundOfNotWater'])
NormStartQ = float(config['DEFAULT']['NormStartQ'])
NormEndQ = float(config['DEFAULT']['NormEndQ'])
PlotStartIdx = int(config['DEFAULT']['PlotStartIdx'])
PlotEndIdx = int(config['DEFAULT']['PlotEndIdx'])
WaterOutlierLowerBound = float(config['DEFAULT']['WaterOutlierLowerBound'])
PairStartQ = 1
PairEndQ = 6

class PulseData:
    '''
    Data class for save one pulse`s information (1D x:tth y:int graph)
    Since all tth value is common, do not save tth value.
    '''


    def __init__(self, int_val, I0_val=0.0):
        self.intensity_val = int_val
        self.laser_is_on = False
        self.droplet_hit = False
        self.ice_paek_sum = 0
        self.water_peak_sum = 0
        self.ice_type = None
        self.water_type = None
        self.is_normalized = False
        self.norm_range_sum = 0
        self.neg_delay_laser_on = False
        self.I0_vaule = 0
        self.norm_with_I0 = False
        self.pair_range_sum = 0

        if I0_val != 0.0:
            self.I0_vaule = I0_val
            self.norm_with_I0 = True
            self.intensity_val = int_val / I0_val

    def classify_data(self, common_q):
        # refer constant in anal_cond.cfg
        for int_idx, q in enumerate(common_q):
            if IcePeakStartQ < q < IcePeakEndQ:
                # print("now ice q is :", q)
                self.ice_paek_sum += self.intensity_val[int_idx]
            elif WaterPeakStartQ < q < WaterPeakEndQ:
                # print("now water q is :", q)
                self.water_peak_sum += self.intensity_val[int_idx]

            # normalization with different range
            if NormStartQ < q < NormEndQ:
                self.norm_range_sum += self.intensity_val[int_idx]
            if PairStartQ < q < PairEndQ:
                self.pair_range_sum += self.intensity_val[int_idx]

        # print("ice sum", self.ice_paek_sum, "water sum", self.water_peak_sum)
        # data type classification
        # in cpp code, use water range + 3
        if self.ice_paek_sum > LowerBoundOfIce:
            self.ice_type = IceType.ICE
        elif self.ice_paek_sum < UpperBoundOfNotIce:
            self.ice_type = IceType.NOT_ICE
        else:
            self.ice_type = IceType.GRAY_ICE

        if self.water_peak_sum > WaterOutlierLowerBound:
            self.water_type = WaterType.OUTLIER
        elif self.water_peak_sum > LowerBoundOfWater:
            self.water_type = WaterType.WATER
        elif self.water_peak_sum < UpperBoundOfNotWater:
            self.water_type = WaterType.NOT_WATER
        else:
            self.water_type = WaterType.GRAY_WATER

        return self.ice_paek_sum, self.water_peak_sum

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