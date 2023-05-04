from palxfel_scatter.diff_pair_1dcurve.MultiRunProc import MultiRunProc
from palxfel_scatter.anisotropy.AnisoAnal import AnisoAnal
import numpy as np

now_run_num = 56


def single_run_DB_make(run_number):
    UpperBoundOfNotWater = 15000
    LowerBoundOfWater = 20000
    WaterOutlierLowerBound = 70000  # need to also change cut criteria in AnisoAnal
    singleDB = [(run_number, UpperBoundOfNotWater, LowerBoundOfWater, WaterOutlierLowerBound)]
    return singleDB


nowDataDB = single_run_DB_make(now_run_num)

# common constant for calculating q values
XrayEnergy = 9.7  # keV unit
NormFactor = 100000  # Normalization factor (sum of all integration)
FileCommonRoot = "/home/common/exp_data_2021/PAL-XFEL_20210514/rawdata/"

def make_data():
    global XrayEnergy, FileCommonRoot, now_run_num
    make_pair_info = True
    dataPreProcessor = MultiRunProc(nowDataDB)
    dataPreProcessor.common_variables(x_ray_energy=XrayEnergy, file_common_root=FileCommonRoot)
    dataPreProcessor.set_file_name_and_read_tth()
    # get graph for decide criteria
    dataPreProcessor.vapor_anal(incr_dist_plot=True, plot_outlier=True, sum_file_out=False, plot_vapor_avg=True)
    # save intensity files
    dataPreProcessor.read_intensity_only_water(np_file_out=False, plot_watersum_pass = True, rm_vapor=True)  # if need q_val -> make np_file_out True
    dataPreProcessor.pairwise_diff_calc(fileout_pair_info=make_pair_info)
    dataPreProcessor.additional_process_diff(fileout_pair_info=make_pair_info)

    # dataPreProcessor.plot_water_sum_dist(each_run_plot=False, sum_file_out=True)
    # # save intensity files
    # dataPreProcessor.read_intensity_only_water(np_file_out=False)  # if need q_val -> make np_file_out True
    # dataPreProcessor.fileout_pair_info_only(pair_file_out=True)

make_data()