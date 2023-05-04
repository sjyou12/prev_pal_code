"""
extract necessary information for anaisotropy signal from in-exp analysis code
"""

from codes.MultiRunProc import MultiRunProc
import math
import numpy as np
from codes.AnisoAnal import AnisoAnal

now_analysis_run = 53


def make_water_sum_file(run_num=53, norm_val=False, pair_info=False):
    # common constant for calculating q values
    XrayEnergy = 14  # keV unit
    XrayWavelength = 12.3984 / XrayEnergy  # Angstrom unit (10^-10 m)
    QCoefficient = 4 * math.pi / XrayWavelength
    NormFactor = 100000  # Normalization factor (sum of all integration)
    FileCommonRoot = "/home/common/exp_data/PAL-XFEL_20201217-back/rawData/"

    run_num = int(run_num)
    nowSingleRunData = [(run_num, 50000, 100000, 900000)]
    dataPreProcessor = MultiRunProc(nowSingleRunData)
    dataPreProcessor.common_variables(file_common_root=FileCommonRoot)
    dataPreProcessor.set_file_name_and_read_tth()
    ## get graph for decide criteria
    '''
    use plot_water_sum_dist function for make water_sum file (necessary for anisotropy analysis)
    otherwise, use read_intensity_only_water function
    '''
    if norm_val:
        dataPreProcessor.plot_water_sum_dist(each_run_plot=False, sum_file_out=False)
    if pair_info:
        # save intensity files
        dataPreProcessor.read_intensity_only_water()
        dataPreProcessor.fileout_pair_info_only(pair_file_out=True)

# make_water_sum_file(run_num=now_analysis_run, pair_info=True) # analysis code with made water_sum data
load_water_sum_file_root = "../results/anisotropy/run" + str(now_analysis_run) + "_watersum.npy"
water_sum = np.load(load_water_sum_file_root)
load_pair_info_file_root = "../results/anisotropy/run" + str(now_analysis_run) + "_pairinfo.npy"
pair_info = np.load(load_pair_info_file_root, allow_pickle=True)

print(water_sum.shape, pair_info.shape)

def anal_anisotropy_with_water_sum_single_delay(water_sum_data, run_num=53, delay_num=6, pair_info_data=None):
    anisoAnalyzer = AnisoAnal()
    now_common_path = "/home/common/exp_data/PAL-XFEL_20201217-back/rawData/"
    now_mask_file = "/home/common/exp_data/PAL-XFEL_20201217-back/scratch/mask_droplet_1222.h5"
    now_beam_center = [723.145, 723.498]  # unit : pixels
    now_sample_detector_distance = 90.958  # unit : mm

    anisoAnalyzer.set_common_env(common_path=now_common_path)
    anisoAnalyzer.set_mask(now_mask_file, show_mask=False)
    anisoAnalyzer.set_img_info(now_beam_center[0], now_beam_center[1], now_sample_detector_distance)
    anisoAnalyzer.read_single_delay_h5_files(run_num=run_num, delay_num=delay_num)
    if pair_info_data is None:
        anisoAnalyzer.make_normalized_diff_img(water_sum_data)
    else:
        anisoAnalyzer.make_normalized_pair_diff_img(water_sum_data, pair_info_data, delay_num, True)
    anisoAnalyzer.aniso_anal_diff_img(result_plot=True)

def anal_anisotropy_with_water_sum_all_delay(water_sum_data, run_num=53, pair_info_data=None, test_delay=7):
    anisoAnalyzer = AnisoAnal()
    now_common_path = "/home/common/exp_data/PAL-XFEL_20201217-back/rawData/"
    now_mask_file = "/home/common/exp_data/PAL-XFEL_20201217-back/scratch/mask_droplet_1222.h5"
    now_beam_center = [723.145, 723.498]  # unit : pixels
    now_sample_detector_distance = 90.958  # unit : mm

    anisoAnalyzer.set_common_env(common_path=now_common_path)
    anisoAnalyzer.set_mask(now_mask_file, show_mask=False)
    anisoAnalyzer.set_img_info(now_beam_center[0], now_beam_center[1], now_sample_detector_distance)
    # anisoAnalyzer.aniso_anal_single_run_all_delay(run_num=run_num, norm_data=water_sum_data, pair_info=pair_info_data)
    anisoAnalyzer.read_img_file_names(run_num=run_num)
    anisoAnalyzer.aniso_anal_each_delay(norm_data=water_sum_data, pair_info=pair_info_data, idx_delay=test_delay)

# anal_anisotropy_with_water_sum_single_delay(water_sum_data=water_sum, run_num=53, delay_num=6,pair_info_data=pair_info)
test_delays = range(7, 15)
for each_delay in test_delays:
    anal_anisotropy_with_water_sum_all_delay(water_sum_data=water_sum, run_num=53, pair_info_data=pair_info, test_delay=each_delay)
print("here")