from palxfel_scatter.diff_pair_1dcurve.MultiRunProc import MultiRunProc
from palxfel_scatter.anisotropy.AnisoAnal import AnisoAnal
import numpy as np

now_run_num = 71
# 31-th time delay is strange

def single_run_DB_make(run_number):
    UpperBoundOfNotWater = 8000
    LowerBoundOfWater = 10000
    WaterOutlierLowerBound = 30000  # need to also change cut criteria in AnisoAnal
    singleDB = [(run_number, UpperBoundOfNotWater, LowerBoundOfWater, WaterOutlierLowerBound)]
    return singleDB

nowDataDB = single_run_DB_make(now_run_num)

# common constant for calculating q values
XrayEnergy = 9.7  # keV unit
NormFactor = 100000  # Normalization factor (sum of all integration)
FileCommonRoot = "/xfel/ffs/dat/scan/"

def make_data():
    global XrayEnergy, FileCommonRoot, now_run_num
    make_pair_info = True
    dataPreProcessor = MultiRunProc(nowDataDB)
    dataPreProcessor.common_variables(x_ray_energy=XrayEnergy, file_common_root=FileCommonRoot)
    dataPreProcessor.set_file_name_and_read_tth()
    # get graph for decide criteria
    dataPreProcessor.find_peak_pattern(incr_dist_plot=True, plot_outlier=True, sum_file_out=True)
    # save intensity files
    dataPreProcessor.read_intensity_only_water(np_file_out=False)  # if need q_val -> make np_file_out True
    dataPreProcessor.pairwise_diff_calc(fileout_pair_info=make_pair_info)
    dataPreProcessor.additional_process_diff(fileout_pair_info=make_pair_info)

# make_data()

print("make data end")

def anal_anistorpy(run_num):
    load_water_sum_file_root = "../results/anisotropy/run" + str(run_num) + "_watersum.npy"
    water_sum = np.load(load_water_sum_file_root)
    load_pair_info_file_root = "../results/anisotropy/run" + str(run_num) + "_pairinfo.npy"
    pair_info = np.load(load_pair_info_file_root, allow_pickle=True)

    print(water_sum.shape, pair_info.shape)

    def anal_anisotropy_each_delay(water_sum_data, run_num=53, pair_info_data=None, test_delay=7):
        anisoAnalyzer = AnisoAnal()
        now_common_path = "/xfel/ffs/dat/scan/"
        now_mask_file = "/xfel/ffs/dat/ue_210514_FXL/scratch/210516_mask_1.h5"
        now_bkg_file = "/xfel/ffs/dat/scan/210517_bg_00002_DIR/eh1rayMX_img/00000001_00000300.h5"
        now_beam_center = [713.351, 724.525]  # unit : pixels
        now_sample_detector_distance = 89.608  # unit : mm

        anisoAnalyzer.set_common_env(common_path=now_common_path)
        anisoAnalyzer.set_mask(now_mask_file, show_mask=False)
        anisoAnalyzer.set_background(now_bkg_file)
        anisoAnalyzer.set_img_info(now_beam_center[0], now_beam_center[1], now_sample_detector_distance)
        # anisoAnalyzer.aniso_anal_single_run_all_delay(run_num=run_num, norm_data=water_sum_data, pair_info=pair_info_data)
        anisoAnalyzer.read_img_file_names(run_num=run_num)
        anisoAnalyzer.aniso_anal_each_delay(norm_data=water_sum_data, pair_info=pair_info_data, idx_delay=test_delay, run_num=run_num)

    test_delays = range(0, 55)
    # test_delays = range(11, 41)
    for each_delay in test_delays:
        anal_anisotropy_each_delay(water_sum_data=water_sum, run_num=run_num, pair_info_data=pair_info, test_delay=each_delay)

anal_anistorpy(run_num=now_run_num)