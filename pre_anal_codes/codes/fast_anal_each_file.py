import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
import configparser
from codes.FastProcess import FastProcessDataSet

# get criteria info
config = configparser.ConfigParser()
config.read('anal_cond.cfg')

LowerBoundOfIce = float(config['DEFAULT']['LowerBoundOfIce'])
UpperBoundOfNotIce = float(config['DEFAULT']['UpperBoundOfNotIce'])
LowerBoundOfWater = float(config['DEFAULT']['LowerBoundOfWater'])
UpperBoundOfNotWater = float(config['DEFAULT']['UpperBoundOfNotWater'])

# run 19 error : no 008_002 file
# run_name_temp_map = dict(run20=260)
run_name_temp_map = dict(run20=260, run21=260,
                         run23=250, run24=250, run25=245, run26=245, run27=240, run28=240, run29=235, run30=235)
# run18=270,
# run19=270,
# run_name_temp_map = dict(run20=260, run23=250)


CommonDelayNum = 16
CommonDelayIdx = ['001_001', '002_001', '003_001', '004_001', '005_001', '006_001', '007_001', '008_001',
                  '001_002', '002_002', '003_002', '004_002', '005_002', '006_002', '007_002', '008_002']
run_name_temp_map = dict(run35=250, run36=250, run37=250, run38=250, run39=250, run40=250)
run_name_temp_map = dict(run52=250)
CommonDelayNum = 54
CommonDelayIdx = ['001_001', '001_002', '001_003', '001_004', '001_005', '001_006', '001_007', '001_008', '001_009', '001_010', '001_011', '001_012', '001_013', '001_014', '001_015', '001_016', '001_017', '001_018', '001_019', '001_020', '001_021', '001_022', '001_023', '001_024', '001_025', '001_026', '001_027', '001_028', '001_029', '001_030', '001_031', '001_032', '001_033', '001_034', '001_035', '001_036', '001_037', '001_038', '001_039', '001_040', '001_041', '001_042', '001_043', '001_044', '001_045', '001_046', '001_047', '001_048', '001_049', '001_050', '001_051', '001_052', '001_053', '001_054']
# CommonDelayIdx = ['001_001', '001_002', '001_003', '001_004', '001_005', '001_006', '001_007', '001_008', '001_009', '001_010', '001_011']


# h5 file name
for each_file in run_name_temp_map.keys():
    test_file_family_name = each_file
    test_file_index = "00001"
    test_file_main_name = test_file_family_name + "_" + test_file_index
    test_file_dir = "/xfel/ffs/dat/scan/" + test_file_main_name + "_DIR"
    print(test_file_dir)

    # output file setting
    outDir = "../results/"
    delay_diff_out_file_name = outDir + "difference/" + "each_delay_diff_" + test_file_main_name + ".dat"
    delay_diff_out_Fp = open(delay_diff_out_file_name, 'w')
    cut_delay_diff_file_name = outDir + "difference/" + "cut_delay_diff_" + test_file_main_name + ".dat"
    cut_diff_out_Fp = open(cut_delay_diff_file_name, 'w')
    time_delay_avg_out_file_name = outDir + "difference/" + "time_delay_avg_diff_" + test_file_main_name + ".dat"
    time_delay_avg_out_Fp = open(time_delay_avg_out_file_name, 'w')
    cut_time_delay_avg_file_name = outDir + "difference/" + "cut_time_avg_diff_" + test_file_main_name + ".dat"
    cut_time_delay_avg_out_Fp = open(cut_time_delay_avg_file_name, 'w')

    laser_off_avg_file_name = outDir + "t_dep/laser_off_avg/" + "laser_off_avg_" + test_file_main_name + ".npy"
    delay_group_avg_file_name = outDir + "t_dep/delay_group_avg/" + "delay_group_avg_" + test_file_main_name + ".npy"
    each_delay_avg_file_name = outDir + "t_dep/each_delay_avg/"+ "each_delay_avg_" + test_file_main_name + ".npy"
    laser_off_each_delay_avg_file_name = outDir + "t_dep/off_each_delay_avg/" + "off_each_avg_" + test_file_main_name + ".npy"

    NowData = FastProcessDataSet()
    NowData.set_file_name(test_file_family_name, test_file_index)
    NowData.set_delay_num(CommonDelayNum, CommonDelayIdx)
    print(NowData.file_main_name, NowData.file_dir)
    NowData.load_each_file()

    NowData.read_all_intensity_value(norm_with_I0=True)
    NowData.close_all_files()

    NowData.decide_criteria(show_ice_plot=False)
    NowData.set_file_pointer('each-delay-diff', delay_diff_out_Fp)
    NowData.set_file_pointer('time-delay-avg-diff', time_delay_avg_out_Fp)
    NowData.set_file_pointer('cut-each-delay-diff', cut_diff_out_Fp)
    NowData.set_file_pointer('cut-time-delay-avg-diff', cut_time_delay_avg_out_Fp)
    NowData.set_file_pointer('laser-off-avg-np', laser_off_avg_file_name)
    NowData.set_file_pointer('delay-group-avg-np', delay_group_avg_file_name)
    NowData.set_file_pointer('each-delay-avg-np', each_delay_avg_file_name)
    NowData.set_file_pointer('off-delay-avg-np', laser_off_each_delay_avg_file_name)


    NowData.pairwise_diff_calc(file_out=True, show_cmp=False, show_neg_pair=False, plot_each_avg=False)
    NowData.process_laser_off(save_all_avg=True, save_delay_avg=True)
    NowData.additional_process_diff(show_before_cutted=True, show_after_cutted=False, file_out=True,
                                         svd_with_cut=True, weighted_group_avg=False, sep_np_save=True)
    NowData.calc_SVD_and_plot(file_out=True)

# TODO : save delay-avgerage numpy value as file.