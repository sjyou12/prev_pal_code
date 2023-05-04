import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
import configparser
from codes.DataClasses import ReadOneDataSet

# h5 file name
test_file_family_name = "run38"
test_file_index = "00001"
# test_file_main_name = test_file_family_name + "_" + test_file_index
test_file_main_name = test_file_family_name
# test_file_dir = "/xfel/ffs/dat/scan/" + test_file_main_name + "_DIR"
test_file_dir = "/home/common/exp_data/PAL-XFEL_20201217-back/rawData/" + test_file_family_name
print(test_file_dir)

# get criteria info
config = configparser.ConfigParser()
config.read('anal_cond.cfg')

LowerBoundOfIce = float(config['DEFAULT']['LowerBoundOfIce'])
UpperBoundOfNotIce = float(config['DEFAULT']['UpperBoundOfNotIce'])
LowerBoundOfWater = float(config['DEFAULT']['LowerBoundOfWater'])
UpperBoundOfNotWater = float(config['DEFAULT']['UpperBoundOfNotWater'])

# output file setting
outDir = "../results/"
delay_cmp_out_file_name = outDir + "difference/" + "diff_cmp_delay_" + test_file_main_name + ".dat"
delay_cmp_out_Fp = open(delay_cmp_out_file_name, 'w')
cut_delay_cmp_file_name = outDir + "difference/" + "cut_diff_cmp_" + test_file_main_name + ".dat"
cut_cmp_out_Fp = open(cut_delay_cmp_file_name, 'w')

# for run 21
NowDelayNum = 3
NowDelayIdx = ['001_001', '001_002', '001_003']

# NowDelayIdx = ['001_001', '001_002', '001_003', '001_004']
# NowDelayIdx = ['001_001', '001_002', '001_003', '001_004', '001_005',
#                '001_006', '001_007', '001_008', '001_009', '001_010']

# for run 12
# NowDelayNum = 10
# NowDelayIdx = ['001_001', '002_001', '003_001', '004_001', '005_001',
#                '001_002', '002_002', '003_002', '004_002', '005_002']
# for run 13
# NowDelayNum = 12
# NowDelayIdx = ['001_001', '002_001', '003_001', '004_001',
#                '001_002', '002_002', '003_002', '004_002',
#                '001_003', '002_003', '003_003', '004_003']
# for run 18
#NowDelayNum = 16
#NowDelayIdx = ['001_001', '002_001', '003_001', '004_001',
#               '005_001', '006_001', '007_001', '008_001',
#               '001_002', '002_002', '003_002', '004_002',
#               '005_002', '006_002', '007_002', '008_002']
# for run 34
NowDelayNum = 54
# NowDelayNum = 11 # for run 41
NowDelayIdx = ['001_001', '001_002', '001_003', '001_004', '001_005', '001_006', '001_007', '001_008', '001_009', '001_010', '001_011', '001_012', '001_013', '001_014', '001_015', '001_016', '001_017', '001_018', '001_019', '001_020', '001_021', '001_022', '001_023', '001_024', '001_025', '001_026', '001_027', '001_028', '001_029', '001_030', '001_031', '001_032', '001_033', '001_034', '001_035', '001_036', '001_037', '001_038', '001_039', '001_040', '001_041', '001_042', '001_043', '001_044', '001_045', '001_046', '001_047', '001_048', '001_049', '001_050', '001_051', '001_052', '001_053', '001_054']


readTestData = ReadOneDataSet()
readTestData.set_file_name(test_file_family_name, test_file_index)
readTestData.set_delay_num(NowDelayNum, NowDelayIdx)
print(test_file_main_name, readTestData.file_main_name)
print(test_file_dir, readTestData.file_dir)
readTestData.load_each_file()
# for data_idx in range(10):
#     readTestData.plot_int_per_tth(0, data_idx)
readTestData.read_all_intensity_value(norm_with_I0=False)
# readTestData.plot_all_in_one_graph(0)
# readTestData.plot_all_in_one_graph(1)
# readTestData.plot_all_in_one_graph(2)
readTestData.close_all_files()


# plot_all_given(data_int_val_list)
# readTestData.plot_front_selected(15)

readTestData.decide_criteria(show_ice_plot=False)
readTestData.set_file_pointer('delay-cmp-diff', delay_cmp_out_Fp)
readTestData.set_file_pointer('cut-delay-cmp-diff', cut_cmp_out_Fp)

# readTestData.pairwise_diff_calc(file_out=False)
readTestData.pairwise_diff_calc(file_out=False, show_whole_avg=False, show_neg_pair=False, plot_each_avg=False)
readTestData.process_laser_off()
readTestData.additional_process_diff(show_before_cutted=False, show_after_cutted=False, file_out=True, svd_with_cut=True)
# readTestData.save_signal_per_delay_np()
# readTestData.calc_SVD_and_plot(file_out=True)

