import numpy as np
from matplotlib import pyplot as plt
import os
import datetime


now_run_num = 34
weighted_averaged_run = [26, 28]

weighted_average = False
timestamp = "2022-1-5-11hr.5min.28sec"
right_time_delay_list = [-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 15.0, 30.0, 50.0, 70.0, 100.0, 300.0, 1000.0]

aniso_whole_dat_list = []
aniso_whole_stderr_list = []
aniso_abs_sum_list = []

test_delays = range(len(right_time_delay_list))
def make_aniso_abs_sum():
    global q_val_file
    global aniso_abs_sum_list
    for each_delay in test_delays:
        if weighted_average:
            anisotropy_file = []
            file_open_path = "../results/anisotropy/Anal_result_dat_file/run{0:04d}_{1:04d}_avg/{2}".format(weighted_averaged_run[0], weighted_averaged_run[1], timestamp)
            anisotropy_file = np.load(file_open_path + "/run{0:04d}_{1:04d}_avg_delay{2}_aniso.npy".format(weighted_averaged_run[0], weighted_averaged_run[1], (each_delay + 1)))
            aniso_abs_sum_list.append(np.sum(np.abs(anisotropy_file)))
        else:
            file_open_path = "../results/anisotropy/anal_result/run{0:04d}/run{0}_{1}".format(now_run_num, timestamp)
            anisotropy_file = np.load(file_open_path + "/run{0}_delay{1}_aniso.npy".format(now_run_num, each_delay + 1))
            aniso_abs_sum_list.append(np.sum(np.abs(anisotropy_file)))


def dat_file_out(out_file_list):
    file_out_path = "../results/anisotropy/svd/"

    if weighted_average:
        file_out_path = file_out_path + "/run{0:04d}_{1:04d}_avg_aniso_abs_sum.dat".format(weighted_averaged_run[0], weighted_averaged_run[1])
    else:
        file_out_path = file_out_path + "/run{0}_aniso_abs_sum.dat".format(now_run_num)

    dat_file = open(file_out_path, 'w')
    dat_file.write("time" + "\t" + "summation of absolute value of each delay" + "\n")
    for line_num in range(len(right_time_delay_list)):
        dat_file.write(str(right_time_delay_list[line_num]) + "\t" + str(out_file_list[line_num]))
        dat_file.write("\n")

make_aniso_abs_sum()
dat_file_out(aniso_abs_sum_list)