import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import os
import re


def load_pd_val(dat_common_path, run_num_arr):
    all_run_pd_arr = []
    for run_num in run_num_arr:
        now_dat_path = dat_common_path + "run_{0:05d}_DIR/eh1pdcom_channel1p/".format(run_num)
        now_run_pd_arr = []
        for file_idx, now_h5_file in enumerate(os.listdir(now_dat_path)):
            if file_idx != 0:
                break
            if now_h5_file.endswith(".h5"):
                now_h5 = h5.File(now_dat_path + now_h5_file, "r")
                now_file_keys = list(now_h5.keys())
                for now_key in now_file_keys:
                    now_pulseID = int(re.findall("(.*)\.(.*)_(.*)", now_key)[0][2])
                    if now_pulseID % 12 == 0:
                        now_pd_val = np.array(now_h5[now_key])
                        now_run_pd_arr.append(now_pd_val)
        all_run_pd_arr.append(now_run_pd_arr)
    return all_run_pd_arr

def calc_pd_dist(all_run_pd_arr, run_num_arr):
    for run_idx, run_num in enumerate(run_num_arr):
        now_run_pd_arr = all_run_pd_arr[run_idx]
        now_run_pd_avg = np.average(now_run_pd_arr)
        now_run_pd_std = np.std(now_run_pd_arr)
        plt.hist(now_run_pd_arr, bins=100, label="run {}".format(run_num))
        plt.legend()
        plt.show()
        print("run {} pd avg: {}, std: {}".format(run_num, now_run_pd_avg, now_run_pd_std))


dat_common_path = "/xfel/ffs/dat/scan/"
run_num_arr = [53, 185]
all_run_pd_arr = []
all_run_pd_arr = load_pd_val(dat_common_path, run_num_arr)
calc_pd_dist(all_run_pd_arr, run_num_arr)

