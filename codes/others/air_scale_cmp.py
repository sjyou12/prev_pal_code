import os
import h5py as h5
import matplotlib.pyplot as plt
import numpy as np

air_run = 6
cmp_sample_run = 5
cmp_sample_delay_idx = 12
dark_run_name = "230427_bkg_1_00005_DIR"

common_file_path = "/xfel/ffs/dat/scan/"
sample_int_file_path = "/xfel/ffs/dat/ue_230427_FXL/analysis/run_{0:05d}_DIR/".format(cmp_sample_run)

air_run_name = f"run_{air_run:05d}_DIR"
cmp_sample_run_name = f"run_{cmp_sample_run:05d}_DIR"

air_int_file_dir = f"{common_file_path}{air_run_name}/eh1rayMXAI_int/"
dark_int_file_dir = f"{common_file_path}{dark_run_name}/eh1rayMXAI_int/"
# sample_int_file_dir = f"{common_file_path}{cmp_sample_run_name}/eh1rayMXAI_int/"
sample_int_file_dir = f"{sample_int_file_path}"

def calc_dir_avg_int(int_file_dir):
    int_file_names = os.listdir(int_file_dir)
    int_data_list = []
    for file_name in int_file_names:
        now_file_name = int_file_dir + file_name
        now_obj = h5.File(now_file_name, "r")
        keys = list(now_obj.keys())
        for each_key in keys:
            now_data = now_obj[each_key]
            int_data_list.append(now_data)
    int_data_list = np.array(int_data_list)
    avg_int = np.average(int_data_list, axis=0)
    return avg_int

def file_avg_int(int_file_dir, file_name):
    int_data_list = []
    now_file_name = int_file_dir + file_name
    now_obj = h5.File(now_file_name, "r")
    keys = list(now_obj.keys())
    for each_key in keys:
        now_data = now_obj[each_key]
        int_data_list.append(now_data)
    int_data_list = np.array(int_data_list)
    avg_int = np.average(int_data_list, axis=0)
    return avg_int

air_avg_int = calc_dir_avg_int(air_int_file_dir)
dark_avg_int = calc_dir_avg_int(dark_int_file_dir)

sample_int_file_name = f"001_001_{cmp_sample_delay_idx:03d}.h5"
sample_avg_int = file_avg_int(sample_int_file_dir, sample_int_file_name)

fig, axs = plt.subplots(figsize=(12,6))
axs.plot(air_avg_int, label="air")
axs.plot(dark_avg_int, label="dark")
axs.plot(sample_avg_int, label="sample")
axs.legend()
fig.show()

np.savetxt("air_avg_int.txt", air_avg_int)
np.savetxt("sample_avg_int.txt", sample_avg_int)