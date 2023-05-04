import numpy as np
import h5py as h5
import os

dat_common_path = "/xfel/ffs/dat/scan/"
run_num = 29
target_run_num = 29
now_dat_path = dat_common_path + "run_{0:05d}_DIR/".format(run_num)
now_dir_list = os.listdir(now_dat_path)
split_num = 500
num_imgs = 1000

for dir_name in now_dir_list:
    if dir_name in ['eh1rayMXAI_int', 'eh1rayMXAI_tth', 'eh1rayMX_img']:
        now_h5 = h5.File(now_dat_path + dir_name + "/001_001_080.h5", "r")
        now_file_keys = list(now_h5.keys())
        split_h5_dat = []
        split_key_arr = []
        temp_arr = []
        temp_key_arr = []
        os.makedirs("/xfel/ffs/dat/ue_230427_FXL/run_{0:05d}_DIR/".format(target_run_num) + dir_name, exist_ok=True)
        for key_idx, now_key in enumerate(now_file_keys):
            temp_key_arr.append(now_key)
            temp_arr.append(list(now_h5[now_key]))
        now_h5.close()
        split_data_arr = np.split(np.array(temp_arr), num_imgs/split_num)
        split_key_arr = np.split(np.array(temp_key_arr), num_imgs/split_num)

        # for split_idx in range(num_imgs/split_num):
        for split_idx in range(len(split_key_arr)):
            h5Fp = h5.File("/xfel/ffs/dat/ue_230427_FXL/run_{0:05d}_DIR/".format(target_run_num) + dir_name + "/001_001_{0:03d}.h5".format(split_idx+1), 'w')
            now_key_list = split_key_arr[split_idx]
            now_data_list = split_data_arr[split_idx]
            for key_idx, each_key in enumerate(now_key_list):
                h5Fp.create_dataset(each_key, data=now_data_list[key_idx])
            h5Fp.close()
        print("finish split {}".format(dir_name))
    else:
        continue




