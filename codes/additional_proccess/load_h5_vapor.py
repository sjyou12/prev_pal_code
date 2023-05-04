import h5py as h5
import numpy as np

runList = 58
#file_common_name = "run58"
merge_run_list = [50, 54]
merge_multi_run = False
expand_negative_pool = False
print_criteria = 10
average_criteria = 1000

def save_each_run_vapor_h5(runList, merge_run_list, merge_multi_run, expand_negative_pool):
    key_list = []
    temp_vapor_img_list = []
    temp_average_list = []
    average_num = 0
    file_common_name = "run" + str(runList)

    h5_file_save_root = "/data/exp_data/PAL-XFEL_20210514/rawdata/" + "run{0:04d}_00001_DIR/".format(runList) + "eh1rayMX_img/"
    file_open_root = "../results/vapor_signal/" + file_common_name + "_vapor.npy"
    temp_vapor_info_list = np.load(file_open_root)
    file_out_root = "../results/vapor_signal_h5/vapor_img_" + file_common_name + ".npy"
    for idx in range(len(temp_vapor_info_list)):
        delay_idx = temp_vapor_info_list[idx][1]
        each_key = temp_vapor_info_list[idx][2]
        now_shot_I0 = float(temp_vapor_info_list[idx][3])
        key_list.append(each_key)
        delay_num = int(delay_idx) + 1
        temp_save_root = h5_file_save_root + "001_001_{0:03d}".format(delay_num) + ".h5"
        now_file = h5.File(temp_save_root, 'r')
        temp_img_data = np.array(now_file[each_key], dtype=float)
        now_file.close()
        now_img_data = np.multiply(temp_img_data, now_shot_I0)
        temp_vapor_img_list.append(now_img_data)
        #vapor_img_list.append([now_img_data, each_key])
        if (idx + 1) % print_criteria == 0:
            print("read {0} / {1} shot".format(idx + 1, len(temp_vapor_info_list)))
        if len(temp_vapor_info_list) > 1000:
            if (idx + 1) % average_criteria == 0:
                temp_average_list.append(np.average(temp_vapor_img_list, axis=0))
                temp_vapor_img_list = []
                average_num += 1
            elif ((len(temp_vapor_info_list) % average_criteria) < average_criteria) & ((idx + 1) == len(temp_vapor_info_list)):
                temp_average_list.append(np.average(temp_vapor_img_list, axis=0))
                temp_vapor_img_list = []
    if len(temp_vapor_info_list) > 1000:
        average_vapor_h5_img = np.average(temp_average_list, axis=0)
    else:
        average_vapor_h5_img = np.average(temp_vapor_img_list, axis=0)
    np.save(file_out_root, average_vapor_h5_img)

save_each_run_vapor_h5(runList, merge_run_list, merge_multi_run, expand_negative_pool)