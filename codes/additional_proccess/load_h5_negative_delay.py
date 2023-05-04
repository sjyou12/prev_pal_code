import h5py as h5
import numpy as np
runList = 68
file_common_name = "run{0}".format(runList)
merge_run_list = [50, 54]
merge_multi_run = False
expand_negative_pool = True
print_criteria = 10
save_criteria = 1000

def save_each_run_neagtive_h5(runList, file_common_name, merge_run_list, merge_multi_run, expand_negative_pool):
    key_list = []
    negative_delay_int_list = []
    other_run_key_list = []
    save_num = 0

    if merge_multi_run:
        if expand_negative_pool:
            for run_num in merge_run_list:
                h5_file_save_root = "/data/exp_data/PAL-XFEL_20210514/rawdata/" + "run{0:04d}_00001_DIR/".format(run_num) + "eh1rayMX_img/"
                file_open_root = "../results/each_run_watersum_int/laser_off_all_delay_list_run{0}_{1}.npy".format(merge_run_list[0], merge_run_list[1]) #TODO need to prepare for more runs
                temp_pulse_info_list = np.load(file_open_root)
                file_out_root = "../results/each_run_negative_delay_img/negative_delay_img_of_run{0}_{1}.npy".format(merge_run_list[0], merge_run_list[1])
                if run_num == merge_run_list[0]:
                    for idx in range(len(temp_pulse_info_list)):
                        delay_idx = temp_pulse_info_list[idx][0]
                        each_key = temp_pulse_info_list[idx][1]
                        key_list.append(each_key)
                        delay_num = int(delay_idx) + 1
                        temp_save_root = h5_file_save_root + "001_001_{0:03d}".format(delay_num) + ".h5"
                        now_file = h5.File(temp_save_root, 'r')
                        try:
                            now_img_data = np.array(now_file[each_key], dtype=float)
                            now_file.close()
                            negative_delay_int_list.append([now_img_data, each_key])
                            #negative_delay_int_list[each_key] = now_img_data
                        except:
                            other_run_key_list.append([delay_idx, each_key])
                        if (idx + 1) % print_criteria == 0:
                            print("read {0} / {1} file of run{2}".format(idx + 1, len(temp_pulse_info_list), run_num))

                else:
                    for idx in range(len(other_run_key_list)):
                        delay_idx = other_run_key_list[idx][0]
                        each_key = other_run_key_list[idx][1]
                        delay_num = int(delay_idx) + 1
                        temp_save_root = h5_file_save_root + "001_001_{0:03d}".format(delay_num) + ".h5"
                        now_file = h5.File(temp_save_root, 'r')
                        now_img_data = np.array(now_file[each_key], dtype=float)
                        now_file.close()
                        #negative_delay_int_list[each_key] = now_img_data
                        negative_delay_int_list.append([now_img_data, each_key])
                        if (idx + 1) % print_criteria == 0:
                            print("read {0} / {1} file of run{2}".format(idx + 1, len(temp_pulse_info_list), run_num))
        else:
            for run_num in merge_run_list:
                h5_file_save_root = "/data/exp_data/PAL-XFEL_20210514/rawdata/" + "run{0:04d}_00001_DIR/".format(run_num) + "eh1rayMX_img/"
                file_open_root = "../results/merge_run_small_ice_pass/small_ice_test_pass_negative_delay_pulse_info_run{0}_{1}.npy".format(merge_run_list[0], merge_run_list[1]) #TODO need to prepare for more runs
                temp_pulse_info_list = np.load(file_open_root)
                file_out_root = "../results/each_run_negative_delay_img/negative_delay_img_of_run{0}_{1}.npy".format(merge_run_list[0], merge_run_list[1])
                if run_num == merge_run_list[0]:
                    for idx in range(len(temp_pulse_info_list)):
                        delay_idx = temp_pulse_info_list[idx][0]
                        each_key = temp_pulse_info_list[idx][1]
                        key_list.append(each_key)
                        delay_num = int(delay_idx) + 1
                        temp_save_root = h5_file_save_root + "001_001_{0:03d}".format(delay_num) + ".h5"
                        now_file = h5.File(temp_save_root, 'r')
                        try:
                            now_img_data = np.array(now_file[each_key], dtype=float)
                            now_file.close()
                            negative_delay_int_list.append([now_img_data, each_key])
                        except:
                            other_run_key_list.append([delay_idx, each_key])
                        if (idx + 1) % print_criteria == 0:
                            print("read {0} / {1} file of run{2}".format(idx + 1, len(temp_pulse_info_list), run_num))

                else:
                    for idx in range(len(other_run_key_list)):
                        delay_idx = other_run_key_list[idx][0]
                        each_key = other_run_key_list[idx][1]
                        delay_num = int(delay_idx) + 1
                        temp_save_root = h5_file_save_root + "001_001_{0:03d}".format(delay_num) + ".h5"
                        now_file = h5.File(temp_save_root, 'r')
                        now_img_data = np.array(now_file[each_key], dtype=float)
                        now_file.close()
                        negative_delay_int_list.append([now_img_data, each_key])
                        if (idx + 1) % print_criteria == 0:
                            print("read {0} / {1} file of run{2}".format(idx + 1, len(temp_pulse_info_list), run_num))
        np.save(file_out_root, negative_delay_int_list)
        np.save("../results/each_run_negative_delay_img/keys_of_negative_delay_img_of_run{0}_{1}.npy".format(merge_run_list[0], merge_run_list[1]), key_list)
    else:
        h5_file_save_root = "/data/exp_data/PAL-XFEL_20210514/rawdata/" + "run{0:04d}_00001_DIR/".format(runList) + "eh1rayMX_img/"
        file_open_root = "../results/each_run_watersum_int/laser_off_all_delay_list_" + file_common_name + ".npy"
        temp_pulse_info_list = np.load(file_open_root)
        file_out_root = "../results/each_run_negative_delay_img/negative_delay_img_of_" + file_common_name + ".npy"
        for idx in range(len(temp_pulse_info_list)):
            delay_idx = temp_pulse_info_list[idx][1]
            each_key = temp_pulse_info_list[idx][2]
            key_list.append(each_key)
            delay_num = int(delay_idx) + 1
            temp_save_root = h5_file_save_root + "001_001_{0:03d}".format(delay_num) + ".h5"
            now_file = h5.File(temp_save_root, 'r')
            now_img_data = np.array(now_file[each_key], dtype=float)
            now_file.close()
            negative_delay_int_list.append([now_img_data, each_key])
            if (idx + 1) % print_criteria == 0:
                print("read {0} / {1} shot".format(idx + 1, len(temp_pulse_info_list)))
            if len(temp_pulse_info_list) > 2000:
                if (idx + 1) % save_criteria == 0:
                    temp_file_out_root = "../results/each_run_negative_delay_img/negative_delay_img_of_" + file_common_name + "_" + str(int((idx + 1)/save_criteria)) + ".npy"
                    np.save(temp_file_out_root, negative_delay_int_list)
                    negative_delay_int_list = []
                    save_num += 1
                elif (idx + 1) == len(temp_pulse_info_list):
                    if ((idx + 1)%len(temp_pulse_info_list) < save_criteria) & ((idx + 1)/len(temp_pulse_info_list) == 1):
                        temp_file_out_root = "../results/each_run_negative_delay_img/negative_delay_img_of_" + file_common_name + "_" + str(save_num + 1) + ".npy"
                        np.save(temp_file_out_root, negative_delay_int_list)
        if len(temp_pulse_info_list) > 2000:
            np.save("../results/each_run_negative_delay_img/keys_of_negative_delay_img_of_" + file_common_name + ".npy", key_list)
            pass
        else:
            #now_file.close()
            np.save(file_out_root, negative_delay_int_list)
            np.save("../results/each_run_negative_delay_img/keys_of_negative_delay_img_of_" + file_common_name + ".npy", key_list)

save_each_run_neagtive_h5(runList, file_common_name, merge_run_list, merge_multi_run, expand_negative_pool)