import h5py as h5
from matplotlib import pyplot as plt
import numpy as np
import os
import re
import datetime
import pyFAI
import pyFAI.azimuthalIntegrator
import pyFAI.detectors
from scipy.stats import linregress
from scipy import interpolate

class ImgData:
    laser_is_on = None
    img_key = None

    def __init__(self, img_data, pulseID, img_key):
        self.raw_data = img_data
        self.pulseID = int(pulseID)
        self.img_key = img_key
        self.is_normalized = False
        if (self.pulseID % 12) == 0:
            self.laser_is_on = True
        else:
            self.laser_is_on = False


class AnisoAnal:
    common_file_path = None
    mask_file_path = None
    beam_center_x = 0.0
    beam_center_y = 0.0
    sample_detector_dist = 0.0
    detector_num_pixel = 1440  # 1440x1440 pixel detector
    pixel_size = 156e-6  # unit of m
    xray_energy = 9.7  # keV unit
    xray_wavelength = 12.3984 / xray_energy  # Angstrom unit (10^-10 m)
    xray_wavelength_in_m = xray_wavelength * 1e-10  # unit of m
    norm_on_img_list = []
    norm_off_img_list = []
    remain_area_by_mask = []
    mask_arr = None
    now_delay_idx = -1
    now_avg_diff_img = []
    log_Fp = None
    want_log = False

    q_interp = False

    def __init__(self):
        self.on_img_list = []
        self.off_img_list = []
        self.each_delay_cutted_q = []
        self.each_delay_cutted_iso = []
        self.each_delay_cutted_aniso = []
        self.each_delay_cutted_stderr_iso = []
        self.each_delay_cutted_stderr_aniso = []
        self.img_file_names = []
        self.img_dir_path = []
        self.bkg_img = []
        self.UpperBoundOfNotWater = 0
        self.LowerBoundOfWater = 0
        self.WaterOutlierLowerBound = 0

    def set_print_log_file(self, run_num, want_log):
        self.want_log = want_log
        if want_log:
            out_file_name = "/data/exp_data/myeong0609/PAL-XFEL_20210514/analysis/codes/run_" + str(run_num) + "_log.txt"
            self.log_Fp = open(out_file_name, 'a')
            #self.log_Fp.write("Successful open of log file\n")

    def set_common_env(self, common_path):
        self.common_file_path = common_path

    # def set_img_info(self, center_x, center_y, sd_dist): # old version, information with fit2d
    #     self.beam_center_x = center_x  # unit : pixels
    #     self.beam_center_y = center_y  # unit : pixels
    #     self.sample_detector_dist = sd_dist  # unit : mm

    def set_img_info(self, poni_file_path): # new version, information from pyFAI-calib2
        self.AzimuthalIntegrator = pyFAI.load(poni_file_path)

    def set_mask(self, mask_file, show_mask=False):
        self.mask_file_path = mask_file
        now_mask_file = h5.File(self.mask_file_path, 'r')
        self.mask_arr = np.array(now_mask_file['mask'])
        self.remain_area_by_mask = np.subtract(np.ones_like(self.mask_arr), self.mask_arr)
        print("successful read of mask :", self.mask_file_path)
        if show_mask:
            plt.pcolor(self.remain_area_by_mask)
            plt.colorbar()
            plt.show()

    def set_background(self, bkg_file):
        # now_bkg_file = h5.File(bkg_file, 'r')
        # keys_now_file = list(now_bkg_file.keys())
        #
        # bkg_img_datas = []
        # for each_key in keys_now_file:
        #     bkg_img_datas.append(np.array(now_bkg_file[each_key]))
        #
        # self.bkg_img = np.average(np.array(bkg_img_datas), axis=0)
        self.bkg_img = bkg_file
        print("end background img calc")

    def read_single_delay_h5_files(self, run_num, delay_num):
        run_img_dir_path = self.common_file_path + "run" + str(run_num) + "/eh2rayMX_img/"
        img_h5_file_names = os.listdir(run_img_dir_path)
        img_h5_file_names.sort()

        for each_file_name in img_h5_file_names:
            if each_file_name != img_h5_file_names[delay_num]:
                continue  # test for first file
            now_file_path = run_img_dir_path + each_file_name
            now_file = h5.File(now_file_path, 'r')
            now_delay_idx_1_based = int(re.findall("(.*)_(.*)_(.*)\.(.*)", each_file_name)[0][2])
            self.now_delay_idx = now_delay_idx_1_based
            print("successful read of", now_delay_idx_1_based, "- th delay img file : ", now_file_path)
            now_file_keys = list(now_file.keys())
            for each_key in now_file_keys:
                now_pulseID = int(re.findall("(.*)\.(.*)_(.*)", each_key)[0][2])
                now_img_data = np.array(now_file[each_key], dtype=float)
                now_data = ImgData(now_img_data, now_pulseID, each_key)
                if now_data.laser_is_on:
                    self.on_img_list.append(now_data)
                else:
                    self.off_img_list.append(now_data)
            now_file.close()

    def aniso_anal_single_run_all_delay(self, run_num, norm_data, pair_info):
        run_img_dir_path = self.common_file_path + "run" + str(run_num) + "/eh2rayMX_img/"
        img_h5_file_names = os.listdir(run_img_dir_path)
        img_h5_file_names.sort()
        # test_delay_num = range(3, 10)
        test_delay_num = [12, 13]

        for idx_delay, each_file_name in enumerate(img_h5_file_names):
            if idx_delay not in test_delay_num:
                continue
            now_file_path = run_img_dir_path + each_file_name
            now_file = h5.File(now_file_path, 'r')
            now_delay_idx_1_based = int(re.findall("(.*)_(.*)_(.*)\.(.*)", each_file_name)[0][2])
            self.now_delay_idx = now_delay_idx_1_based
            print("successful read of", now_delay_idx_1_based, "- th delay img file : ", now_file_path)
            now_file_keys = list(now_file.keys())
            for each_key in now_file_keys:
                now_pulseID = int(re.findall("(.*)\.(.*)_(.*)", each_key)[0][2])
                now_img_data = np.array(now_file[each_key], dtype=float)
                now_data = ImgData(now_img_data, now_pulseID, each_key)
                if now_data.laser_is_on:
                    self.on_img_list.append(now_data)
                else:
                    self.off_img_list.append(now_data)
            now_file.close()

            print("before make_normalized_pair_diff_img")
            self.make_normalized_pair_diff_img(norm_data, pair_info, idx_delay, True)
            print("before aniso_anal_diff_img")
            self.aniso_anal_diff_img(result_plot=True)
        self.multi_aniso_plot()

    def read_img_file_names(self, merge_run_list, merge_multi_run, run_num):
        if merge_multi_run:
            for idx_run in range(len(merge_run_list)):
                run_img_dir_path = self.common_file_path + "run_{0:05d}_DIR/".format(merge_run_list[idx_run]) + "eh1rayMX_img/"
                self.img_dir_path.append(run_img_dir_path)
                img_h5_file_names = os.listdir(run_img_dir_path)
                img_h5_file_names.sort()
                self.img_file_names.append(img_h5_file_names)
                img_h5_file_names = [] #delete for saving memory
        else:
            run_img_dir_path = self.common_file_path + "run_{0:05d}_DIR/".format(run_num) + "eh1rayMX_img/"
            self.img_dir_path = run_img_dir_path
            img_h5_file_names = os.listdir(run_img_dir_path)
            img_h5_file_names.sort()
            self.img_file_names = img_h5_file_names

    def aniso_anal_each_delay(self, now_time, paused_run, merge_run_list, merge_multi_run, expand_negative_pool, rm_vapor, norm_data, pair_info, idx_delay, run_num):
        if merge_multi_run:
            for idx_run in range(len(merge_run_list)):
                now_file_name = self.img_file_names[idx_run][idx_delay]
                now_file_path = self.img_dir_path[idx_run] + now_file_name
                temp_negative_delay_list = []

                now_file = h5.File(now_file_path, 'r')
                now_delay_idx_1_based = int(re.findall("(.*)_(.*)_(.*)\.(.*)", now_file_name)[0][2])
                self.now_delay_idx = now_delay_idx_1_based
                print("successful read of", now_delay_idx_1_based, "- th delay img file : ", now_file_path)
                now_file_keys = list(now_file.keys())
                for each_key in now_file_keys:
                    now_pulseID = int(re.findall("(.*)\.(.*)_(.*)", each_key)[0][2])
                    now_img_data = np.array(now_file[each_key], dtype=float)
                    now_img_data = now_img_data - self.bkg_img
                    now_data = ImgData(now_img_data, now_pulseID, each_key)
                    if expand_negative_pool:
                        if now_data.laser_is_on:
                            self.on_img_list.append(now_data)
                        else:
                            pass
                    else:
                        if now_data.laser_is_on:
                            self.on_img_list.append(now_data)
                        else:
                            self.off_img_list.append(now_data)
                now_file.close()
                now_file_keys = []
                now_img_data = []     # Delete data for saving memory
                if expand_negative_pool:
                    key_list_laser_off_pool = []
                    delay_off_key_list = []
                    idx = 1
                    temp_key_list_open_path = "../results/each_run_negative_delay_img/keys_of_negative_delay_img_of_run{0}_{1}.npy".format(merge_run_list[0], merge_run_list[1])
                    temp_key_list = np.load(temp_key_list_open_path)
                    for num_pair in range(len(pair_info[idx_delay])):
                        where_key = 0
                        where_key = (np.where(pair_info[idx_delay][num_pair][1] == temp_key_list))[0][0]
                        key_list_laser_off_pool.append(temp_key_list[where_key])
                    try:
                        temp_file_open_path = "../results/each_run_negative_delay_img/negative_delay_img_of_run{0}_{1}.npy".format(merge_run_list[0], merge_run_list[1])
                        temp_negative_delay_list = np.load(temp_file_open_path, allow_pickle=True)
                    except:
                        for file_idx in range(10):
                            each_file_negative_delay_list = []
                            file_idx += 1
                            temp_file_open_path = "../results/each_run_negative_delay_img/negative_delay_img_of_run" + str(run_num) + "_" + str(file_idx) + "_.npy"
                            if file_idx == 1:
                                temp_negative_delay_list = np.load(temp_file_open_path, allow_pickle=True)
                            else:
                                each_file_negative_delay_list = np.load(temp_file_open_path, allow_pickle=True)
                                temp_negative_delay_list = np.vstack(
                                    (temp_negative_delay_list, each_file_negative_delay_list))

                    for key in key_list_laser_off_pool:
                        where_int = None
                        for idx_pulse in range(len(temp_negative_delay_list)):
                            if temp_negative_delay_list[idx_pulse][1] == key:
                                where_int = idx_pulse
                                break
                        now_temp_img = temp_negative_delay_list[where_int][0] - self.bkg_img
                        now_temp_pulseID = int(re.findall("(.*)\.(.*)_(.*)", key)[0][2])
                        temp_img_data = ImgData(now_temp_img, now_temp_pulseID, key)
                        self.off_img_list.append(temp_img_data)
                    temp_negative_delay_list = []
                    temp_key_list = []
                    now_temp_img = []

                    # for idx in range(len(temp_negative_delay_list)):
                    #     now_temp_img = temp_negative_delay_list[idx] - self.bkg_img
                    #     now_temp_pulseID = int(re.findall("(.*)\.(.*)_(.*)", temp_key_list[idx])[0][2])
                    #     temp_img_data = ImgData(now_temp_img, now_temp_pulseID, temp_key_list[idx])
                    #     self.off_img_list.append(temp_img_data)
                    temp_negative_delay_list = []
                    temp_key_list = []
                    now_temp_img = []
        else:
            now_file_name = self.img_file_names[idx_delay]
            now_file_path = self.img_dir_path + now_file_name
            temp_negative_delay_list = []

            now_file = h5.File(now_file_path, 'r')
            now_delay_idx_1_based = int(re.findall("(.*)_(.*)_(.*)\.(.*)", now_file_name)[0][2])
            self.now_delay_idx = now_delay_idx_1_based
            print("successful read of", now_delay_idx_1_based, "- th delay img file : ", now_file_path)
            now_file_keys = list(now_file.keys())
            if rm_vapor:
                file_open_path = "../results/vapor_signal_h5/vapor_img_run{0}.npy".format(run_num)
                vapor_img = np.load(file_open_path)

            for each_key in now_file_keys:
                now_pulseID = int(re.findall("(.*)\.(.*)_(.*)", each_key)[0][2])
                now_img_data = np.array(now_file[each_key], dtype=float)
                if rm_vapor:
                    now_img_data = now_img_data - self.bkg_img - vapor_img
                else:
                    now_img_data = now_img_data - self.bkg_img #todo release the code after bkg_img test
                now_data = ImgData(now_img_data, now_pulseID, each_key)
                if expand_negative_pool:
                    if now_data.laser_is_on:
                        self.on_img_list.append(now_data)
                    else:
                        pass
                else:
                    if now_data.laser_is_on:
                        self.on_img_list.append(now_data)
                    else:
                        self.off_img_list.append(now_data)
            now_file.close()
            now_file_keys = []
            now_img_data = []  # Delete data for saving memory
            if expand_negative_pool:
                key_list_laser_off_pool = []
                delay_off_key_list = []
                idx = 1
                temp_key_list_open_path = "../results/each_run_negative_delay_img/keys_of_negative_delay_img_of_run" + str(run_num) + ".npy"
                temp_key_list = np.load(temp_key_list_open_path)
                for num_pair in range(len(pair_info[idx_delay])):
                    where_key = 0
                    where_key = (np.where(pair_info[idx_delay][num_pair][1] == temp_key_list))[0][0]
                    key_list_laser_off_pool.append(temp_key_list[where_key])
                try:
                    temp_file_open_path = "../results/each_run_negative_delay_img/negative_delay_img_of_run" + str(run_num) + ".npy"
                    temp_negative_delay_list = np.load(temp_file_open_path, allow_pickle=True)
                except:
                    for file_idx in range(5): #TODO change file range when changes the run number
                        each_file_negative_delay_list = []
                        file_idx += 1
                        temp_file_open_path = "../results/each_run_negative_delay_img/negative_delay_img_of_run" + str(run_num) + "_" + str(file_idx) + ".npy"
                        if file_idx == 1:
                            temp_negative_delay_list = np.load(temp_file_open_path, allow_pickle=True)
                        else:
                            each_file_negative_delay_list = np.load(temp_file_open_path, allow_pickle=True)
                            temp_negative_delay_list = np.vstack((temp_negative_delay_list ,each_file_negative_delay_list))

                for key in key_list_laser_off_pool:
                    where_int = None
                    for idx_pulse in range(len(temp_negative_delay_list)):
                        if temp_negative_delay_list[idx_pulse][1] == key:
                            where_int = idx_pulse
                            break
                            '''for key in key_list_laser_off_pool:
                                    now_temp_img = temp_negative_delay_list.item().get(key) - self.bkg_img'''
                    now_temp_img = temp_negative_delay_list[where_int][0] - self.bkg_img
                    now_temp_pulseID = int(re.findall("(.*)\.(.*)_(.*)", key)[0][2])
                    temp_img_data = ImgData(now_temp_img, now_temp_pulseID, key)
                    self.off_img_list.append(temp_img_data)
                temp_negative_delay_list = []
                temp_key_list = []
                now_temp_img = []

        print("before make_normalized_pair_diff_img")
        # TODO : off result plot
        self.make_normalized_pair_diff_img(norm_data, pair_info, idx_delay, run_num, paused_run, img_plot=True)
        print("before aniso_anal_diff_img")
        self.aniso_anal_diff_img(result_plot=False)
        self.now_delay_file_out(now_time, idx_delay, run_num, merge_run_list, merge_multi_run)
        # self.multi_aniso_plot()

    def get_diff_img(self):
        print("on img len : ", len(self.on_img_list), " / off img len : ", len(self.off_img_list))
        diff_img_len = min(len(self.on_img_list), len(self.off_img_list))
        diff_img_data_list = []
        for diff_idx in range(diff_img_len):
            diff_img_data = self.on_img_list[diff_idx].raw_data - self.off_img_list[diff_idx].raw_data
            diff_img_data_list.append(diff_img_data)
            '''if diff_idx == 0:
                plt.pcolor(diff_img.masked_data)
                plt.colorbar()
                plt.show()'''
        avg_diff_img = np.average(diff_img_data_list, axis=0)
        avg_diff_img = np.array(avg_diff_img, dtype=np.double)
        self.now_avg_diff_img = avg_diff_img
        plt.pcolor(avg_diff_img)
        plt.colorbar()
        plt.xlabel("x pixel")
        plt.ylabel("y pixel")
        plt.show()


        masked_val = avg_diff_img.copy()
        masked_val = np.where(self.mask_arr == 1, np.nan, masked_val)
        '''masked_val = np.full_like(self.mask_arr, np.nan, dtype=np.double)
        first_len, second_len = masked_val.shape

        for i_idx in range(first_len):
            for j_idx in range(second_len):
                if self.mask_arr[i_idx][j_idx] == 0:
                    masked_val[i_idx][j_idx] = avg_diff_img[i_idx][j_idx]'''

        plt.title("masked")
        plt.pcolor(masked_val)
        plt.colorbar()
        plt.xlabel("x pixel")
        plt.ylabel("y pixel")
        plt.show()

    def make_normalized_diff_img(self, norm_data):
        print("on img len : ", len(self.on_img_list), " / off img len : ", len(self.off_img_list))
        diff_img_len = min(len(self.on_img_list), len(self.off_img_list))
        self.norm_on_img_list = self.normalize_img_list(self.on_img_list, norm_data)
        norm_on_raw_img = [each_img.raw_data for each_img in self.norm_on_img_list]
        self.on_img_list = []
        print("end normalize on img")
        self.norm_off_img_list = self.normalize_img_list(self.off_img_list, norm_data)
        norm_off_raw_img = [each_img.raw_data for each_img in self.norm_off_img_list]
        self.off_img_list = []
        print("end normalize off img")
        print("outlier remove after : on (", len(self.norm_on_img_list), "), off (", len(self.norm_off_img_list), ")")


        avg_on_img = np.average(norm_on_raw_img, axis=0)
        avg_off_img = np.average(norm_off_raw_img, axis=0)
        avg_diff_img = avg_on_img - avg_off_img
        self.now_avg_diff_img = avg_diff_img

        plt.pcolor(avg_diff_img)
        plt.colorbar()
        plt.xlabel("x pixel")
        plt.ylabel("y pixel")
        plt.show()

        masked_val = avg_diff_img.copy()
        masked_val = np.where(self.mask_arr == 1, np.nan, masked_val)

        plt.title("masked")
        plt.pcolor(masked_val)
        plt.colorbar()
        plt.xlabel("x pixel")
        plt.ylabel("y pixel")
        plt.show()

    def make_normalized_pair_diff_img(self, norm_data, pair_info, delay_num, run_num, paused_run, img_plot=False):
        print("on img len : ", len(self.on_img_list), " / off img len : ", len(self.off_img_list))
        self.norm_on_img_list = self.normalize_img_list(self.on_img_list, norm_data)
        self.on_img_list = []
        print("end normalize on img")
        self.norm_off_img_list = self.normalize_img_list(self.off_img_list, norm_data)
        self.off_img_list = []
        print("end normalize off img")
        print("outlier remove after : on (", len(self.norm_on_img_list), "), off (", len(self.norm_off_img_list), ")")

        avg_diff_img = self.on_off_pairing_img_make(self.norm_on_img_list, self.norm_off_img_list, pair_info, delay_num, paused_run)

        self.now_avg_diff_img = avg_diff_img
        if img_plot:
            # plt.pcolor(avg_diff_img)
            # plt.colorbar()
            # plt.xlabel("x pixel")
            # plt.ylabel("y pixel")
            # plt.show()

            masked_val = avg_diff_img.copy()
            masked_val = np.where(self.mask_arr == 1, np.nan, masked_val)
            img_save_root = "../results/anisotropy/anal_result/run{0}_delay{1}_2d_diff_img.npy".format(run_num, delay_num + 1)
            np.save(img_save_root, masked_val)

            # plt.title("masked image of " + (str(delay_num) +1) + "-th delay")
            # plt.pcolor(masked_val)
            # plt.colorbar()
            # plt.xlabel("x pixel")
            # plt.ylabel("y pixel")
            # plt.show()

    @staticmethod
    def on_off_pairing_img_make(on_img_list, off_img_list, pair_info, delay_idx, paused_run):
        on_key_list = np.array([img.img_key for img in on_img_list])
        off_key_list = np.array([img.img_key for img in off_img_list])
        total_sum_each_delay = []
        diff_img_list = []
        #for each_pair_data in pair_info[delay_idx]:
        for idx, each_pair_data in enumerate(pair_info[delay_idx]):
            # print(each_pair_data)
            on_key = each_pair_data[0]
            off_key = each_pair_data[1]
            # TODO : remove try-except part
            try:
                on_img_idx = (np.where(on_key_list == on_key))[0][0]
                off_img_idx = (np.where(off_key_list == off_key))[0][0]
            except:
                print("error key : {}(on) / {}(off)".format(on_key, off_key))
            if paused_run:
                try:
                    on_img_data = on_img_list[on_img_idx].raw_data
                    off_img_data = off_img_list[off_img_idx].raw_data
                    diff_img_data = on_img_data - off_img_data
                    total_sum_each_img = np.sum(diff_img_data)
                    # if total_sum_each_img > 2E6:
                    #     strange_diff_list.append([total_sum_each_img, int(idx)])
                    # total_sum_each_delay.append(total_sum_each_img)
                    diff_img_list.append(diff_img_data)
                except:
                    pass
            else:
                on_img_data = on_img_list[on_img_idx].raw_data
                off_img_data = off_img_list[off_img_idx].raw_data
                diff_img_data = on_img_data - off_img_data
                total_sum_each_img = np.sum(diff_img_data)
                # if total_sum_each_img > 2E6:
                #     strange_diff_list.append([total_sum_each_img, int(idx)])
                #total_sum_each_delay.append(total_sum_each_img)
                diff_img_list.append(diff_img_data)
        #print("sum of difference of each image in " + str(delay_idx) + "-th delay")
        #np.savetxt("strange difference of each image in " + str(delay_idx) + "-th delay", strange_diff_list, fmt='%s', delimiter='/')
        avg_diff_img = np.average(diff_img_list, axis=0)
        return avg_diff_img

    # @staticmethod
    def normalize_img_list(self, img_list, key_norm_val_pair_list):
        print("now img list size : ", np.shape(img_list) , "now key_norm_val_pair_list : ", np.shape(key_norm_val_pair_list))
        UpperBoundOfNotWater = self.UpperBoundOfNotWater
        LowerBoundOfWater = self.LowerBoundOfWater
        WaterOutlierLowerBound = self.WaterOutlierLowerBound
        do_outlier_rm = True

        key_list = np.transpose(key_norm_val_pair_list)[0]
        normalized_img_list = []
        for img_idx, each_img in enumerate(img_list):
            if not each_img.is_normalized :
                now_key = each_img.img_key
                try:
                    where_key_is = (np.where(key_list == now_key))[0][0]
                except:
                    continue
                now_norm_val = float(key_norm_val_pair_list[where_key_is][1])
                if do_outlier_rm:
                    if (now_norm_val > WaterOutlierLowerBound) | (now_norm_val < LowerBoundOfWater):
                        continue
                now_norm_val = now_norm_val / 1e5  # for scaling
                # print(now_key, key_norm_val_pair_list[where_key_is])
                norm_img_data = np.divide(each_img.raw_data, now_norm_val)
                each_img.is_normalized = True
                norm_img = ImgData(norm_img_data, each_img.pulseID, each_img.img_key)
                normalized_img_list.append(norm_img)
                if img_idx % 100 == 0:
                    print("now work until ", img_idx)

            #elif each_img.is_normalized:

        key_norm_val_pair_list = []
        key_list = []
        return normalized_img_list

    def aniso_anal_diff_img(self, result_plot=False, azi_intg_plot=False):
        # set azimuthal integrator
        azi_intgrator = self.AzimuthalIntegrator #self.set_azimuthal_integrator()
        azi_result_2d = azi_intgrator.integrate2d(self.now_avg_diff_img, npt_rad=250, npt_azim=45, polarization_factor=0.996,
                                                  mask=self.mask_arr, unit="q_A^-1", method="splitpixel")
        # original value of npt_rad is 1024/for previous PAL-XFEL, we used 100

        q_A_inv_unit = azi_result_2d[1]
        q_m_inv_unit = q_A_inv_unit * 1e10
        phi_deg_unit = azi_result_2d[2]
        # if azi_intg_plot:
        #     plt.pcolor(azi_result_2d[0])
        #     plt.colorbar()
        #     plt.xlabel("radial axis = theta (Q)")
        #     plt.ylabel("azimutal axis = phi ")
        #     plt.show()
        #
        #     plt.title("azimutal average with Q / phi axis")
        #     plt.pcolor(q_A_inv_unit, phi_deg_unit, azi_result_2d[0])
        #     plt.colorbar()s
        #     plt.xlabel("Q (A^-1)")
        #     plt.ylabel("phi (deg)")
        #     plt.show()
        twotheta_rad = 2 * np.arcsin((q_m_inv_unit * self.xray_wavelength_in_m) / (4 * np.pi))

        now_delay_iso_list = []
        now_delay_aniso_list = []
        now_delay_stderr_iso_list = []
        now_delay_stderr_aniso_list = []
        for tth_idx, each_tth_val in enumerate(twotheta_rad):
            now_iso, now_aniso, now_stderr_iso, now_stderr_aniso = self.anisotropy_fit(each_tth_val, phi_deg_unit, azi_result_2d[0][:, tth_idx], q_A_inv_unit[tth_idx], tth_idx, False)
            now_delay_iso_list.append(now_iso)
            now_delay_aniso_list.append(now_aniso)
            now_delay_stderr_iso_list.append(now_stderr_iso)
            now_delay_stderr_aniso_list.append(now_stderr_aniso)

        now_delay_iso_list = np.array(now_delay_iso_list)
        now_delay_aniso_list = np.array(now_delay_aniso_list)
        now_delay_stderr_iso_list = np.array(now_delay_stderr_iso_list)
        now_delay_stderr_aniso_list = np.array(now_delay_stderr_aniso_list)
        cutted_iso = now_delay_iso_list[~np.isnan(now_delay_iso_list)]
        cutted_aniso = now_delay_aniso_list[~np.isnan(now_delay_iso_list)]
        cutted_q_val = q_A_inv_unit[~np.isnan(now_delay_iso_list)]
        cutted_stderr_iso = now_delay_stderr_iso_list[~np.isnan(now_delay_iso_list)]
        cutted_stderr_aniso = now_delay_stderr_aniso_list[~np.isnan(now_delay_iso_list)]

        now_delay_iso_list = []
        now_delay_aniso_list = [] # delete data for saving memory

        normalization_range = cutted_iso[(cutted_q_val >= 1.5) & (cutted_q_val <= 3.5)]
        norm_range_iso_sum = np.sum(normalization_range)
        normalization_range = []
        # print(normalization_range)

        self.each_delay_cutted_q.append(cutted_q_val)
        self.each_delay_cutted_iso.append(cutted_iso)
        self.each_delay_cutted_aniso.append(cutted_aniso)
        self.each_delay_cutted_stderr_iso.append(cutted_stderr_iso)
        self.each_delay_cutted_stderr_aniso.append(cutted_stderr_aniso)

        # if result_plot:
        #     plt.title("isotropic signal of " + str(self.now_delay_idx) + "-th delay")
        #     plt.scatter(cutted_q_val, cutted_iso)
        #     plt.errorbar(cutted_q_val, cutted_iso, yerr=cutted_stderr_iso, linestyle='None', ecolor='k', capsize=2, marker='o', markersize=1.5)
        #     plt.xlabel("Q (A^-1)")
        #     plt.ylabel("dS_0")
        #     plt.show()
        #
        #     plt.title("isotropic signal of " + str(self.now_delay_idx) + "-th delay")
        #     plt.scatter(cutted_q_val, cutted_iso)
        #     plt.axhline(norm_range_iso_sum)
        #     plt.xlabel("Q (A^-1)")
        #     plt.ylabel("dS_0")
        #     plt.show()
        #
        # plt.title("anisotropic signal of " + str(self.now_delay_idx) + "-th delay")
        # plt.scatter(cutted_q_val, cutted_aniso)
        # plt.errorbar(cutted_q_val, cutted_aniso, yerr=cutted_stderr_aniso, linestyle='None', ecolor='k', capsize=2, marker='o', markersize=1.5)
        # plt.xlabel("Q (A^-1)")
        # plt.ylabel("dS_2")
        # plt.show()

    @staticmethod
    def anisotropy_fit(twotheta_rad, phi_deg, azi_intg_data, q_val, q_idx, test_plot=False):
        test_plot_idx = 200
        # cos theta_q = - cos theta(rad) * cos phi(rad)
        cos_theta_q = -np.cos(twotheta_rad / 2) * np.cos(np.deg2rad(phi_deg))
        # P_2(x) = (3x^2 - 1) /2
        x_data = 0.5 * (3 * np.power(cos_theta_q, 2) - 1)
        y_data = azi_intg_data

        x_cutted_data = x_data[y_data != 0]
        y_cutted_data = y_data[y_data != 0]

        if len(y_cutted_data) == 0:
            return np.nan, np.nan, np.nan, np.nan

        regResult = linregress(x_cutted_data, y_cutted_data)

        ds0_isotropic = regResult.intercept
        ds2_anisotropic = regResult.slope
        iso_stderr = regResult.intercept_stderr
        aniso_stderr = regResult.stderr

        if test_plot and (q_idx == test_plot_idx):
            # print("test plot q pos y data", y_data)
            # print("test plot q pos cutted y data", y_cutted_data)

            # plt.title("P_2")
            # plt.scatter(phi_deg, x_data)
            # plt.xlabel("phi (deg)")
            # plt.ylabel("P_2")
            # plt.show()

            fit_line_x = np.linspace(min(x_cutted_data), max(x_cutted_data), 100)
            fit_line_y = regResult.slope * fit_line_x + regResult.intercept

            plt.title("dS at Q = " + str(q_val) +" A^-1")
            plt.scatter(x_cutted_data, y_cutted_data)
            plt.plot(fit_line_x, fit_line_y, color='r')
            plt.xlabel("P_2")
            plt.ylabel("dS")
            plt.show()

        return ds0_isotropic, ds2_anisotropic, iso_stderr, aniso_stderr

    def set_azimuthal_integrator(self):
        sd_dist_in_m = self.sample_detector_dist * 1e-3  # unit transfer mm -> m
        beam_center_x_in_m = self.beam_center_x * self.pixel_size
        beam_center_y_in_m = self.beam_center_y * self.pixel_size
        # temp_detector = pyFAI.detector_factory("RayonixMx225hs")  # Rayonix MX-225HS
        now_detector = pyFAI.detectors.Detector(pixel1=self.pixel_size, pixel2=self.pixel_size, max_shape=(self.detector_num_pixel, self.detector_num_pixel))
        azi_intg = pyFAI.azimuthalIntegrator.AzimuthalIntegrator(dist=sd_dist_in_m,
                                                                 poni1=beam_center_y_in_m, poni2=beam_center_x_in_m,
                                                                 wavelength=self.xray_wavelength_in_m, detector=now_detector)
        print("here")
        return azi_intg

    def multi_aniso_plot(self):
        for delay_idx, each_aniso_val in enumerate(self.each_delay_cutted_aniso):
            plt.plot(self.each_delay_cutted_q[delay_idx], each_aniso_val, label=str(delay_idx+1)+"-th idx")
        plt.title("anisotropic signal of multiple delay")
        plt.xlabel("Q (A^-1)")
        plt.ylabel("dS_2")
        plt.legend()
        plt.show()

    def now_delay_file_out(self, now_time, idx_delay, run_num, merge_run_list = [], merge_multi_run = False):
        if merge_multi_run:
            file_save_root = "../results/anisotropy/anal_result/run{0}_{1}/run{0}_{1}_{3}/run{0}_{1}_delay{2}".format(merge_run_list[0],merge_run_list[1],idx_delay + 1, now_time) #TODO might be better if save each file in one folder
        else:
            # file_save_root = "../results/anisotropy/anal_result/run_{0:05d}/run{0}_{2}/run{0}_delay{1}".format(run_num, idx_delay + 1, now_time)
            file_save_root = "../results/anisotropy/anal_result/run_{0:05d}/run{0}_delay{1}".format(run_num, idx_delay + 1)

        q_val_file_name = file_save_root + "_qval"
        iso_file_name = file_save_root + "_iso"
        aniso_file_name = file_save_root + "_aniso"
        iso_stderr_file_name = file_save_root + "_stderr_iso"
        aniso_stderr_file_name = file_save_root + "_stderr_aniso"

        np.save(q_val_file_name, self.each_delay_cutted_q[0])
        np.save(iso_file_name, self.each_delay_cutted_iso[0])
        np.save(aniso_file_name, self.each_delay_cutted_aniso[0])
        np.save(iso_stderr_file_name, self.each_delay_cutted_stderr_iso[0])
        np.save(aniso_stderr_file_name, self.each_delay_cutted_stderr_aniso[0])

        print("successful result save : delay " + str(idx_delay + 1) + " - th (1-based)")