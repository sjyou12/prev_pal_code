import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares
import os


para_unit = {'FWHM': 1E-1 / 2.35482, 'x0': 1, 't1': 1, 't2': 1, 'a1': 1, 'a2': 1}


class MinuitParam:
    def __init__(self, p_name, p_limit, p_is_fixed, p_max_oom, p_min_oom, p_fixed_value, no_random_initial):
        self.name = p_name  # string
        self.is_fixed = p_is_fixed
        self.fixed_value = p_fixed_value
        self.limit = p_limit  # range a ~ param ~ b -> (a, b)
        self.fit_results = []
        self.max_order_of_magnitude = p_max_oom
        self.min_order_of_magnitude = p_min_oom
        self.no_random_initial = no_random_initial


class ScattAnalyzer:
    def __init__(self):
        self.ScatteringFunc = None
        self.param_obj_list = []
        self.random_param_inits = []

        self.input_time_delay = []
        self.input_raw_data = []
        self.LC_chi2 = []
        self.input_raw_data_len = 0
        self.calculate_time_list = []
        self.original_calc_time_list = []
        self.time_step = 0
        self.time_delays = []
        self.exp_calc_time_delay = []
        self.num_rsv_to_fit = 0
        self.sample_LSV = []
        self.input_raw_q_val = []
        self.iso_heating = None
        self.iso_second = None
        self.anisotropy_process = None
        self.isotropy_process = None
        self.aniso_single_exp = None
        self.aniso_double_exp = None
        self.aniso_stretched_exp = None
        self.aniso_p_gallo = None
        self.aniso_cos_square_fit = None
        self.aniso_single_exp_without_conv = None

        self.weighted_average = None
        self.temporary_mask = None
        self.mask_list = [] #list for saving start & end point
        self.mask_start_idx = None
        self.mask_end_idx = None
        # self.aniso_cut_gaussian = None

        self.input_common_root = None
        self.infile_name = None
        self.input_file = None
        self.previous_dat_load_path = None

        self.plot_for_save = []
        self.best_params_save = []
        self.best_chi_square_save = None
        self.best_params_erros = []
        self.num_log_out = 0
        self.log_out_file_common_name = None
        self.now_time = []
        self.runNum = 0
        self.runList = []
        self.file_out_path = None
        self.file_common_name = None
        self.dat_file_time_stamp = None
        self.finish_minuit = None
        self.aniso_rsv_max_val = 0
        self.material_run_list = []
        self.iso_LC = None
        self.material_is_droplet = None

    def set_scattering_function(self, givenFunc):
        self.ScatteringFunc = givenFunc

    def set_anal_data_as_file(self, input_file_common_root, given_file_name, num_rsv_to_fit, sing_val = False, weighted_average=False):
        self.input_common_root = input_file_common_root
        self.infile_name = given_file_name
        file_path = input_file_common_root + given_file_name
        self.num_rsv_to_fit = num_rsv_to_fit
        self.weighted_average = weighted_average
        print("now read file : ", file_path)
        self.input_file = open(file_path, 'r')
        self.read_input_file(sing_val)

    def read_input_file(self, sing_val):
        front_skip_line_num = 1
        is_front_line_skipped = False
        file_time_val = []
        file_data = []
        front_skipped_data = None
        line_idx = 0
        ref_line_split_text = []

        while 1:
            now_input_file_read_data = self.input_file.readlines(10000)
            if not now_input_file_read_data:
                break
            if not is_front_line_skipped:
                front_skipped_data = now_input_file_read_data[:front_skip_line_num]
                now_input_file_read_data = now_input_file_read_data[front_skip_line_num:]
                is_front_line_skipped = True
            for each_read_line in now_input_file_read_data:
                now_split_text = each_read_line.split()
                now_split_float = []
                if line_idx == 0:
                    line_idx += 1
                else:
                    line_idx += 1
                if line_idx == 3:
                    ref_line_split_text = now_split_text
                if (len(ref_line_split_text) != len(now_split_text) and (line_idx > 2)):
                    break
                for each_split_data in now_split_text:
                    if each_split_data == 'singVal1' or each_split_data == 'singVal2':
                        break
                    else:
                        now_split_float.append(float(each_split_data))
                file_time_val.append(now_split_float[0])
                # file_data.append(now_split_float[1:])
                try:
                    file_data.append(now_split_float[self.num_rsv_to_fit])
                    self.LC_chi2.append(now_split_float[-1])
                except:
                    break
        self.input_time_delay = np.array(file_time_val)
        # self.input_time_delay = np.round(self.input_time_delay * 1E-12, 15)
        self.input_raw_data = np.array(file_data)
        if np.min(self.input_raw_data) < (-0.3):
            self.input_raw_data = -self.input_raw_data
        self.input_raw_data_len = len(file_time_val)
        # self.front_delay_info_read(front_skipped_data)
        if self.anisotropy_process:
            if (self.aniso_double_exp or self.aniso_p_gallo or self.temporary_mask or self.aniso_single_exp_without_conv) and (self.num_rsv_to_fit == 1):
                for time_idx in range(len(self.input_time_delay)):
                    if np.max(self.input_time_delay) >= 10:
                        # if self.input_time_delay[time_idx] == 3:
                        #     self.input_time_delay = self.input_time_delay[:(time_idx+1)]
                        #     self.input_raw_data = self.input_raw_data[:(time_idx + 1)]
                        if self.temporary_mask:
                            if (self.input_time_delay[time_idx] == self.mask_list[0]):
                                self.mask_start_idx = time_idx
                            elif (self.input_time_delay[time_idx] == self.mask_list[1]):
                                self.mask_end_idx = time_idx

                        if self.input_time_delay[time_idx] == 3.2:
                            exp_calc_time_start_time_idx = time_idx
                        if self.input_time_delay[time_idx] == 10 and self.input_time_delay[-1] == 10: #TODO change after done run 35
                            self.input_time_delay = self.input_time_delay[:(time_idx+1)]
                            self.input_raw_data = self.input_raw_data[:(time_idx+1)]
                            self.exp_calc_time_delay = self.input_time_delay[exp_calc_time_start_time_idx:(time_idx+1)]
                            self.LC_chi2 = self.LC_chi2[:(time_idx)+1]
                            break
                        elif self.input_time_delay[time_idx] == 30:
                            self.input_time_delay = self.input_time_delay[:(time_idx+1)]
                            self.input_raw_data = self.input_raw_data[:(time_idx+1)]
                            self.exp_calc_time_delay = self.input_time_delay[exp_calc_time_start_time_idx:(time_idx+1)]
                            self.LC_chi2 = self.LC_chi2[:(time_idx)+1]
                            break
                        elif self.input_time_delay[time_idx] == self.input_time_delay[-1]:
                            self.input_time_delay = self.input_time_delay[:(time_idx+1)]
                            self.input_raw_data = self.input_raw_data[:(time_idx+1)]
                            self.exp_calc_time_delay = self.input_time_delay[exp_calc_time_start_time_idx:(time_idx+1)]
                            self.LC_chi2 = self.LC_chi2[:(time_idx)+1]
                    else:
                        if self.temporary_mask:
                            if self.input_time_delay[time_idx] == self.mask_list[0]:
                                self.mask_start_idx = time_idx
                            elif self.input_time_delay[time_idx] == self.mask_list[1]:
                                self.mask_end_idx = time_idx
                        if self.input_time_delay[time_idx] == 3:
                            self.input_time_delay = self.input_time_delay[:(time_idx+1)]
                            self.input_raw_data = self.input_raw_data[:(time_idx + 1)]
                            break

            elif self.aniso_cos_square_fit:
                if self.aniso_cos_square_fit:
                    max_val = np.max(self.input_raw_data)
                    self.input_raw_data = (self.input_raw_data / max_val) * 0.01

            elif self.aniso_double_exp and (self.num_rsv_to_fit == 2):
                for time_idx in range(len(self.input_time_delay)):
                    if np.max(self.input_time_delay) >= 10:
                        if self.input_time_delay[time_idx] == 3:
                            exp_calc_time_start_time_idx = time_idx
                        if self.input_time_delay[time_idx] == 50: #TODO change after done run 35
                            self.input_time_delay = self.input_time_delay[:(time_idx+1)]
                            self.input_raw_data = self.input_raw_data[:(time_idx+1)]
                            self.exp_calc_time_delay = self.input_time_delay[exp_calc_time_start_time_idx:(time_idx+1)]
                            self.LC_chi2 = self.LC_chi2[:(time_idx)+1]
                            if (self.num_rsv_to_fit == 2) and (self.input_raw_data[-1] < 0):
                                self.input_raw_data = -self.input_raw_data
                            # elif (self.num_rsv_to_fit == 2) and (self.runNum == 68):
                            #     self.input_raw_data = -self.input_raw_data
                            break
                        elif self.input_time_delay[time_idx] == 10 and self.input_time_delay[-1] == 10:
                            self.input_time_delay = self.input_time_delay[:(time_idx+1)]
                            self.input_raw_data = self.input_raw_data[:(time_idx+1)]
                            self.exp_calc_time_delay = self.input_time_delay[exp_calc_time_start_time_idx:(time_idx+1)]
                            self.LC_chi2 = self.LC_chi2[:(time_idx)+1]
                            max_idx = np.argmax(np.abs(self.input_raw_data))
                            if (self.num_rsv_to_fit == 2) and (self.input_raw_data[max_idx] < 0):
                                self.input_raw_data = -self.input_raw_data
                        else:
                            continue
                    else:
                        if self.input_time_delay[time_idx] == 3:
                            self.input_time_delay = self.input_time_delay[:(time_idx+1)]
                            self.input_raw_data = self.input_raw_data[:(time_idx + 1)]
                            if (self.num_rsv_to_fit == 2) and (self.input_raw_data[-1] < 0):
                                self.input_raw_data = -self.input_raw_data
                            elif (self.num_rsv_to_fit == 2) and (self.runNum == 68):
                                self.input_raw_data = -self.input_raw_data
                            # if self.input_time_delay[time_idx] == 3E-12:
                            # exp_calc_time_start_time_idx = time_idx
                        # if self.input_time_delay[time_idx] == 10: #TODO change after done run 35
                        #     self.input_time_delay = self.input_time_delay[:(time_idx+1)]
                        #     self.input_raw_data = self.input_raw_data[:(time_idx+1)]
                        #     self.exp_calc_time_delay = self.input_time_delay[exp_calc_time_start_time_idx:(time_idx+1)]
                            break
                        else:
                            continue
            elif self.aniso_single_exp and (self.num_rsv_to_fit == 2):
                for time_idx in range(len(self.input_time_delay)):
                    if np.max(self.input_time_delay) >= 10:
                        if self.input_time_delay[time_idx] == 3:
                            exp_calc_time_start_time_idx = time_idx
                        if self.input_time_delay[time_idx] == 50: #TODO change after done run 35
                            self.input_time_delay = self.input_time_delay[:(time_idx+1)]
                            self.input_raw_data = self.input_raw_data[:(time_idx+1)]
                            self.exp_calc_time_delay = self.input_time_delay[exp_calc_time_start_time_idx:(time_idx+1)]
                            self.LC_chi2 = self.LC_chi2[:(time_idx)+1]
                            if (self.num_rsv_to_fit == 2) and (self.input_raw_data[-1] < 0):
                                self.input_raw_data = -self.input_raw_data
                            # elif (self.num_rsv_to_fit == 2) and (self.runNum == 68):
                            #     self.input_raw_data = -self.input_raw_data
                            break
                        elif self.input_time_delay[time_idx] == 10 and self.input_time_delay[-1] == 10:
                            self.input_time_delay = self.input_time_delay[:(time_idx+1)]
                            self.input_raw_data = self.input_raw_data[:(time_idx+1)]
                            self.exp_calc_time_delay = self.input_time_delay[exp_calc_time_start_time_idx:(time_idx+1)]
                            self.LC_chi2 = self.LC_chi2[:(time_idx)+1]
                            max_idx = np.argmax(np.abs(self.input_raw_data))
                            if (self.num_rsv_to_fit == 2) and (self.input_raw_data[max_idx] < 0):
                                self.input_raw_data = -self.input_raw_data
                        else:
                            continue
                    else:
                        if self.input_time_delay[time_idx] == 3:
                            self.input_time_delay = self.input_time_delay[:(time_idx+1)]
                            self.input_raw_data = self.input_raw_data[:(time_idx + 1)]
                            max_idx = np.argmax(np.abs(self.input_raw_data))
                            if (self.num_rsv_to_fit == 2) and (self.input_raw_data[max_idx] < 0):
                                self.input_raw_data = -self.input_raw_data
                            break
                        else:
                            continue
            elif self.aniso_stretched_exp:
                # for time_idx in range(len(self.input_time_delay)):
                #     if self.input_time_delay[time_idx] == 3:  # TODO change after done run 35
                #         self.input_time_delay = self.input_time_delay[:(time_idx + 1)]
                #         self.input_raw_data = self.input_raw_data[:(time_idx + 1)]
                #         break
                #     else:
                #         continue
                for time_idx in range(len(self.input_time_delay)):
                    if np.max(self.input_time_delay) > 100:
                        if self.input_time_delay[time_idx] == 3:
                            # self.input_time_delay = self.input_time_delay[:(time_idx+1)]
                            # self.input_raw_data = self.input_raw_data[:(time_idx + 1)]
                            # if self.input_time_delay[time_idx] == 3E-12:
                            exp_calc_time_start_time_idx = time_idx
                        if self.input_time_delay[time_idx] == 10: #TODO change after done run 35
                            self.input_time_delay = self.input_time_delay[:(time_idx+1)]
                            self.input_raw_data = self.input_raw_data[:(time_idx+1)]
                            self.exp_calc_time_delay = self.input_time_delay[exp_calc_time_start_time_idx:(time_idx+1)]
                            if (self.num_rsv_to_fit == 2) and (self.input_raw_data[-1] < 0):
                                self.input_raw_data = -self.input_raw_data
                            elif (self.num_rsv_to_fit == 2) and (self.runNum == 68):
                                self.input_raw_data = -self.input_raw_data
                            break
                        else:
                            continue
                    else:
                        if self.input_time_delay[time_idx] == 3:
                            self.input_time_delay = self.input_time_delay[:(time_idx+1)]
                            self.input_raw_data = self.input_raw_data[:(time_idx + 1)]
                            # if self.input_time_delay[time_idx] == 3E-12:
                            # exp_calc_time_start_time_idx = time_idx
                        # if self.input_time_delay[time_idx] == 10: #TODO change after done run 35
                        #     self.input_time_delay = self.input_time_delay[:(time_idx+1)]
                        #     self.input_raw_data = self.input_raw_data[:(time_idx+1)]
                        #     self.exp_calc_time_delay = self.input_time_delay[exp_calc_time_start_time_idx:(time_idx+1)]
                            break
                        else:
                            continue

        elif self.iso_heating:
            for time_idx in range(len(self.input_time_delay)):
                if self.input_time_delay[time_idx] > 30: #TODO change after done run 35
                    self.input_time_delay = self.input_time_delay[:time_idx]
                    self.input_raw_data = self.input_raw_data[:time_idx]
                    break
                else:
                    continue
        if sing_val:
            sing_val_root = self.input_common_root + "/run{0}-iso-cut_SingVal.dat".format(self.runNum)
            sing_val_file = open(sing_val_root, 'r')
            sing_val_list = sing_val_file.readlines(100)
            target_sing_val = sing_val_list[self.num_rsv_to_fit]
            self.input_raw_data = self.input_raw_data * float(target_sing_val)
        file_data = []

    def read_input_file_for_LC(self, input_file_common_root, material_file_time_stamp, material_run_list, material_sole_run, iso_LC=False, material_is_droplet=False, weighted_average=False):
        self.input_common_root = input_file_common_root
        self.weighted_average = weighted_average
        self.iso_LC = iso_LC
        self.material_is_droplet = material_is_droplet
        # print("now read file : ", data_file_path)
        if weighted_average:
            if len(self.runList) == 2:
                delay_file_name = "run{0:04d}_{1:04d}_avg_".format(self.runList[0], self.runList[1])
            elif len(self.runList) == 3:
                delay_file_name = "run{0:04d}_{1:04d}_{2:04d}_avg_".format(self.runList[0], self.runList[1], self.runList[2])
        else:
            delay_file_name = "run{0}_".format(self.runNum)
        delay_num = 1
        while 1:
            if self.runNum == 65 and (delay_num in [38, 39]):
                self.input_raw_data.append(np.zeros(60))
                delay_num += 1
            else:
                try:
                    if iso_LC:
                        target_data = np.load(input_file_common_root + delay_file_name + "delay{0}_iso.npy".format(delay_num))
                    else:
                        target_data = np.load(input_file_common_root + delay_file_name + "delay{0}_aniso.npy".format(delay_num))
                    self.input_raw_data.append(target_data[6:66])
                    delay_num += 1
                except:
                    break
        if iso_LC:
            temp_sample_1 = np.loadtxt("../results/anisotropy/svd/run{0:04d}_{1:04d}_avg/run{0}_{1}_avg_{2}/run{0:04d}_{1:04d}_avg-iso-cut_LSV.dat".format(material_run_list[0], material_run_list[1], material_file_time_stamp[0]), skiprows=1)
            # temp_sample_2 = np.loadtxt("../results/anisotropy/svd/run{0:04d}/run{0}_{1}/run{0}-aniso-cut_LSV.dat".format(material_sole_run, material_file_time_stamp[1]), skiprows=1)
            # average_aniso_delay = []
            # for delay_idx in range(10):
            #     average_aniso_delay.append(np.load("../results/anisotropy/anal_result/run{0:04d}/run{0}_{1}/run{0}_delay{2}_aniso.npy".format(material_sole_run, material_file_time_stamp[1], delay_idx + 32)))
            # average_aniso_delay = np.average(average_aniso_delay, axis=0)
            # temp_max_aniso_delay = np.load("../results/anisotropy/anal_result/run{0:04d}/run{0}_{1}/run{0}_delay12_aniso.npy".format(material_sole_run, material_file_time_stamp[1]))
            # averaged_LSV1 = np.average((temp_sample_1[:,1], temp_sample_2[:,1]), axis=0)
            # averaged_LSV2 = np.average((temp_sample_1[:,2], temp_sample_2[:,2]), axis=0)
            self.sample_LSV.append(temp_sample_1[:, 1])
            self.sample_LSV.append(temp_sample_1[:, 2])
            # self.sample_LSV.append(temp_sample_2[:, 1])
            # self.sample_LSV.append(temp_sample_2[:, 2])
            if weighted_average:
                if len(self.runList) == 2:
                    temp_rsv = np.loadtxt("../results/anisotropy/svd/run{0:04d}_{1:04d}_avg/run{0}_{1}_avg_{2}/run{0:04d}_{1:04d}_avg-iso-cut_RSV.dat".format(self.runList[0], self.runList[1], self.dat_file_time_stamp), skiprows=1)
                elif len(self.runList) == 3:
                    temp_rsv = np.loadtxt("../results/anisotropy/svd/run{0:04d}_{1:04d}_{2:04d}_avg/run{0}_{1}_{2}_avg_{3}/run{0:04d}_{1:04d}_{2:04d}_avg-iso-cut_RSV.dat".format(self.runList[0], self.runList[1], self.runList[2], self.dat_file_time_stamp), skiprows=1)
            else:
                temp_rsv = np.loadtxt("../results/anisotropy/svd/run{0:04d}/run{0}_{1}/run{0}-iso-cut_RSV.dat".format(self.runNum, self.dat_file_time_stamp), skiprows=1)
        else:
            if material_is_droplet:
                temp_sample_1 = np.loadtxt("../results/anisotropy/svd/run{0:04d}_{1:04d}_avg/run{0}_{1}_avg_{2}/run{0:04d}_{1:04d}_avg-aniso-cut_LSV.dat".format(material_run_list[0], material_run_list[1], material_file_time_stamp[0]), skiprows=1)
                self.sample_LSV.append(temp_sample_1[:, 1])
                self.sample_LSV.append(temp_sample_1[:, 2])
            else:
                temp_sample_2 = np.loadtxt("../results/anisotropy/svd/run{0:04d}/run{0}_{1}/run{0}-aniso-cut_LSV.dat".format(material_sole_run, material_file_time_stamp[1]), skiprows=1)
                self.sample_LSV.append(temp_sample_2[:, 1])
                self.sample_LSV.append(temp_sample_2[:, 2])

            if weighted_average:
                if len(self.runList) == 2:
                    temp_rsv = np.loadtxt("../results/anisotropy/svd/run{0:04d}_{1:04d}_avg/run{0}_{1}_avg_{2}/run{0:04d}_{1:04d}_avg-aniso-cut_RSV.dat".format(self.runList[0], self.runList[1], self.dat_file_time_stamp), skiprows=1)
                elif len(self.runList) == 3:
                    temp_rsv = np.loadtxt("../results/anisotropy/svd/run{0:04d}_{1:04d}_{2:04d}_avg/run{0}_{1}_{2}_avg_{3}/run{0:04d}_{1:04d}_{2:04d}_avg-aniso-cut_RSV.dat".format(self.runList[0], self.runList[1], self.runList[2], self.dat_file_time_stamp), skiprows=1)
            else:
                temp_rsv = np.loadtxt("../results/anisotropy/svd/run{0:04d}/run{0}_{1}/run{0}-aniso-cut_RSV.dat".format(self.runNum, self.dat_file_time_stamp), skiprows=1)
        temp_q_val = np.load("../results/anisotropy/anal_result/run{0:04d}/run{0}_{1}/run{0}_delay12_qval.npy".format(material_sole_run, material_file_time_stamp[1]))
        self.input_raw_q_val = (temp_q_val[6:66])
        self.input_time_delay = temp_rsv[:, 0]
        # self.aniso_rsv_max_val = np.max(temp_rsv[:, 11])
        self.material_run_list = material_run_list

    def data_from_given_arr(self, q_val, anal_data):
        self.input_raw_q_val = np.array(q_val)
        self.input_raw_data = np.array(anal_data)

        self.input_raw_data_len = len(self.input_raw_q_val)

    def front_delay_info_read(self, front_skipped_data):
        first_line_data = front_skipped_data[0]
        split_info = first_line_data.split()
        print(split_info)
        time_delay_list = []
        for each_time_text in split_info[1:]:
            time_delay_list.append(float(each_time_text))
        print(time_delay_list)
        self.time_delays = time_delay_list

    def set_fit_param(self, param_name, left_limit=None, right_limit=None, is_fixed=False, max_oom=None, min_oom=None, no_random_initial=False, fixed_value=0):
        if (max_oom is not None) and (min_oom is None):
            raise ValueError('Need both max_order_of_magnitude and min_order_of_magnitude')
        if (min_oom is not None) and (max_oom is None):
            raise ValueError('Need both max_order_of_magnitude and min_order_of_magnitude')
        if (max_oom is None) and (min_oom is None):
            if (left_limit is None) and (right_limit is None):
                raise ValueError('Need one of "limit range" and "oom range"')
        newParam = MinuitParam(param_name, (left_limit, right_limit), is_fixed, max_oom, min_oom, fixed_value, no_random_initial)
        self.param_obj_list.append(newParam)

    def make_random_param_init(self, random_try_num):
        param_num = len(self.param_obj_list)
        random_arr = np.random.rand(param_num, random_try_num)

        transfer_rand_arr_list = []
        for idx_param, each_param in enumerate(self.param_obj_list):
            now_max_oom = each_param.max_order_of_magnitude
            now_min_oom = each_param.min_order_of_magnitude
            now_l_lim, now_r_lim = each_param.limit

            # possible case of oom & limit
            # 1) both oom is given
            # 2) both oom is not given & both limit is given

            if (now_min_oom is not None) and (now_max_oom is not None):
                # case 1] both oom is given
                random_exponents = random_arr[idx_param] * (now_max_oom - now_min_oom) + now_min_oom
                if each_param.is_fixed or each_param.no_random_initial:
                    x = np.arange(random_try_num, dtype=float)
                    transfer_rand_arr = np.full_like(x, each_param.fixed_value)
                else:
                    if idx_param % 2 == 0:
                        transfer_rand_arr = np.power(10, random_exponents)
                    elif idx_param % 2 == 1:
                        transfer_rand_arr = -np.power(10, random_exponents)
            else:
                # case 2] both oom is not given, but both limit is given
                if each_param.is_fixed or each_param.no_random_initial:
                    # transfer_rand_arr = np.full_like(each_param.fixed_value, random_try_num)
                    x = np.arange(random_try_num, dtype=float)
                    transfer_rand_arr = np.full_like(x, each_param.fixed_value)
                else:
                    transfer_rand_arr = random_arr[idx_param] * (now_r_lim - now_l_lim) + now_l_lim
            transfer_rand_arr_list.append(transfer_rand_arr)
        random_inits = np.array(transfer_rand_arr_list)
        self.random_param_inits = random_inits

    def random_initial_fit(self, random_try_num=10, plot_min_chi_square=False, log_print_per_try=100, plot_skip=True):
        # if self.aniso_cos_square_fit:
        #     data_yerr = 1
        # else:
        #     data_yerr = self.LC_chi2
        data_yerr = 1

        # assertion for no data is given
        assert len(self.input_time_delay) != 0, "No input time delays"
        assert len(self.input_raw_data) != 0, "No input data to analyze"

        now_delay_exp_data = self.input_raw_data
        common_time_delay = self.input_time_delay

        # assert self.ScatteringFunc is not None
        if self.anisotropy_process:
            if self.aniso_single_exp or self.temporary_mask:
                exponential_least_square = LeastSquares(common_time_delay, now_delay_exp_data, data_yerr, self.single_exponential_function_convolution)
            elif self.aniso_double_exp:
                # if self.num_rsv_to_fit == 2 :
                #     exponential_least_square = LeastSquares(common_time_delay, now_delay_exp_data, data_yerr, self.heating_convolution)
                # else:
                    exponential_least_square = LeastSquares(common_time_delay, now_delay_exp_data, data_yerr, self.new_convolution_double_exp)
            elif self.aniso_stretched_exp:
                exponential_least_square = LeastSquares(common_time_delay, now_delay_exp_data, data_yerr, self.stretched_exponential_function_convolution)
            elif self.aniso_p_gallo:
                exponential_least_square = LeastSquares(common_time_delay, now_delay_exp_data, data_yerr, self.p_gallo_function_convolution)
            elif self.aniso_cos_square_fit:
                exponential_least_square = LeastSquares(common_time_delay, now_delay_exp_data, data_yerr, self.exp_decay)
            elif self.aniso_single_exp_without_conv:
                fit_start_idx = np.where(common_time_delay == self.mask_list[1])[0][0]
                exponential_least_square = LeastSquares(common_time_delay, now_delay_exp_data, data_yerr, self.fixed_t0_single_exp)

        elif self.isotropy_process:
            if self.iso_heating:
                exponential_least_square = LeastSquares(common_time_delay, now_delay_exp_data, data_yerr, self.exp_heating)
            elif self.iso_second:
                exponential_least_square = LeastSquares(common_time_delay, now_delay_exp_data, data_yerr, self.new_convolution_double_exp)
        param_limits = [param.limit for param in self.param_obj_list]

        self.make_random_param_init(random_try_num)

        result_chi_squares = []
        result_param_values = []
        result_param_errors = []
        fit_result_tracker = []

        try:
            previous_dat_file = open(self.previous_dat_load_path + ".dat", "r")
            read_line_num = 0
            while 1:
                now_input_file_read_data = previous_dat_file.readlines(10000)
                if not now_input_file_read_data:
                    break
                for each_read_line in now_input_file_read_data:
                    now_split_text = each_read_line.split()
                    read_line_num += 1
                    if read_line_num == 1:
                        continue
                    else:
                        if now_split_text[0] == 'chi2':
                            result_chi_squares.append(float(now_split_text[1]))
                        else:
                            if read_line_num == (len(self.param_obj_list) + 3):
                                break
                            else:
                                if not result_param_values:
                                    result_param_values.append([float(now_split_text[1])])
                                    result_param_errors.append([float(now_split_text[2])])
                                else:
                                    result_param_values[0].append(float(now_split_text[1]))
                                    result_param_errors[0].append(float(now_split_text[2]))
                    # result_param_values[0] = np.array(result_param_values[0])
                temp_params = np.array(result_param_values[0])
                result_param_values[0] = temp_params
                temp_errors = np.array(result_param_errors[0])
                result_param_errors[0] = temp_errors
                break
                        # now_split_float.append(float(each_split_data))
                    # result_chi_squares.append(now_split_float[1][1])
                    # file_data.append(now_split_float[1:])
                    # try:
                    #     file_data.append(now_split_float[self.num_rsv_to_fit])
                    # except:
                    #     break
        except:
            pass

        num_fit_param = 0
        param_names = []

        for idx_try in range(random_try_num):
            assert len(self.random_param_inits) != 0
            now_initials = self.random_param_inits[:, idx_try]
            m = Minuit(exponential_least_square, *now_initials)
            # m.errors = [1E-5, 1E-5, 1E-5, 1E-5, 1E-5, 1E-5, 1E-5]
            m.errors = [1E-5 for idx in range(len(self.param_obj_list))]
            m.limits = param_limits
            for param_idx, param in enumerate(self.param_obj_list):
                if param.is_fixed:
                    m.fixed[param.name] = True
                else:
                    pass
            # m.fixed['FWHM', 'x0'] = True
            if self.temporary_mask:
                exponential_least_square.mask = (common_time_delay < common_time_delay[self.mask_start_idx]) | (common_time_delay > common_time_delay[self.mask_end_idx])
            elif self.aniso_single_exp_without_conv:
                exponential_least_square.mask = (common_time_delay >= common_time_delay[fit_start_idx])
            m.simplex()
            m.migrad()
            m.hesse()
            # print(m.params)
            # print(m.valid)

            if m.valid:
                result_chi_squares.append(m.fval)
                result_param_values.append(np.array(m.values))
                result_param_errors.append(np.array(m.errors))


            if idx_try == 0:
                num_fit_param = m.nfit
                param_names = m.parameters

            # if idx_try % log_print_per_try == (log_print_per_try - 1):
                # if plot_min_chi_square:
                #     min_chi_square_until_now = np.min(result_chi_squares)
                #     print("now {0} time fitting. min chi square value : {1:3.8e}".format(idx_try + 1, min_chi_square_until_now))
                #     fit_result_tracker.append([idx_try + 1, min_chi_square_until_now])
                # self.write_log_info(idx_try, result_chi_squares)

        if result_chi_squares:
            best_run_idx = np.nanargmin(result_chi_squares)
            best_chi_square = result_chi_squares[best_run_idx]
            best_param = result_param_values[best_run_idx]
            best_param_error = result_param_errors[best_run_idx]
        else:
            print("None of them are valid")
            return

        self.best_params_save = best_param
        self.best_chi_square_save = best_chi_square
        self.best_params_erros = best_param_error
        self.finish_minuit = True

        # print("chi_square values :")
        # print(result_chi_squares)
        print("best run idx : {0} -> chi square value : {1:3.8e}".format(best_run_idx, best_chi_square))
        print("parameters :", best_param, "(error : ", best_param_error)

        if self.anisotropy_process:
            if self.aniso_single_exp or self.temporary_mask:
                self.plot_for_save = self.single_exponential_function_convolution_to_final_fit(self.calculate_time_list, *best_param)
            elif self.aniso_double_exp:
                self.plot_for_save = self.new_convolution_double_exp_to_final_fit(self.calculate_time_list, *best_param)
            elif self.aniso_stretched_exp:
                self.plot_for_save = self.stretched_exponential_function_convolution(self.calculate_time_list, *best_param)
            elif self.aniso_p_gallo:
                self.plot_for_save = self.p_gallo_function_convolution(self.calculate_time_list, *best_param)
            elif self.aniso_cos_square_fit:
                self.plot_for_save = self.exp_decay(self.calculate_time_list, *best_param)
            elif self.aniso_single_exp_without_conv:
                self.plot_for_save = self.fixed_t0_single_exp(self.original_calc_time_list, *best_param)


        elif self.isotropy_process:
            if self.iso_second:
                self.plot_for_save = self.new_convolution_double_exp_to_final_fit(self.calculate_time_list, *best_param)
            elif self.iso_heating:
                self.plot_for_save = self.exp_heating(self.calculate_time_list, *best_param)

        if not plot_skip:
            # test plots
            plt.plot(common_time_delay, now_delay_exp_data, label="data")
            plt.plot(common_time_delay, self.ScatteringFunc(common_time_delay, *best_param), label="best fit")

            # display legend with some fit info
            fit_info = [
                f"$\\chi^2$ / $n_\\mathrm{{dof}}$ = {best_chi_square:.1f} / {len(common_time_delay) - num_fit_param}",
            ]
            for p, v, e in zip(param_names, best_param, best_param_error):
                fit_info.append(f"{p} = ${v:.3f} \\pm {e:.3f}$")

            plt.legend(title="\n".join(fit_info))
            # plt.ylim((-50, 50))
            plt.show()

            if plot_min_chi_square:
                fit_result_tracker = np.transpose(np.array(fit_result_tracker))
                plt.plot(fit_result_tracker[0], fit_result_tracker[1])
                plt.title("fit result tracker each {0} fit".format(log_print_per_try))
                plt.xlabel("total fitting try")
                plt.ylabel("min chi square ")
                plt.show()

    def exp_decay(self, q_val, a1, t1, x0, y0, a2, t2):
        global para_unit
        exponential_temp = []
        if self.aniso_double_exp or self.iso_second:
            if len(q_val) != 1:
                for q_idx in range(len(q_val)):
                    if (q_val[q_idx] - x0 * para_unit['x0']) > 0:
                        exponential_temp.append(a1 * para_unit['a1'] * np.exp(-((q_val[q_idx] - x0 * para_unit['x0']) / (t1 * para_unit['t1']))) + a2 * para_unit['a2'] * np.exp(-((q_val[q_idx] - x0 * para_unit['x0']) / (t2 * para_unit['t2']))) + y0)
                    else:
                        exponential_temp.append(float(0))
                        # exponential_temp.append(y0)
                return exponential_temp

            elif len(q_val) == 1:
                if (q_val[0] - x0 * para_unit['x0']) > 0:
                    calc_val = (a1 * para_unit['a1'] * np.exp(-((q_val[0] - x0 * para_unit['x0']) / (t1 * para_unit['t1']))) + a2 * para_unit['a2'] * np.exp(-((q_val[0] - x0 * para_unit['x0']) / (t2 * para_unit['t2']))) + y0)
                else:
                    calc_val = float(0)
                return calc_val
        elif self.aniso_single_exp or self.temporary_mask or self.aniso_single_exp_without_conv:
            if len(q_val) != 1:
                for q_idx in range(len(q_val)):
                    if (q_val[q_idx] - x0 * para_unit['x0']) > 0:
                        exponential_temp.append(a1 * para_unit['a1'] * np.exp(-((q_val[q_idx] - x0 * para_unit['x0']) / (t1 * para_unit['t1']))) + y0)
                    else:
                        exponential_temp.append(float(0))

                return exponential_temp

            elif len(q_val) == 1:
                if (q_val[0] - x0 * para_unit['x0']) > 0:
                    calc_val = (a1 * para_unit['a1'] * np.exp(-((q_val[0] - x0 * para_unit['x0']) / (t1 * para_unit['t1']))) + + y0)
                else:
                    calc_val = float(0)
                return calc_val
        elif self.aniso_cos_square_fit:
            a2 = a1 * a2
            t2 = t1 + t2
            for q_idx in range(len(q_val)):
                if (q_val[q_idx] - x0 * para_unit['x0']) > 0:
                    exponential_temp.append(a1 * para_unit['a1'] * np.exp(-((q_val[q_idx] - x0 * para_unit['x0']) / (t1 * para_unit['t1']))) + a2 * para_unit['a2'] * np.exp(-((q_val[q_idx] - x0 * para_unit['x0']) / (t2 * para_unit['t2']))) + y0)
                else:
                    exponential_temp.append(float(0))
                    # exponential_temp.append(y0)
            return exponential_temp

    def stretched_exp(self, time_list, a, t, beta, x0, y0):
        stretched_exp_temp = []
        if len(time_list) != 1:
            for q_idx in range(len(time_list)):
                if (time_list[q_idx] - x0 * para_unit['x0']) > 0:
                    stretched_exp_temp.append(a * para_unit['a1'] * np.exp(-(((time_list[q_idx] - x0 * para_unit['x0']) / (t * para_unit['t1'])) ** beta)) + y0)
                else:
                    stretched_exp_temp.append(float(0))
            return stretched_exp_temp

        elif len(time_list) == 1:
            if (time_list[0] - x0 * para_unit['x0']) > 0:
                calc_val = (a * para_unit['a1'] * np.exp(-(((time_list[0] - x0 * para_unit['x0']) / (t * para_unit['t1'])) ** beta)) + y0)
            else:
                calc_val = float(0)
            return calc_val

    def gaussian(self, q_val, FWHM):
        gaussian_temp = (1 / (FWHM * para_unit['FWHM'] * (np.sqrt(2 * np.pi))) * np.exp(-pow(q_val / (FWHM * para_unit['FWHM']), 2.0) / 2.0))
        return gaussian_temp

    def new_convolution_double_exp(self, time_list, a1, t1, x0, y0, ratio_a, delta_t, FWHM):
        # self.calculate_time_list = np.round(np.arange(-1, 3.001, 10E-3), 3)
        t2 = t1+delta_t
        # t2 = delta_t
        a2 = ratio_a * a1
        # a2 = ratio_a
        time_list = self.calculate_time_list
        gaussian_temp = self.gaussian(time_list, FWHM)
        # if self.num_rsv_to_fit == 2:
        #     gaussian_temp = -gaussian_temp
        temp_time = self.time_step * (len(self.calculate_time_list) - 1)
        temp_time_arr = np.linspace(-temp_time, temp_time, len(self.calculate_time_list) * 2 - 1)
        exponential_temp = self.exp_decay(temp_time_arr, a1, t1, x0, y0, a2, t2)
        temp_convoluted_function = []
        convoluted_function = []
        temp_convoluted_function = np.convolve(exponential_temp, gaussian_temp, mode='valid') * self.time_step
        if len(self.exp_calc_time_delay) != 0:
            exponential_temp = self.exp_decay(self.exp_calc_time_delay, a1, t1, x0, y0, a2, t2)
            temp_convoluted_function = np.concatenate((temp_convoluted_function[:-1], exponential_temp), axis=0)
            time_list = np.concatenate((time_list[:-1], self.exp_calc_time_delay), axis=0)
        else:
            pass
        gaussian_temp = []
        # temp_arr = np.concatenate(temp_convoluted_function, time_list, axis=0)
        for idx in range(len(time_list)):
        # for idx in range(len(time_list)+len(self.exp_calc_time_delay)-1):
            if time_list[idx] in self.input_time_delay:
                convoluted_function.append(temp_convoluted_function[idx])
            else:
                continue
        return convoluted_function

    def new_convolution_double_exp_to_final_fit(self, time_list, a1, t1, x0, y0, ratio_a, delta_t, FWHM):
        # self.calculate_time_list = np.round(np.arange(-1, 3.001, 10E-3), 3)
        t2 = t1+delta_t
        # t2 = delta_t
        a2 = ratio_a * a1
        # a2 = ratio_a
        time_list = self.calculate_time_list
        gaussian_temp = self.gaussian(time_list, FWHM)
        # if self.num_rsv_to_fit == 2:
        #     gaussian_temp = -gaussian_temp
        temp_time = self.time_step * (len(self.calculate_time_list) - 1)
        temp_time_arr = np.linspace(-temp_time, temp_time, len(self.calculate_time_list) * 2 - 1)
        exponential_temp = self.exp_decay(temp_time_arr, a1, t1, x0, y0, a2, t2)
        temp_convoluted_function = []
        convoluted_function = []
        temp_convoluted_function = np.convolve(exponential_temp, gaussian_temp, mode='valid') * self.time_step
        if len(self.exp_calc_time_delay) != 0:
            exponential_temp = self.exp_decay(self.exp_calc_time_delay, a1, t1, x0, y0, a2, t2)
            temp_convoluted_function = np.concatenate((temp_convoluted_function[:-1], exponential_temp), axis=0)
            self.calculate_time_list = np.concatenate((time_list[:-1], self.exp_calc_time_delay), axis=0)
        else:
            pass
        for idx in range(len(time_list)):
        # for idx in range(len(time_list)+len(self.exp_calc_time_delay)-1):
            if time_list[idx] in self.input_time_delay:
                convoluted_function.append(temp_convoluted_function[idx])
            else:
                continue
        return temp_convoluted_function

    def single_exponential_function_convolution(self, time_list, a1, t1, x0, y0, FWHM):
        if self.temporary_mask:
            time_list = self.original_calc_time_list
        else:
            time_list = self.calculate_time_list
        gaussian_temp = self.gaussian(time_list, FWHM)
        a2 = 0
        t2 = 0
        temp_convoluted_function = []
        convoluted_function = []

        if self.temporary_mask:
            temp_time = self.time_step * (len(self.original_calc_time_list) - 1)
            temp_time_arr = np.linspace(-temp_time, temp_time, len(self.original_calc_time_list) * 2 - 1)
        else:
            temp_time = self.time_step * (len(self.calculate_time_list) - 1)
            temp_time_arr = np.linspace(-temp_time, temp_time, len(self.calculate_time_list) * 2 - 1)
        exponential_temp = self.exp_decay(temp_time_arr, a1, t1, x0, y0, a2, t2)
        temp_convoluted_function = []
        temp_convoluted_function = np.convolve(exponential_temp, gaussian_temp, mode='valid') * self.time_step
        gaussian_temp = []

        temp = []
        for time_idx in range(len(self.original_calc_time_list)):
            if self.original_calc_time_list[time_idx] in self.calculate_time_list:
                temp.append(temp_convoluted_function[time_idx])

        if len(self.exp_calc_time_delay) != 0:
            exponential_temp = self.exp_decay(self.exp_calc_time_delay, a1, t1, x0, y0, a2, t2)
            if self.temporary_mask:
                temp_convoluted_function = np.concatenate((temp[:-1], exponential_temp), axis=0)
                time_list = np.concatenate((self.calculate_time_list[:-1], self.exp_calc_time_delay), axis=0)
            else:
                temp_convoluted_function = np.concatenate((temp_convoluted_function[:-1], exponential_temp), axis=0)
                time_list = np.concatenate((time_list[:-1], self.exp_calc_time_delay), axis=0)
        else:
            if self.temporary_mask:
                time_list = self.calculate_time_list
                temp_convoluted_function = temp
        for idx in range(len(time_list)):
            if time_list[idx] in self.input_time_delay:
                convoluted_function.append(temp_convoluted_function[idx])
            else:
                continue

        return convoluted_function

    def single_exponential_function_convolution_to_final_fit(self, time_list, a1, t1, x0, y0, FWHM):
        if self.temporary_mask:
            time_list = self.original_calc_time_list
        else:
            time_list = self.calculate_time_list
        gaussian_temp = self.gaussian(time_list, FWHM)
        temp_convoluted_function = []
        a2 = 0
        t2 = 0
        if self.temporary_mask:
            temp_time = self.time_step * (len(self.original_calc_time_list) - 1)
            temp_time_arr = np.linspace(-temp_time, temp_time, len(self.original_calc_time_list) * 2 - 1)
        else:
            temp_time = self.time_step * (len(self.calculate_time_list) - 1)
            temp_time_arr = np.linspace(-temp_time, temp_time, len(self.calculate_time_list) * 2 - 1)
        exponential_temp = self.exp_decay(temp_time_arr, a1, t1, x0, y0, a2, t2)
        temp_convoluted_function = []
        convoluted_function = []
        temp_convoluted_function = np.convolve(exponential_temp, gaussian_temp, mode='valid') * self.time_step
        gaussian_temp = []

        # temp = []
        # for time_idx in range(len(self.original_calc_time_list)):
        #     if self.original_calc_time_list[time_idx] in self.calculate_time_list:
        #         temp.append(temp_convoluted_function[time_idx])

        if len(self.exp_calc_time_delay) != 0:
            exponential_temp = self.exp_decay(self.exp_calc_time_delay, a1, t1, x0, y0, a2, t2)
            if self.temporary_mask:
                temp_convoluted_function = np.concatenate((temp_convoluted_function[:-1], exponential_temp), axis=0)
                time_list = np.concatenate((self.original_calc_time_list[:-1], self.exp_calc_time_delay), axis=0)
                self.original_calc_time_list = time_list
            else:
                temp_convoluted_function = np.concatenate((temp_convoluted_function[:-1], exponential_temp), axis=0)
                time_list = np.concatenate((time_list[:-1], self.exp_calc_time_delay), axis=0)
        else:
            pass

        return temp_convoluted_function

    def stretched_exponential_function_convolution(self, time_list, a, t, beta, x0, y0, FWHM):
        time_list = self.calculate_time_list
        gaussian_temp = self.gaussian(time_list, FWHM)
        temp_convoluted_function = []
        convoluted_function = []
        temp_time = self.time_step * (len(self.calculate_time_list) - 1)
        temp_time_arr = np.linspace(-temp_time, temp_time, len(self.calculate_time_list) * 2 - 1)
        stretched_exponential_temp = self.stretched_exp(temp_time_arr, a, t, beta, x0, y0)
        temp_convoluted_function = np.convolve(stretched_exponential_temp, gaussian_temp, mode='valid') * self.time_step
        # for time_point_idx in range(len(time_list)):
        #     if time_point_idx == 0:
        #         temp_calc_time = []
        #         for calc_idx in range(len(self.calculate_time_list)):
        #             temp_calc_time.append(time_list[time_point_idx] - time_list[calc_idx])
        #         stretched_exponential_temp = self.stretched_exp(temp_calc_time, a, t, beta, x0, y0)
        #     else:
        #         temp_time_list = []
        #         temp_time_list.append(time_list[time_point_idx] - time_list[0])
        #         stretched_exponential_temp = np.roll(stretched_exponential_temp, 1)
        #         stretched_exponential_temp[0] = self.stretched_exp(temp_time_list, a, t, beta, x0, y0)
        #     temp_calc_val = np.multiply(stretched_exponential_temp, gaussian_temp) * self.time_step
        #     temp_convolution_val = np.sum(temp_calc_val)
        #     temp_convoluted_function.append([time_list[time_point_idx], temp_convolution_val])
        gaussian_temp = []
        if len(self.exp_calc_time_delay) != 0:
            stretched_exponential_temp = self.stretched_exp(self.exp_calc_time_delay, a, t, beta, x0, y0)
            temp_convoluted_function = np.concatenate((temp_convoluted_function[:-1], stretched_exponential_temp), axis=0)
            time_list = np.concatenate((time_list[:-1], self.exp_calc_time_delay), axis=0)
        else:
            pass

        for idx in range(len(time_list)):
        # for idx in range(len(time_list)+len(self.exp_calc_time_delay)-1):
            if time_list[idx] in self.input_time_delay:
                convoluted_function.append(temp_convoluted_function[idx])
            else:
                continue
        if self.finish_minuit:
            self.calculate_time_list = time_list
            return temp_convoluted_function

        else:
            return convoluted_function

    def stretched_exponential_function_convolution_to_final_fit(self, time_list, a, t, beta, x0, y0, FWHM):
        time_list = self.calculate_time_list
        gaussian_temp = self.gaussian(time_list, FWHM)
        temp_convoluted_function = []
        convoluted_function = []
        for time_point_idx in range(len(time_list)):
            if time_point_idx == 0:
                temp_calc_time = []
                for calc_idx in range(len(self.calculate_time_list)):
                    temp_calc_time.append(time_list[time_point_idx] - time_list[calc_idx])
                stretched_exponential_temp = self.stretched_exp(temp_calc_time, a, t, beta, x0, y0)
            else:
                temp_time_list = []
                temp_time_list.append(time_list[time_point_idx] - time_list[0])
                stretched_exponential_temp = np.roll(stretched_exponential_temp, 1)
                stretched_exponential_temp[0] = self.stretched_exp(temp_time_list, a, t, beta, x0, y0)
            temp_calc_val = np.multiply(stretched_exponential_temp, gaussian_temp) * self.time_step
            temp_convolution_val = np.sum(temp_calc_val)
            temp_convoluted_function.append(temp_convolution_val)
        gaussian_temp = []

        return temp_convoluted_function

    def p_gallo_function_convolution(self, time_list, a, t1, x0, y0, t2, beta, FWHM):
        time_list = self.calculate_time_list
        gaussian_temp = self.gaussian(time_list, FWHM)
        # if self.num_rsv_to_fit == 2:
        #     gaussian_temp = -gaussian_temp
        temp_time = self.time_step * (len(self.calculate_time_list) - 1)
        temp_time_arr = np.linspace(-temp_time, temp_time, len(self.calculate_time_list) * 2 - 1)
        p_function_val = self.p_gallo_function(temp_time_arr, a, t1, x0, y0, t2, beta)
        temp_convoluted_function = []
        convoluted_function = []
        temp_convoluted_function = np.convolve(p_function_val, gaussian_temp, mode='valid') * self.time_step
        if len(self.exp_calc_time_delay) != 0:
            p_function_val = self.p_gallo_function(self.exp_calc_time_delay, a, t1, x0, y0, t2, beta)
            temp_convoluted_function = np.concatenate((temp_convoluted_function[:-1], p_function_val), axis=0)
            time_list = np.concatenate((time_list[:-1], self.exp_calc_time_delay), axis=0)
        else:
            pass
        gaussian_temp = []
        # temp_arr = np.concatenate(temp_convoluted_function, time_list, axis=0)
        for idx in range(len(time_list)):
            # for idx in range(len(time_list)+len(self.exp_calc_time_delay)-1):
            if time_list[idx] in self.input_time_delay:
                convoluted_function.append(temp_convoluted_function[idx])
            else:
                continue
        if self.finish_minuit:
            self.calculate_time_list = time_list
            return temp_convoluted_function
        else:
            return convoluted_function

    def p_gallo_function(self, q_val, a, t1, x0, y0, t2, beta):
        global para_unit
        function_val = []
        if len(q_val) != 1:
            for q_idx in range(len(q_val)):
                if (q_val[q_idx] - x0 * para_unit['x0']) > 0:
                    function_val.append((1-a) * para_unit['a1'] * np.exp(-np.power(((q_val[q_idx] - x0 * para_unit['x0']) / (t1 * para_unit['t1'])),2)) + a * para_unit['a1'] * np.exp(-(((q_val[q_idx] - x0 * para_unit['x0']) / (t2 * para_unit['t2'])) ** beta)) + y0)
                else:
                    function_val.append(float(0))
                    # exponential_temp.append(y0)
            return function_val

        elif len(q_val) == 1:
            if (q_val[0] - x0 * para_unit['x0']) > 0:
                calc_val = ((1-a) * para_unit['a1'] * np.exp(-np.power(((q_val[0] - x0 * para_unit['x0']) / (t1 * para_unit['t1'])),2)) + a * para_unit['a1'] * np.exp(-(((q_val[0] - x0 * para_unit['x0']) / (t2 * para_unit['t2'])) ** beta)) + y0)
            else:
                calc_val = float(0)
            return calc_val

    def fixed_t0_single_exp(self, q_val, a1, t1, x0, y0):
        a2 = 0
        t2 = 0
        if len(self.exp_calc_time_delay):
            time_list = np.concatenate((self.calculate_time_list[:-1], self.exp_calc_time_delay), axis=0)
        else:
            time_list = self.calculate_time_list
        if not self.finish_minuit:
            temp_collected_data = []
            exponential_temp = self.exp_decay(time_list, a1, t1, x0, y0, a2, t2)
            for idx in range(len(time_list)):
                if time_list[idx] in self.input_time_delay:
                    temp_collected_data.append(exponential_temp[idx])
            return temp_collected_data
        elif self.finish_minuit:
            time_list = np.concatenate((self.original_calc_time_list[:-1], self.exp_calc_time_delay),axis=0)
            exponential_temp = self.exp_decay(time_list, a1, t1, x0, y0, a2, t2)
            self.calculate_time_list = time_list
            return exponential_temp

    def exp_heating(self, q_val, a2, t2, x0):
        exponential_temp = []
        for q_idx in range(len(q_val)):
            if (q_val[q_idx] - x0 * para_unit['x0']) > 0:
                exponential_temp.append(a2 * para_unit['a2'] * np.exp(-((q_val[q_idx] - x0 * para_unit['x0']) / (t2 * para_unit['t2']))) - a2)
                # if temp_val >= 0:
                #     exponential_temp.append(temp_val)
                # else :
                #     exponential_temp.append(0.0)
            else:
                exponential_temp.append(0.0)
        return exponential_temp

    def heating_convolution(self, time_list, a, t, x0, FWHM):
        time_list = self.calculate_time_list
        gaussian_temp = self.gaussian(time_list, FWHM)
        gaussian_temp = -gaussian_temp
        temp_time = self.time_step * (len(self.calculate_time_list) - 1)
        temp_time_arr = np.linspace(-temp_time, temp_time, len(self.calculate_time_list) * 2 - 1)
        exponential_temp = self.exp_heating(temp_time_arr, a, t, x0)
        temp_convoluted_function = []
        convoluted_function = []
        temp_convoluted_function = np.convolve(exponential_temp, gaussian_temp, mode='valid') * self.time_step
        if len(self.exp_calc_time_delay) != 0:
            exponential_temp = self.exp_heating(self.exp_calc_time_delay, a, t, x0)
            temp_convoluted_function = np.concatenate((temp_convoluted_function[:-1], exponential_temp), axis=0)
            time_list = np.concatenate((time_list[:-1], self.exp_calc_time_delay), axis=0)
        else:
            pass
        gaussian_temp = []
        for idx in range(len(time_list)):
            if time_list[idx] in self.input_time_delay:
                convoluted_function.append(temp_convoluted_function[idx])
            else:
                continue
        if self.finish_minuit:
            self.calculate_time_list = time_list
            return temp_convoluted_function
        else:
            return convoluted_function

    def linear_combination(self, q_val, c1, c2):#, c3):
        after_LC = c1 * self.sample_LSV[0] + c2 * self.sample_LSV[1] # + c3 * self.sample_LSV[2]

        return after_LC

    def run_minuit_with_LC(self, plot_each_delay = False):
        data_yerr = 1

        c1_init = 0
        c2_init = 0
        # c3_init = 0
        fit_diff = []

        plt.plot(self.input_raw_q_val, self.sample_LSV[0], label="Materials 1")
        plt.plot(self.input_raw_q_val, self.sample_LSV[1], label="Materials 2")
        plt.title("Materials for Linear combination")
        plt.legend()
        plt.show()

        for delay_idx in range(len(self.input_raw_data)):
            LC_least_square = LeastSquares(self.input_raw_q_val, self.input_raw_data[delay_idx], data_yerr, self.linear_combination)
            m = Minuit(LC_least_square, c1=c1_init, c2=c2_init)#, c3=c3_init)
            # m = Minuit(LC_least_square, *now_initials)
            m.errors = (1E-5, 1E-5)#, 1E-5)
            m.limits = [(None, None), (None, None)]#, (None, None)]

            m.simplex()
            m.migrad()
            m.hesse()

            if m.valid:
                # print(m.params)
                # print(m.values)
                self.best_params_save.append(m.values)
                self.plot_for_save.append(m.fval)
                if plot_each_delay:
                    plt.plot(self.input_raw_q_val, self.input_raw_data[delay_idx], label="Exp {0} ps ({1}-th delay)".format(self.input_time_delay[delay_idx], delay_idx + 1))
                    plt.plot(self.input_raw_q_val, self.linear_combination(self.input_raw_q_val, *m.values), label="Fitted data")# + "\n" + "chi2 : " + str(np.round(m.fval, 10)))
                    temp_fit_result = self.linear_combination(self.input_raw_q_val, *m.values)
                    fit_diff.append(self.input_raw_data[delay_idx]-temp_fit_result)
                    # chi2 = plt.plot(label="chi2 : " + str(np.round(m.fval, 10)))
                    plt.legend(title="chi2 : " + str(np.round(m.fval, 10)) )
                    # plt.ylim(-1.2, 1.2)
                    plt.show()

        if len(self.plot_for_save) != len(self.input_raw_data):
            print("There are delays that are not valid with linear combination")
            return

        else:
            if self.iso_LC:
                if self.weighted_average:
                    if len(self.runList) == 2:
                        # dat_file = open("../results/anisotropy/svd/run{0:04d}_{1:04d}_avg/run{0}_{1}_avg_{2}/run{0:04d}_{1:04d}_contributions_from_aniso_raw_pattern.dat".format(self.runList[0], self.runList[1], self.dat_file_time_stamp), 'w')
                        if [45, 46] == self.material_run_list:
                            dat_file = open("../results/anisotropy/svd/run{0:04d}_{1:04d}_avg/run{0}_{1}_avg_{2}/run{0:04d}_{1:04d}_iso_contributions_from_LSV_1_2_of_droplet.dat".format(self.runList[0], self.runList[1], self.dat_file_time_stamp), 'w')
                        else:
                            dat_file = open("../results/anisotropy/svd/run{0:04d}_{1:04d}_avg/run{0}_{1}_avg_{2}/run{0:04d}_{1:04d}_iso_contributions_from_LSV_1_2.dat".format(self.runList[0], self.runList[1], self.dat_file_time_stamp), 'w')
                    elif len(self.runList) == 3:
                        dat_file = open("../results/anisotropy/svd/run{0:04d}_{1:04d}_{2:04d}_avg/run{0}_{1}_{2}_avg_{3}/run{0:04d}_{1:04d}_{2:04d}_iso_contributions_from_aniso_raw_pattern.dat".format(self.runList[0], self.runList[1],self.runList[2], self.dat_file_time_stamp), 'w')
                else:
                    # dat_file = open("../results/anisotropy/svd/run{0:04d}/run{0}_{1}/run{0}_contributions_from_aniso_raw_pattern.dat".format(self.runNum, self.dat_file_time_stamp), 'w')
                    if [45, 46] == self.material_run_list:
                        dat_file = open("../results/anisotropy/svd/run{0:04d}/run{0}_{1}/run{0}_iso_contributions_from_LSV_1_2_of_droplet.dat".format(self.runNum, self.dat_file_time_stamp), 'w')
                    else:
                        dat_file = open("../results/anisotropy/svd/run{0:04d}/run{0}_{1}/run{0}_iso_contributions_from_LSV_1_2.dat".format(self.runNum, self.dat_file_time_stamp), 'w')
            else:
                if self.weighted_average:
                    if len(self.runList) == 2:
                        # dat_file = open("../results/anisotropy/svd/run{0:04d}_{1:04d}_avg/run{0}_{1}_avg_{2}/run{0:04d}_{1:04d}_contributions_from_aniso_raw_pattern.dat".format(self.runList[0], self.runList[1], self.dat_file_time_stamp), 'w')
                        if [45, 46] == self.material_run_list:
                            dat_file = open("../results/anisotropy/svd/run{0:04d}_{1:04d}_avg/run{0}_{1}_avg_{2}/run{0:04d}_{1:04d}_contributions_from_LSV_1_2_of_droplet.dat".format(self.runList[0], self.runList[1], self.dat_file_time_stamp), 'w')
                        else:
                            dat_file = open("../results/anisotropy/svd/run{0:04d}_{1:04d}_avg/run{0}_{1}_avg_{2}/run{0:04d}_{1:04d}_contributions_from_LSV_1_2.dat".format(self.runList[0], self.runList[1], self.dat_file_time_stamp), 'w')
                    elif len(self.runList) == 3:
                        dat_file = open("../results/anisotropy/svd/run{0:04d}_{1:04d}_{2:04d}_avg/run{0}_{1}_{2}_avg_{3}/run{0:04d}_{1:04d}_{2:04d}_contributions_from_aniso_raw_pattern.dat".format(self.runList[0], self.runList[1],self.runList[2], self.dat_file_time_stamp), 'w')
                else:
                    # dat_file = open("../results/anisotropy/svd/run{0:04d}/run{0}_{1}/run{0}_contributions_from_aniso_raw_pattern.dat".format(self.runNum, self.dat_file_time_stamp), 'w')
                    if [45, 46] == self.material_run_list:
                        dat_file = open("../results/anisotropy/svd/run{0:04d}/run{0}_{1}/run{0}_contributions_from_LSV_1_2_of_droplet.dat".format(self.runNum, self.dat_file_time_stamp), 'w')
                    else:
                        dat_file = open("../results/anisotropy/svd/run{0:04d}/run{0}_{1}/run{0}_contributions_from_LSV_1_2.dat".format(self.runNum, self.dat_file_time_stamp), 'w')
            # first_row_data = ['Time', 'contribution from LSV1', 'contribution from LSV2', 'contribution from LSV3', 'chi2']
            first_row_data = ['Time', 'contribution from LSV1', 'contribution from LSV2', 'chi2']

            # first_row_data = ['Time', 'contribution from major', 'contribution from averaged pattern', 'chi2']
            self.best_params_save = np.array(self.best_params_save)
            first_singVal_like = np.linalg.norm(self.best_params_save[:, 0])
            second_singVal_like = np.linalg.norm(self.best_params_save[:,1])
            # contribution_max = np.max(self.best_params_save[:, 0])
            # self.best_params_save[:, 0] = (self.best_params_save[:, 0] / contribution_max) * self.aniso_rsv_max_val
            self.best_params_save[:, 0] = (self.best_params_save[:, 0] / first_singVal_like)
            self.best_params_save[:, 1] = (self.best_params_save[:, 1] / second_singVal_like)
            if self.iso_LC:
                if self.best_params_save[:, 0][-1] < 0:
                    self.best_params_save[:,0] = -self.best_params_save[:, 0]
                if self.best_params_save[:, 1][11] < 0:
                    self.best_params_save[:,1] = -self.best_params_save[:, 1]
            for idx in range(len(first_row_data)):
                dat_file.write(first_row_data[idx] + "\t")
            for delay_idx in range(len(self.input_time_delay)):
                if delay_idx == 0:
                    dat_file.write("\t" + "\t" + str("q val") + "\t")
                dat_file.write(str(delay_idx + 1) + "-th delay" + "\t")
            dat_file.write("\n")

            # for idx in range(len(self.best_params_save)+len(self.input_time_delay)+1):
            for idx in range(len(self.input_raw_q_val)):
                if idx < 1:
                    dat_file.write(str(self.input_time_delay[idx]) + "\t")
                    dat_file.write(str(self.best_params_save[idx][0]) + "\t")
                    dat_file.write(str(self.best_params_save[idx][1]) + "\t")
                    # dat_file.write(str(self.best_params_save[idx][2]) + "\t")
                    dat_file.write(str(self.plot_for_save[idx]) + "\t")
                    dat_file.write("singVal1" + "\t" + str(first_singVal_like) + "\t")
                    dat_file.write(str(self.input_raw_q_val[idx]) + "\t")
                    for delay_idx in range(len(self.input_time_delay)):
                        dat_file.write(str(fit_diff[delay_idx][idx]) + "\t")
                    dat_file.write("\n")
                elif idx < 2:
                    dat_file.write(str(self.input_time_delay[idx]) + "\t")
                    dat_file.write(str(self.best_params_save[idx][0]) + "\t")
                    dat_file.write(str(self.best_params_save[idx][1]) + "\t")
                    # dat_file.write(str(self.best_params_save[idx][2]) + "\t")
                    dat_file.write(str(self.plot_for_save[idx]) + "\t")
                    dat_file.write("singVal2" + "\t" + str(second_singVal_like) + "\t")
                    dat_file.write(str(self.input_raw_q_val[idx]) + "\t")
                    for delay_idx in range(len(self.input_time_delay)):
                        dat_file.write(str(fit_diff[delay_idx][idx]) + "\t")
                    dat_file.write("\n")
                elif idx < len(self.input_time_delay):
                    dat_file.write(str(self.input_time_delay[idx]) + "\t")
                    dat_file.write(str(self.best_params_save[idx][0]) + "\t")
                    dat_file.write(str(self.best_params_save[idx][1]) + "\t")
                    dat_file.write(str(self.plot_for_save[idx]) + "\t"+ "\t"+ "\t")
                    dat_file.write(str(self.input_raw_q_val[idx]) + "\t")
                    for delay_idx in range(len(self.input_time_delay)):
                        dat_file.write(str(fit_diff[delay_idx][idx]) + "\t")
                    dat_file.write("\n")
                else:
                    dat_file.write("\t"+ "\t"+ "\t"+ "\t"+ "\t"+ "\t")
                    dat_file.write(str(self.input_raw_q_val[idx]) + "\t")
                    for delay_idx in range(len(self.input_time_delay)):
                        dat_file.write(str(fit_diff[delay_idx][idx]) + "\t")
                    dat_file.write("\n")
            dat_file.close()

    def save_fit_result_as_dat(self, data, calculated_data, exp_data):
        try:
            dat_file = open(self.file_out_path + self.file_common_name + self.now_time +  '.dat', 'w')
        except:
            os.makedirs(self.file_out_path)
            dat_file = open(self.file_out_path + self.file_common_name + self.now_time +  '.dat', 'w')
        if self.temporary_mask:
            self.calculate_time_list = self.original_calc_time_list
        if self.anisotropy_process:
            if self.aniso_double_exp:
                # if self.num_rsv_to_fit == 2:
                #     parameter_name_list = ['chi2', 'a', 't', 'x0', 'FWHM']
                # else:
                    parameter_name_list = ['chi2', 'a1', 't1', 'x0', 'y0', 'a2', 't2', 'FWHM']
                    data[5] = data[1] + data[5]
                    data[4] = data[0] * data[4]
            elif self.aniso_single_exp:
                parameter_name_list = ['chi2', 'a1', 't1', 'x0', 'y0', 'FWHM']
            elif self.aniso_stretched_exp:
                parameter_name_list = ['chi2', 'a', 't', 'beta', 'x0', 'y0', 'FWHM']
            elif self.aniso_p_gallo:
                parameter_name_list = ['chi2', 'a', 't1', 'x0', 'y0', 't2', 'beta', 'FWHM']
            elif self.aniso_cos_square_fit:
                parameter_name_list = ['chi2', 'a1', 't1', 'x0', 'y0', 'a2', 't2']
                data[5] = data[1] + data[5]
                data[4] = data[0] * data[4]
            elif self.temporary_mask:
                parameter_name_list = ['chi2', 'a1', 't1', 'x0', 'y0', 'FWHM']
            elif self.aniso_single_exp_without_conv:
                parameter_name_list = ['chi2', 'a1', 't1', 'x0', 'y0']
        elif self.isotropy_process:
            if self.iso_heating:
                parameter_name_list = ['chi2', 'a2', 't2', 'x0']
            elif self.iso_second:
                parameter_name_list = ['chi2', 'a1', 't1', 'x0', 'y0', 'a2', 't2', 'FWHM']
                data[5] = data[1] + data[5]
                data[4] = data[0] * data[4]
        # if random_initialization:
        if self.anisotropy_process or self.iso_second:
            first_row_data = ['parameters', 'fitted values', 'parameter errors', 'time', 'exp data', 'calculate time','calculated value']
        elif self.iso_heating:
            first_row_data = ['parameters', 'fitted values', 'time', 'parameter errors', 'exp data', 'calculated value for ratio', 'calculate time', 'calculated value']
            calc_val_list_for_ratio = []
            for idx in range(len(self.calculate_time_list)):
                if self.calculate_time_list[idx] in self.input_time_delay:
                    calc_val_list_for_ratio.append(calculated_data[idx])
                else:
                    continue
        if self.iso_heating:
            for idx in range(len(first_row_data)):
                if (idx + 1) == len(first_row_data):
                    dat_file.write(first_row_data[idx] + "\n")
                else:
                    dat_file.write(first_row_data[idx] + "\t")

            for idx in range(len(calculated_data)):
                if idx < len(data) and idx == 0:
                    dat_file.write(str(parameter_name_list[idx]) + "\t")
                    dat_file.write(str(self.best_chi_square_save) + "\t" + "\t")
                    dat_file.write(str(self.input_time_delay[idx]) + "\t")
                    dat_file.write(str(exp_data[idx]) + "\t")
                    dat_file.write(str(calc_val_list_for_ratio[idx]) + "\t")
                    dat_file.write(str(self.calculate_time_list[idx]) + "\t")
                    dat_file.write(str(calculated_data[idx]) + "\n")
                    continue

                if idx < (len(data) + 1):
                    dat_file.write(str(parameter_name_list[idx]) + "\t")
                    dat_file.write(str(data[idx - 1]) + "\t")
                    dat_file.write(str(self.best_params_erros[idx - 1]) + "\t")
                    dat_file.write(str(self.input_time_delay[idx]) + "\t")
                    dat_file.write(str(exp_data[idx]) + "\t")
                    dat_file.write(str(calc_val_list_for_ratio[idx]) + "\t")
                    dat_file.write(str(self.calculate_time_list[idx]) + "\t")
                    dat_file.write(str(calculated_data[idx]) + "\n")
                else:
                    if idx < len(exp_data):
                        dat_file.write("\t" + "\t" + "\t" + str(self.input_time_delay[idx]) + "\t")
                        dat_file.write(str(exp_data[idx]) + "\t")
                        dat_file.write(str(calc_val_list_for_ratio[idx]) + "\t")
                        dat_file.write(str(self.calculate_time_list[idx]) + "\t")
                        dat_file.write(str(calculated_data[idx]) + "\n")
                    else:
                        dat_file.write("\t" + "\t" + "\t" + "\t" + "\t" + "\t" + str(self.calculate_time_list[idx]) + "\t")
                        dat_file.write(str(calculated_data[idx]) + "\n")
        elif self.anisotropy_process or self.iso_second:
            for idx in range(len(first_row_data)):
                if (idx + 1) == len(first_row_data):
                    dat_file.write(first_row_data[idx] + "\n")
                else:
                    dat_file.write(first_row_data[idx] + "\t")

            for idx in range(len(calculated_data)):
                if idx < len(data) and idx == 0:
                    dat_file.write(str(parameter_name_list[idx]) + "\t")
                    dat_file.write(str(self.best_chi_square_save) + "\t" + "\t")
                    dat_file.write(str(self.input_time_delay[idx]) + "\t")
                    dat_file.write(str(exp_data[idx]) + "\t")
                    dat_file.write(str(self.calculate_time_list[idx]) + "\t")
                    dat_file.write(str(calculated_data[idx]) + "\n")
                    continue

                if idx < (len(data) + 1):
                    dat_file.write(str(parameter_name_list[idx]) + "\t")
                    dat_file.write(str(data[idx - 1]) + "\t")
                    dat_file.write(str(self.best_params_erros[idx - 1]) + "\t")
                    dat_file.write(str(self.input_time_delay[idx]) + "\t")
                    dat_file.write(str(exp_data[idx]) + "\t")
                    dat_file.write(str(self.calculate_time_list[idx]) + "\t")
                    dat_file.write(str(calculated_data[idx]) + "\n")
                else:
                    if idx < len(exp_data):
                        dat_file.write("\t" + "\t" + "\t" + str(self.input_time_delay[idx]) + "\t")
                        dat_file.write(str(exp_data[idx]) + "\t")
                        dat_file.write(str(self.calculate_time_list[idx]) + "\t")
                        dat_file.write(str(calculated_data[idx]) + "\n")
                    else:
                        dat_file.write("\t" + "\t" + "\t" + "\t" + "\t" + str(self.calculate_time_list[idx]) + "\t")
                        dat_file.write(str(calculated_data[idx]) + "\n")

    def write_log_info(self, calc_idx, chi2_save_list):
        if self.weighted_average:
            if self.anisotropy_process:
                if len(self.runList) == 2:
                    if self.aniso_single_exp:
                        out_file_name = self.log_out_file_common_name + "single_exp_log.txt"
                    elif self.aniso_double_exp:
                        if self.num_rsv_to_fit == 2:
                            out_file_name = self.log_out_file_common_name + "unexpected_component_log.txt"
                        else:
                            out_file_name = self.log_out_file_common_name + "double_exp_log.txt"
                    elif self.aniso_stretched_exp:
                        out_file_name = self.log_out_file_common_name + "stretched_exp_log.txt"
                    elif self.aniso_p_gallo:
                        out_file_name = self.log_out_file_common_name + "p_gallo_function_log.txt"
                    elif self.temporary_mask:
                        out_file_name = self.log_out_file_common_name + "gaussian_cut_log.txt"
                    elif self.aniso_single_exp_without_conv:
                        out_file_name = self.log_out_file_common_name + "single_exp_without_convolution_log.txt"

                elif len(self.runList) == 3:
                    if self.aniso_single_exp:
                        out_file_name = self.log_out_file_common_name + "single_exp_log.txt"
                    elif self.aniso_double_exp:
                        out_file_name = self.log_out_file_common_name + "double_exp_log.txt"
                    elif self.aniso_stretched_exp:
                        out_file_name = self.log_out_file_common_name + "stretched_exp_log.txt"
                    elif self.aniso_p_gallo:
                        out_file_name = self.log_out_file_common_name + "p_gallo_function_log.txt"
                    elif self.temporary_mask:
                        out_file_name = self.log_out_file_common_name + "gaussian_cut_log.txt"
            elif self.isotropy_process:
                if self.iso_heating:
                    if len(self.runList) == 2:
                        out_file_name = "/data/exp_data/myeong0609/PAL-XFEL_20210514/analysis/minuit/run_{0}_{1}_heating_log.txt".format(self.runList[0], self.runList[1])

                    elif len(self.runList) == 3:
                        out_file_name = "/data/exp_data/myeong0609/PAL-XFEL_20210514/analysis/minuit/run_{0}_{1}_{2}_heating_log.txt".format(self.runList[0], self.runList[1], self.runList[2])
                elif self.iso_second:
                    if len(self.runList) == 2:
                        out_file_name = "/data/exp_data/myeong0609/PAL-XFEL_20210514/analysis/minuit/run_{0}_{1}_second_rsv_log.txt".format(self.runList[0], self.runList[1])

                    elif len(self.runList) == 3:
                        out_file_name = "/data/exp_data/myeong0609/PAL-XFEL_20210514/analysis/minuit/run_{0}_{1}_{2}_second_rsv_log.txt".format(self.runList[0], self.runList[1], self.runList[2])
        else:
            if self.anisotropy_process:
                if self.aniso_single_exp:
                    out_file_name = self.log_out_file_common_name + "single_exp_log.txt"
                elif self.aniso_double_exp:
                    if self.num_rsv_to_fit == 2:
                        out_file_name = self.log_out_file_common_name + "unexpected_component_log.txt"
                    else:
                        out_file_name = self.log_out_file_common_name + "double_exp_log.txt"
                elif self.aniso_stretched_exp:
                    out_file_name = self.log_out_file_common_name  + "stretched_exp_log.txt"
                elif self.aniso_p_gallo:
                    out_file_name = self.log_out_file_common_name + "p_gallo_function_log.txt"
                elif self.aniso_cos_square_fit:
                    out_file_name = self.log_out_file_common_name + "md_sim_result_log.txt"
                elif self.temporary_mask:
                    out_file_name = self.log_out_file_common_name + "gaussian_cut_log.txt"
                elif self.aniso_single_exp_without_conv:
                    out_file_name = self.log_out_file_common_name + "single_exp_without_convolution_log.txt"

            elif self.isotropy_process:
                if self.iso_second:
                    out_file_name = "/data/exp_data/myeong0609/PAL-XFEL_20210514/analysis/minuit/run_" + str(self.runNum) + "_second_rsv_log.txt"
                elif self.iso_heating:
                    out_file_name = "/data/exp_data/myeong0609/PAL-XFEL_20210514/analysis/minuit/run_" + str(self.runNum) + "_heating_log.txt"
        if self.num_log_out == 0:
            try:
                log = open(out_file_name, 'r')
                os.remove(out_file_name)
            except:
                pass
            log = open(out_file_name, 'a')
            if self.weighted_average:
                if len(self.runList) == 2:
                    run_num_info = "Status of anisotropic fitting analysis for run {0}&{1}".format(self.runList[0], self.runList[1]) + "\n"
                    timeStamp = "Time stamp of this run is " + self.now_time + "\n"
                elif len(self.runList) == 3:
                    run_num_info = "Status of anisotropic fitting analysis for run {0}&{1}&{2}".format(self.runList[0], self.runList[1], self.runList[2]) + "\n"
                    timeStamp = "Time stamp of this run is " + self.now_time + "\n"
            else:
                run_num_info = "Status of anisotropic analysis for run " + str(self.runNum) + "\n"
                timeStamp = "Time stamp of this run is " + self.now_time + "\n"
            log.write(run_num_info)
            log.write(timeStamp)
            if chi2_save_list:
                contents = "now try : " + str(calc_idx + 1) + ", minimum chi2 : " + str(np.min(chi2_save_list)) + "\n"
                log.write(contents)
            else:
                contents = "now try : " + str(calc_idx + 1) + ", minimum chi2 : none" + "\n"
                log.write(contents)
            # log = open(out_file_name, 'a')
            # if self.chi2_save_list:
            #     contents = "now try : " + str(calc_idx + 1) + ", minimum chi2 : " + str(np.min(chi2_save_list)) + "\n"
            #     log.write(contents)
            # else:
            #     contents = "now try : " + str(calc_idx + 1) + ", minimum chi2 : none" + "\n"
            #     log.write(contents)
            self.num_log_out += 1
        else:
            log = open(out_file_name, 'a')
            if chi2_save_list:
                contents = "now try : " + str(calc_idx + 1) + ", minimum chi2 : " + str(np.min(chi2_save_list)) + "\n"
                log.write(contents)
            else:
                contents = "now try : " + str(calc_idx + 1) + ", minimum chi2 : none" + "\n"
                log.write(contents)
            self.num_log_out += 1
