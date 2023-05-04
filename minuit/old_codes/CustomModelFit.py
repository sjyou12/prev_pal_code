import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares
import datetime
import os
import scipy.signal as signal

# density_ref = 0.0  # Vacuum // 0.933532  78K, Ih
# density_pump = 0.9970470  # 298 K liquid water 0.9838 // 243.15 K, supercooled liquid
# avogadro_const = 6.022e23
# atomic_number_sum = 10
# molecular_weight = 18.01528
num_of_calculation = 401 #TODO change if tim scale is approaching ns scale
time_step = 10E-3
start_time = -2 #-1E-12

random_initialization = True
num_of_random_initialization = 1000


class CustomModelFit:
    def __init__(self, input_file_common_root, given_file_name, num_of_RSV_to_fit, run_num, need_negative, heating, anisotropy, weighted_average, run_list, previous_result_time_stamp, aniso_single_exponential, aniso_double_exponential, aniso_stretched_exponential):

        self.x0_from_anisotropy = -0.013092766334061

        self.aniso_single_exp = aniso_single_exponential
        self.aniso_double_exp = aniso_double_exponential
        self.aniso_stretched_exp = aniso_stretched_exponential
        self.anisotropy_process = anisotropy
        self.heating_process = heating

        if self.anisotropy_process:
            if self.aniso_double_exp:
                #self.param_limit = [(None, None),(None, None),(None, None),(None, None),(None, None),(None, None),(None, None)]  # a1, t1, x0, y0, a2, t2, FWHM
                # self.param_limit = [(0, None), (1, None), (None, None), (None, None), (0, None), (1, 100), (0.75, 2)]  # a1, t1, x0, y0, a2, t2, FWHM
                self.param_limit = [(0, 5), (1, 10), (-1, 1), (-1, 1), (0, 5), (1, 100), (0.75, 2)]  # a1, t1, x0, y0, a2, t2, FWHM
            elif self.aniso_single_exp:
                self.param_limit = [(0, None), (1, None), (None, None), (None, None), (0, 10)]  # a1, t1, x0, y0, FWHM
            elif self.aniso_stretched_exp:
                self.param_limit = [(0, None), (1, None) ,(0.5, 1), (None, None), (None, None), (0, 10)]  # a, t, beta, x0, y0, FWHM
        elif self.heating_process:
            self.param_limit = [(None, None), (0, 100), (None, None)]  # a2, t2, x0

        self.input_time_delay = []
        self.common_time_delay = []
        self.input_raw_data = []
        self.input_raw_data_len = 0
        self.rsv_component_list = []
        self.target_RSV = num_of_RSV_to_fit
        self.calculate_time_list = []
        if self.anisotropy_process:
            self.num_of_calculation = 401 # TODO change if tim scale is approaching ns scale
        elif self.heating_process:
            self.num_of_calculation = 3101

        for idx in range(self.num_of_calculation):
            self.calculate_time_list.append(start_time + time_step * idx)
        self.calculate_time_list= np.round(self.calculate_time_list, 15)
        self.para_name = ['a1', 't1', 'x0', 'y0', 'a2', 't2', 'FWHM']
        # self.para_unit = {'FWHM':1E-13/2.35482, 'x0':1E-12, 't1':1E-13, 't2':1E-13, 'a1':1, 'a2':1}
        self.para_unit = {'FWHM':1E-1/2.35482, 'x0':1, 't1':1E-1, 't2':1E-1, 'a1':1, 'a2':1}

        self.need_negative = need_negative
        self.weighted_average = weighted_average
        self.runList = run_list
        self.runNum = run_num
        self.temporary_mask = False

        self.input_common_root = input_file_common_root
        self.infile_name = given_file_name

        file_path = input_file_common_root + given_file_name
        print("now read file : ", file_path)
        self.input_file = open(file_path, 'r')
        self.read_input_file()

        time = datetime.datetime.now()
        year = time.year
        month = time.month
        day = time.day
        hour = time.hour
        minute = time.minute
        sec = time.second
        now_time = "{0}-{1}-{2}-{3}hr.{4}min.{5}sec".format(year, month, day, hour, minute, sec)
        self.now_time = now_time

        self.min_validity_parameter_val_list = []
        self.chi2_save_list = []
        if weighted_average:
            if self.anisotropy_process:
                if len(run_list) == 2:
                    file_out_dir_path = './anisotropy_fit_result/run{0:04d}_{1:04d}_avg/anisotropy'.format(run_list[0], run_list[1])
                    self.file_out_path = file_out_dir_path + '/run{0:04d}_{1:04d}_avg_{2}'.format(run_list[0], run_list[1], now_time)
                    self.previous_npy_load_path = file_out_dir_path + '/run{0:04d}_{1:04d}_avg_{2}'.format(run_list[0], run_list[1], previous_result_time_stamp)
                elif len(run_list) == 3:
                    file_out_dir_path = './anisotropy_fit_result/run{0:04d}_{1:04d}_{2:04d}_avg/anisotropy'.format(run_list[0], run_list[1], run_list[2])
                    self.file_out_path = file_out_dir_path + '/run{0:04d}_{1:04d}_{2:04d}_avg_{3}'.format(run_list[0], run_list[1], run_list[2], now_time)
                    self.previous_npy_load_path = file_out_dir_path + '/run{0:04d}_{1:04d}_{2:04d}_avg_{3}'.format(run_list[0], run_list[1], run_list[2], previous_result_time_stamp)
            elif self.heating_process:
                if len(run_list) == 2:
                    file_out_dir_path = './anisotropy_fit_result/run{0:04d}_{1:04d}_avg/heating'.format(run_list[0], run_list[1])
                    self.file_out_path = file_out_dir_path + '/run{0:04d}_{1:04d}_avg_{2}'.format(run_list[0], run_list[1], now_time)
                    self.previous_npy_load_path = file_out_dir_path + '/run{0:04d}_{1:04d}_avg_{2}'.format(run_list[0], run_list[1], previous_result_time_stamp)
                elif len(run_list) == 3:
                    file_out_dir_path = './anisotropy_fit_result/run{0:04d}_{1:04d}_{2:04d}_avg/heating'.format(run_list[0], run_list[1], run_list[2])
                    self.file_out_path = file_out_dir_path + '/run{0:04d}_{1:04d}_{2:04d}_avg_{3}'.format(run_list[0], run_list[1], run_list[2], now_time)
                    self.previous_npy_load_path = file_out_dir_path + '/run{0:04d}_{1:04d}_{2:04d}_avg_{3}'.format(run_list[0], run_list[1], run_list[2], previous_result_time_stamp)
        else:
            if self.anisotropy_process:
                file_out_dir_path = './anisotropy_fit_result/run{0:04d}/anisotropy'.format(run_num)
                self.file_out_path = file_out_dir_path + '/run{0:04d}_{1}'.format(run_num, now_time)
                self.previous_npy_load_path = file_out_dir_path + '/run{0:04d}_{1}'.format(run_num, previous_result_time_stamp)
            elif self.heating_process:
                file_out_dir_path = './anisotropy_fit_result/run{0:04d}/heating'.format(run_num)
                self.file_out_path = file_out_dir_path + '/run{0:04d}_{1}'.format(run_num, now_time)
                self.previous_npy_load_path = file_out_dir_path + '/run{0:04d}_{1}'.format(run_num, previous_result_time_stamp)

        if os.path.isdir(file_out_dir_path):
            pass
        else:
            if self.anisotropy_process:
                if weighted_average:
                    if len(run_list) == 2:
                        os.makedirs("./anisotropy_fit_result/run{0:04d}_{1:04d}_avg/anisotropy".format(run_list[0], run_list[1]))
                    elif len(run_list) == 3:
                        os.makedirs("./anisotropy_fit_result/run{0:04d}_{1:04d}_{2:04d}_avg/anisotropy".format(run_list[0], run_list[1], run_list[2]))
                else:
                    os.makedirs("./anisotropy_fit_result/run{0:04d}/anisotropy".format(run_num))
            elif self.heating_process:
                if weighted_average:
                    if len(run_list) == 2:
                        os.makedirs("./anisotropy_fit_result/run{0:04d}_{1:04d}_avg/heating".format(run_list[0], run_list[1]))
                    elif len(run_list) == 3:
                        os.makedirs("./anisotropy_fit_result/run{0:04d}_{1:04d}_{2:04d}_avg/heating".format(run_list[0], run_list[1], run_list[2]))
                else:
                    os.makedirs("./anisotropy_fit_result/run{0:04d}/heating".format(run_num))



    def read_input_file(self):
        front_skip_line_num = 1
        is_front_line_skipped = False
        file_time_val = []
        file_data = []
        front_skipped_data = None

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
                for each_split_data in now_split_text:
                    now_split_float.append(float(each_split_data))
                file_time_val.append(now_split_float[0])
                #file_data.append(now_split_float[1:])
                try:
                    file_data.append(now_split_float[self.target_RSV])  #Collects anisotropy signal's RSV only
                except:
                    break

        self.input_time_delay = np.array(file_time_val)
        # self.input_time_delay = np.round(self.input_time_delay * 1E-12, 15)
        self.input_raw_data = np.array(file_data)
        self.input_raw_data_len = len(file_time_val)
        self.front_delay_info_read(front_skipped_data)
        if self.anisotropy_process:
                if self.aniso_double_exp:
                    for time_idx in range(len(self.input_time_delay)):
                        if self.input_time_delay[time_idx] == 3E-12:
                            self.input_time_delay = self.input_time_delay[:(time_idx+1)]
                            self.input_raw_data = self.input_raw_data[:(time_idx + 1)]
                            # if self.input_time_delay[time_idx] == 3E-12:
                        #     exp_calc_time_start_time_idx = time_idx
                        # if self.input_time_delay[time_idx] == 10E-12: #TODO change after done run 35
                        #     self.input_time_delay = self.input_time_delay[:(time_idx+1)]
                        #     self.input_raw_data = self.input_raw_data[:(time_idx+1)]
                        #     self.exp_calc_time_delay = self.input_time_delay[exp_calc_time_start_time_idx:(time_idx+1)]
                            break
                        else:
                            continue
                elif self.aniso_stretched_exp:
                    for time_idx in range(len(self.input_time_delay)):
                        if self.input_time_delay[time_idx] == 3E-12:  # TODO change after done run 35
                            self.input_time_delay = self.input_time_delay[:(time_idx + 1)]
                            self.input_raw_data = self.input_raw_data[:(time_idx + 1)]
                            break
                        else:
                            continue

        elif self.heating_process:
            for time_idx in range(len(self.input_time_delay)):
                if self.input_time_delay[time_idx] < 30E-12: #TODO change after done run 35
                    self.input_time_delay = self.input_time_delay[:time_idx]
                    self.input_raw_data = self.input_raw_data[:time_idx]
                    break
                else:
                    continue
        if self.need_negative:
            self.input_raw_data = -self.input_raw_data
        file_data = []

    def front_delay_info_read(self, front_skipped_data):
        first_line_data = front_skipped_data[0]
        split_info = first_line_data.split()
        print(split_info)
        rsv_component_list = []
        for each_component_text in split_info[1:]:
            rsv_component_list.append(str(each_component_text))
        print(rsv_component_list)
        self.rsv_component_list = rsv_component_list

    def custom_function_fit(self):
        raw_data_plot = False
        # raw_data_plot = True
        all_delay_show = True

        test_time_delay_idx = 1

        if raw_data_plot:
            if all_delay_show:
                plt.title("all delay raw data")
                for each_delay_idx in range(len(self.rsv_component_list)):
                    now_time_delay = self.rsv_component_list[each_delay_idx]
                    now_label = "{0}th delay ({1})".format(each_delay_idx, now_time_delay)
                    plt.plot(self.input_time_delay, self.input_raw_data[:, each_delay_idx], label=now_label)
                plt.ylim((-50, 50))
                plt.legend()
                plt.show()

            now_time_delay = self.rsv_component_list[test_time_delay_idx]
            now_title = "{0}th time delay {1} raw data".format(test_time_delay_idx, now_time_delay)
            plt.title(now_title)
            plt.plot(self.input_time_delay, self.input_raw_data[:, test_time_delay_idx])
            plt.ylim((-50, 50))
            plt.show()


        def exp_decay(q_val, a1, t1, x0, y0, a2, t2):
            exponential_temp = []
            if self.aniso_double_exp:
                if len(q_val) != 1:
                    for q_idx in range(len(q_val)):
                        if (q_val[q_idx]-x0*self.para_unit['x0']) > 0 :
                            exponential_temp.append(a1*self.para_unit['a1'] * np.exp(-((q_val[q_idx] - x0*self.para_unit['x0'])/(t1*self.para_unit['t1']))) + a2*self.para_unit['a2'] * np.exp(-((q_val[q_idx] - x0*self.para_unit['x0'])/(t2*self.para_unit['t2'])))+y0)
                        else:
                            exponential_temp.append(float(0))
                            # exponential_temp.append(y0)
                    return exponential_temp

                elif len(q_val) == 1:
                    if (q_val[0] - x0 * self.para_unit['x0']) > 0:
                        calc_val = (a1 * self.para_unit['a1'] * np.exp(-((q_val[0] - x0 * self.para_unit['x0']) / (t1 * self.para_unit['t1']))) + a2 *self.para_unit['a2'] * np.exp(-((q_val[0] - x0 * self.para_unit['x0']) / (t2 * self.para_unit['t2']))) + y0)
                    else:
                        calc_val = float(0)
                    return calc_val
            elif self.aniso_single_exp:
                if len(q_val) != 1:
                    for q_idx in range(len(q_val)):
                        if (q_val[q_idx] - x0 * self.para_unit['x0']) > 0:
                            exponential_temp.append(a1 * self.para_unit['a1'] * np.exp(-((q_val[q_idx] - x0 * self.para_unit['x0']) / (t1 * self.para_unit['t1'])))+ y0)
                        else:
                            exponential_temp.append(float(0))

                    return exponential_temp

                elif len(q_val) == 1:
                    if (q_val[0] - x0 * self.para_unit['x0']) > 0:
                        calc_val = (a1 * self.para_unit['a1'] * np.exp(-((q_val[0] - x0 * self.para_unit['x0']) / (t1 * self.para_unit['t1']))) +  + y0)
                    else:
                        calc_val = float(0)
                    return calc_val

        def stretched_exp(time_list, a, t, beta, x0, y0):
            stretched_exp_temp = []
            if len(time_list) != 1:
                for q_idx in range(len(time_list)):
                    if (time_list[q_idx] - x0 * self.para_unit['x0']) > 0:
                        stretched_exp_temp.append(a * self.para_unit['a1'] * np.exp(-(((time_list[q_idx] - x0 * self.para_unit['x0'])/(t*self.para_unit['t1']))**beta)) + y0)
                    else:
                        stretched_exp_temp.append(float(0))
                return stretched_exp_temp

            elif len(time_list) == 1:
                if (time_list[0] - x0 * self.para_unit['x0']) > 0:
                    calc_val = (a * self.para_unit['a1'] * np.exp(-(((time_list[0] - x0 * self.para_unit['x0'])/(t*self.para_unit['t1']))**beta)) + y0)
                else:
                    calc_val = float(0)
                return calc_val

        def gaussian(q_val, x0, FWHM):
            gaussian_temp = (1 / (FWHM * self.para_unit['FWHM'] * (np.sqrt(2 * np.pi))) * np.exp(-pow((q_val-x0)/(FWHM * self.para_unit['FWHM']), 2.0) / 2.0))
            return gaussian_temp

        def double_exponential_function_convolution(time_list, a1, t1, x0, y0, a2, t2, FWHM):
            # t2 = t1+t2
            time_list = self.calculate_time_list
            gaussian_temp = gaussian(time_list, FWHM)
            temp_convoluted_function = []
            convoluted_function = []
            for time_point_idx in range(len(time_list)):
                if time_point_idx == 0:
                    temp_calc_time = []
                    for calc_idx in range(self.num_of_calculation):
                        temp_calc_time.append(time_list[time_point_idx] - time_list[calc_idx])
                    exponential_temp = exp_decay(temp_calc_time, a1, t1, x0, y0, a2, t2)
                else:
                    temp_time_list = []
                    temp_time_list.append(time_list[time_point_idx] - time_list[0])
                    exponential_temp = np.roll(exponential_temp, 1)
                    exponential_temp[0] = exp_decay(temp_time_list, a1, t1, x0, y0, a2, t2)
                temp_calc_val = np.multiply(exponential_temp, gaussian_temp)*time_step
                temp_convolution_val = np.sum(temp_calc_val)
                temp_convoluted_function.append([time_list[time_point_idx], temp_convolution_val])
            # exponential_temp = exp_decay(self.exp_calc_time_delay, a1, t1, x0, y0, a2, t2)
            # temp_merged_exp = []
            # for time_point_idx in range(len(self.exp_calc_time_delay)):
            #     if time_point_idx == 0:
            #         temp_convoluted_function[self.num_of_calculation-1][1] = exponential_temp[time_point_idx]
            #     else:
            #         temp_convoluted_function.append([self.exp_calc_time_delay[time_point_idx], exponential_temp[time_point_idx]])
            gaussian_temp = []

            for idx in range(len(time_list)):
            # for idx in range(len(time_list)+len(self.exp_calc_time_delay)-1):
                if self.temporary_mask:
                    if time_list[idx] <= 0.5E-12:
                        if temp_convoluted_function[idx][0] in self.input_time_delay:
                            convoluted_function.append(temp_convoluted_function[idx][1])
                        else:
                            continue
                    else:
                        continue
                else:
                    if temp_convoluted_function[idx][0] in self.input_time_delay:
                        convoluted_function.append(temp_convoluted_function[idx][1])
                    else:
                        continue

            return convoluted_function

        def function_convolution_to_final_fit(time_list, a1, t1, x0, y0, a2, t2, FWHM):
            temp_calc_time = []
            time_list = self.calculate_time_list
            gaussian_temp = gaussian(time_list, x0, FWHM)
            temp_convoluted_function = []
            if self.aniso_single_exp:
                a2 = 0
                t2 = 0
            # for time_point_idx in range(len(time_list)):
            #     if time_point_idx == 0:
            #         temp_calc_time = []
            #         for calc_idx in range(self.num_of_calculation):
            #             temp_calc_time.append(time_list[time_point_idx] - time_list[calc_idx])
            #         exponential_temp = exp_decay(temp_calc_time, a1, t1, x0, y0, a2, t2)
            #     else:
            #         temp_time_list = []
            #         temp_time_list.append(time_list[time_point_idx] - time_list[0])
            #         exponential_temp = np.roll(exponential_temp, 1)
            #         exponential_temp[0] = exp_decay(temp_time_list, a1, t1, x0, y0, a2, t2)
            #     temp_calc_val = np.multiply(exponential_temp, gaussian_temp)*time_step
            #     temp_convolution_val = np.sum(temp_calc_val)
            #     temp_convoluted_function.append(temp_convolution_val)
            for time_point_idx in range(len(time_list)):
                if time_point_idx == 0:
                    temp_calc_time = []
                    for calc_idx in range(self.num_of_calculation):
                        temp_calc_time.append(time_list[time_point_idx] - time_list[calc_idx])
                    exponential_temp = exp_decay(temp_calc_time, a1, t1, x0, y0, a2, t2)
                else:
                    temp_time_list = []
                    temp_time_list.append(time_list[time_point_idx] - time_list[0])
                    exponential_temp = np.roll(exponential_temp, 1)
                    exponential_temp[0] = exp_decay(temp_time_list, a1, t1, x0, y0, a2, t2)
                temp_calc_val = np.multiply(exponential_temp, gaussian_temp)*time_step
                temp_convolution_val = np.sum(temp_calc_val)
                temp_convoluted_function.append(temp_convolution_val)
            # temp_calc_time = []
            # for idx in range(1101):
            #     temp_calc_time.append(start_time + 10E-15 * idx)
            # self.calculate_time_list = []
            # self.exp_calc_time_delay = []
            # self.calculate_time_list = temp_calc_time
            # self.calculate_time_list = np.round(self.calculate_time_list, 15)
            # self.exp_calc_time_delay = self.calculate_time_list[(self.num_of_calculation - 1):]
            # exponential_temp = exp_decay(self.exp_calc_time_delay, a1, t1, x0, y0, a2, t2)
            # temp_merged_exp = []
            # for time_point_idx in range(len(self.exp_calc_time_delay)):
            #     if time_point_idx == 0:
            #         temp_convoluted_function[self.num_of_calculation-1] = exponential_temp[time_point_idx]
            #     else:
            #         temp_convoluted_function.append(exponential_temp[time_point_idx])
            return temp_convoluted_function

        def new_convolution_double_exp(time_list, a1, t1, x0, y0, a2, t2, FWHM):
            # self.calculate_time_list = np.round(np.arange(-1, 3.001, 10E-3), 3)
            time_list = self.calculate_time_list
            gaussian_temp = gaussian(time_list, x0, FWHM)
            temp_time = time_step * (num_of_calculation - 1)
            temp_time_arr = np.linspace(-temp_time, temp_time, num_of_calculation * 2 -1)
            exponential_temp = exp_decay(temp_time_arr, a1, t1, x0, y0, a2, t2)
            temp_convoluted_function = []
            convoluted_function = []
            temp_convoluted_function = np.convolve(exponential_temp, gaussian_temp, mode='valid') * time_step
            # temp_arr = np.concatenate(temp_convoluted_function, time_list, axis=0)
            for idx in range(len(time_list)):
            # for idx in range(len(time_list)+len(self.exp_calc_time_delay)-1):
                if time_list[idx] in self.input_time_delay:
                    convoluted_function.append(temp_convoluted_function[idx])
                else:
                    continue
            return convoluted_function

        def new_convolution_double_exp_to_final_fit(time_list, a1, t1, x0, y0, a2, t2, FWHM):
            self.calculate_time_list = np.round(np.arange(-1, 3.001, 10E-3), 3)
            time_list = self.calculate_time_list
            gaussian_temp = gaussian(time_list, x0, FWHM)
            temp_time = time_step * (num_of_calculation - 1)
            temp_time_arr = np.linspace(-temp_time, temp_time, num_of_calculation * 2 -1)
            exponential_temp = exp_decay(temp_time_arr, a1, t1, x0, y0, a2, t2)
            temp_convoluted_function = []
            temp_convoluted_function = np.convolve(exponential_temp, gaussian_temp, mode='valid') * time_step
            return temp_convoluted_function

        def single_exponential_funciton_convolution(time_list, a1, t1, x0, y0, FWHM):
            time_list = self.calculate_time_list
            gaussian_temp = gaussian(time_list, FWHM)
            a2 = 0
            t2 = 0
            temp_convoluted_function = []
            convoluted_function = []
            for time_point_idx in range(len(time_list)):
                if time_point_idx == 0:
                    temp_calc_time = []
                    for calc_idx in range(self.num_of_calculation):
                        temp_calc_time.append(time_list[time_point_idx] - time_list[calc_idx])

                    exponential_temp = exp_decay(temp_calc_time, a1, t1, x0, y0, a2, t2)
                else:
                    temp_time_list = []
                    temp_time_list.append(time_list[time_point_idx] - time_list[0])
                    exponential_temp = np.roll(exponential_temp, 1)
                    exponential_temp[0] = exp_decay(temp_time_list, a1, t1, x0, y0, a2, t2)
                temp_calc_val = np.multiply(exponential_temp, gaussian_temp) * time_step
                temp_convolution_val = np.sum(temp_calc_val)
                temp_convoluted_function.append([time_list[time_point_idx], temp_convolution_val])
            gaussian_temp = []

            for idx in range(len(time_list)):
                if self.temporary_mask:
                    if time_list[idx] <= 0.5E-12:
                        if temp_convoluted_function[idx][0] in self.input_time_delay:
                            convoluted_function.append(temp_convoluted_function[idx][1])
                        else:
                            continue
                    else:
                        continue
                else:
                    if temp_convoluted_function[idx][0] in self.input_time_delay:
                        convoluted_function.append(temp_convoluted_function[idx][1])
                    else:
                        continue

            return convoluted_function

        def single_exponential_function_convolution_to_final_fit(time_list, a1, t1, x0, y0, FWHM):
            time_list = self.calculate_time_list
            gaussian_temp = gaussian(time_list, FWHM)
            temp_convoluted_function = []
            if self.aniso_single_exp:
                a2 = 0
                t2 = 0
            for time_point_idx in range(len(time_list)):
                if time_point_idx == 0:
                    temp_calc_time = []
                    for calc_idx in range(self.num_of_calculation):
                        temp_calc_time.append(time_list[time_point_idx] - time_list[calc_idx])
                    exponential_temp = exp_decay(temp_calc_time, a1, t1, x0, y0, a2, t2)
                else:
                    temp_time_list = []

                    temp_time_list.append(time_list[time_point_idx] - time_list[0])
                    exponential_temp = np.roll(exponential_temp, 1)
                    exponential_temp[0] = exp_decay(temp_time_list, a1, t1, x0, y0, a2, t2)
                temp_calc_val = np.multiply(exponential_temp, gaussian_temp)*time_step
                temp_convolution_val = np.sum(temp_calc_val)
                temp_convoluted_function.append(temp_convolution_val)

            return temp_convoluted_function

        def stretched_exponential_function_convolution(time_list, a, t, beta, x0, y0, FWHM):
            time_list = self.calculate_time_list
            gaussian_temp = gaussian(time_list, FWHM)
            temp_convoluted_function = []
            convoluted_function = []
            for time_point_idx in range(len(time_list)):
                if time_point_idx == 0:
                    temp_calc_time = []
                    for calc_idx in range(self.num_of_calculation):
                        temp_calc_time.append(time_list[time_point_idx] - time_list[calc_idx])
                    stretched_exponential_temp = stretched_exp(temp_calc_time, a, t, beta, x0, y0)
                else:
                    temp_time_list = []
                    temp_time_list.append(time_list[time_point_idx] - time_list[0])
                    stretched_exponential_temp = np.roll(stretched_exponential_temp, 1)
                    stretched_exponential_temp[0] = stretched_exp(temp_time_list, a, t, beta, x0, y0)
                temp_calc_val = np.multiply(stretched_exponential_temp, gaussian_temp) * time_step
                temp_convolution_val = np.sum(temp_calc_val)
                temp_convoluted_function.append([time_list[time_point_idx], temp_convolution_val])
            gaussian_temp = []

            for idx in range(len(time_list)):
                if self.temporary_mask:
                    if time_list[idx] <= 0.5E-12:
                        if temp_convoluted_function[idx][0] in self.input_time_delay:
                            convoluted_function.append(temp_convoluted_function[idx][1])
                        else:
                            continue
                    else:
                        continue
                else:
                    if temp_convoluted_function[idx][0] in self.input_time_delay:
                        convoluted_function.append(temp_convoluted_function[idx][1])
                    else:
                        continue

            return convoluted_function

        def stretched_exponential_function_convolution_to_final_fit(time_list, a, t, beta, x0, y0, FWHM):
            time_list = self.calculate_time_list
            gaussian_temp = gaussian(time_list, FWHM)
            temp_convoluted_function = []
            convoluted_function = []
            for time_point_idx in range(len(time_list)):
                if time_point_idx == 0:
                    temp_calc_time = []
                    for calc_idx in range(self.num_of_calculation):
                        temp_calc_time.append(time_list[time_point_idx] - time_list[calc_idx])
                    stretched_exponential_temp = stretched_exp(temp_calc_time, a, t, beta, x0, y0)
                else:
                    temp_time_list = []
                    temp_time_list.append(time_list[time_point_idx] - time_list[0])
                    stretched_exponential_temp = np.roll(stretched_exponential_temp, 1)
                    stretched_exponential_temp[0] = stretched_exp(temp_time_list, a, t, beta, x0, y0)
                temp_calc_val = np.multiply(stretched_exponential_temp, gaussian_temp) * time_step
                temp_convolution_val = np.sum(temp_calc_val)
                temp_convoluted_function.append(temp_convolution_val)
            gaussian_temp = []

            return temp_convoluted_function

        def exp_heating(q_val, a2, t2, x0):
            exponential_temp = []
            for q_idx in range(len(q_val)):
                if (q_val[q_idx]-x0*self.para_unit['x0']) > 0 :
                    exponential_temp.append(a2*self.para_unit['a2'] * np.exp(-((q_val[q_idx] - x0*self.para_unit['x0'])/(t2*self.para_unit['t2']))) -a2)
                    # if temp_val >= 0:
                    #     exponential_temp.append(temp_val)
                    # else :
                    #     exponential_temp.append(0.0)
                else:
                    exponential_temp.append(0.0)
            return exponential_temp

        def random_initialize(num_of_parameters, num_of_random_initialization):

            param_num = num_of_parameters
            random_try_num = num_of_random_initialization
            random_arr = np.random.rand(param_num, random_try_num)
            specified_oom = {'t2' : '3'}

            transfer_rand_arr_list = []
            for idx_param in range(param_num):
                if idx_param == 5:
                    now_max_oom = specified_oom['t2'] #TODO dictionary value can be a list. Use list when min_oom has to be change too
                else:
                    now_max_oom = 2
                now_min_oom = -3
                digit_random_arr = np.random.randint(low=now_min_oom, high=now_max_oom, size=(2, random_try_num))
                now_l_lim, now_r_lim = self.param_limit[idx_param]
                if now_l_lim is None:
                    now_l_lim = np.power(10.0, digit_random_arr[0])
                if now_r_lim is None:
                    now_r_lim = np.power(10.0, digit_random_arr[1])
                transfer_rand_arr = random_arr[idx_param] * now_r_lim + now_l_lim
                transfer_rand_arr_list.append(transfer_rand_arr)
            random_inits = np.array(transfer_rand_arr_list)
            return random_inits

        def save_fit_result_as_dat(data, calculated_data, exp_data, parameter_random_val_list, best_fit_idx):
            dat_file = open(self.file_out_path+'.dat', 'w')
            if self.anisotropy_process:
                if self.aniso_double_exp:
                    parameter_name_list = ['chi2', 'a1', 't1', 'x0', 'y0', 'a2', 't2', 'FWHM']
                elif self.aniso_single_exp:
                    parameter_name_list = ['chi2', 'a1', 't1', 'x0', 'y0', 'FWHM']
                elif self.aniso_stretched_exp:
                    parameter_name_list = ['chi2', 'a', 't', 'beta', 'x0', 'y0', 'FWHM']
            elif self.heating_process:
                parameter_name_list = ['chi2', 'a2', 't2', 'x0']
            if random_initialization:
                if self.anisotropy_process:
                    first_row_data = ['parameters', 'fitted values', 'time', 'exp data', 'calculate time', 'calculated value']
                elif self.heating_process:
                    first_row_data = ['parameters', 'fitted values', 'time', 'exp data', 'calculated value for ratio', 'calculate time', 'calculated value']
                    calc_val_list_for_ratio = []
                    for idx in range(len(self.calculate_time_list)):
                        if self.calculate_time_list[idx] in self.input_time_delay:
                            calc_val_list_for_ratio.append(calculated_data[idx])
                        else:
                            continue
                #first_row_data = ['parameters', 'initial value', 'fitted values', 'time', 'exp data', 'calculate time', 'calculated value']
                if self.heating_process:
                    for idx in range(len(first_row_data)):
                        if (idx + 1) == len(first_row_data):
                            dat_file.write(first_row_data[idx]+ "\n")
                        else:
                            dat_file.write(first_row_data[idx]+ "\t")

                    for idx in range(len(calculated_data)):
                        if idx < len(data) and idx == 0:
                            dat_file.write(str(parameter_name_list[idx])+ "\t")
                            dat_file.write(str(np.min(self.chi2_save_list))+ "\t")
                            dat_file.write(str(self.input_time_delay[idx]) + "\t")
                            dat_file.write(str(exp_data[idx])+ "\t")
                            dat_file.write(str(calc_val_list_for_ratio[idx])+ "\t")
                            dat_file.write(str(self.calculate_time_list[idx]) + "\t")
                            dat_file.write(str(calculated_data[idx])+ "\n")
                            continue

                        if idx < (len(data) + 1):
                            dat_file.write(str(parameter_name_list[idx])+ "\t")
                            dat_file.write(str(data[idx-1])+ "\t")
                            dat_file.write(str(self.input_time_delay[idx]) + "\t")
                            dat_file.write(str(exp_data[idx])+ "\t")
                            dat_file.write(str(calc_val_list_for_ratio[idx])+ "\t")
                            dat_file.write(str(self.calculate_time_list[idx]) + "\t")
                            dat_file.write(str(calculated_data[idx])+ "\n")
                        else:
                            if idx < len(exp_data):
                                dat_file.write("\t" + "\t" + str(self.input_time_delay[idx]) + "\t")
                                dat_file.write(str(exp_data[idx]) + "\t")
                                dat_file.write(str(calc_val_list_for_ratio[idx]) + "\t")
                                dat_file.write(str(self.calculate_time_list[idx]) + "\t")
                                dat_file.write(str(calculated_data[idx]) + "\n")
                            else:
                                dat_file.write("\t" + "\t" + "\t" + "\t" + "\t"+ str(self.calculate_time_list[idx]) + "\t")
                                dat_file.write(str(calculated_data[idx]) + "\n")
                elif self.anisotropy_process:
                    for idx in range(len(first_row_data)):
                        if (idx + 1) == len(first_row_data):
                            dat_file.write(first_row_data[idx] + "\n")
                        else:
                            dat_file.write(first_row_data[idx] + "\t")

                    for idx in range(len(calculated_data)):
                        if idx < len(data) and idx == 0:
                            dat_file.write(str(parameter_name_list[idx]) + "\t")
                            dat_file.write(str(np.min(self.chi2_save_list)) + "\t")
                            dat_file.write(str(self.input_time_delay[idx]) + "\t")
                            dat_file.write(str(exp_data[idx]) + "\t")
                            dat_file.write(str(self.calculate_time_list[idx]) + "\t")
                            dat_file.write(str(calculated_data[idx]) + "\n")
                            continue

                        if idx < (len(data) + 1):
                            dat_file.write(str(parameter_name_list[idx]) + "\t")
                            dat_file.write(str(data[idx - 1]) + "\t")
                            dat_file.write(str(self.input_time_delay[idx]) + "\t")
                            dat_file.write(str(exp_data[idx]) + "\t")
                            dat_file.write(str(self.calculate_time_list[idx]) + "\t")
                            dat_file.write(str(calculated_data[idx]) + "\n")
                        else:
                            if idx < len(exp_data):
                                dat_file.write("\t" + "\t" + str(self.input_time_delay[idx]) + "\t")
                                dat_file.write(str(exp_data[idx]) + "\t")
                                dat_file.write(str(self.calculate_time_list[idx]) + "\t")
                                dat_file.write(str(calculated_data[idx]) + "\n")
                            else:
                                dat_file.write("\t" + "\t" + "\t" + "\t" + str(self.calculate_time_list[idx]) + "\t")
                                dat_file.write(str(calculated_data[idx]) + "\n")

                min_chi2 = []
                min_chi2.append(np.min(self.chi2_save_list))
                temp_save_arr = np.hstack((min_chi2, data))
                np.save(self.file_out_path + '.npy', temp_save_arr)

            else:
                first_row_data = ['parameters', 'values', 'time', 'exp data', 'calculate time', 'calculated value']
                for idx in range(len(first_row_data)):
                    if (idx + 1) == len(first_row_data):
                        dat_file.write(first_row_data[idx] + "\n")
                    else:
                        dat_file.write(first_row_data[idx] + "\t")

                for idx in range(len(calculated_data)):
                    if idx < len(data):
                        dat_file.write(str(parameter_name_list[idx]) + "\t")
                        dat_file.write(str(data[idx]) + "\t")
                        dat_file.write(str(self.input_time_delay[idx]) + "\t")
                        dat_file.write(str(exp_data[idx]) + "\t")
                        dat_file.write(str(self.calculate_time_list[idx]) + "\t")
                        dat_file.write(str(calculated_data[idx]) + "\n")

                    else:
                        if idx < len(exp_data):
                            # dat_file.write(data[idx] + "\t")
                            dat_file.write("\t" + "\t" + str(self.input_time_delay[idx]) + "\t")
                            dat_file.write(str(exp_data[idx]) + "\t")
                            dat_file.write(str(self.calculate_time_list[idx]) + "\t")
                            dat_file.write(str(calculated_data[idx]) + "\n")

                        else:
                            dat_file.write("\t" + "\t" + "\t" + "\t" + str(self.calculate_time_list[idx]) + "\t")
                            dat_file.write(str(calculated_data[idx]) + "\n")

        def write_log_info(calc_idx):
            if self.weighted_average:
                if self.anisotropy_process:
                    if len(self.runList) == 2:
                        if self.aniso_single_exp:
                            out_file_name = "/data/exp_data/myeong0609/PAL-XFEL_20210514/analysis/minuit/run_{0}_{1}_single_exp_log.txt".format(self.runList[0], self.runList[1])
                        elif self.aniso_double_exp:
                            out_file_name = "/data/exp_data/myeong0609/PAL-XFEL_20210514/analysis/minuit/run_{0}_{1}_double_exp_log.txt".format(self.runList[0], self.runList[1])
                        elif self.aniso_stretched_exp:
                            out_file_name = "/data/exp_data/data/exp_data/myeong0609/PAL-XFEL_20210514/analysis/minuit/run_{0}_{1}_stretched_exp_log.txt".format(self.runList[0], self.runList[1])

                    elif len(self.runList) == 3:
                        if self.aniso_single_exp:
                            out_file_name = "/data/exp_data/myeong0609/PAL-XFEL_20210514/analysis/minuit/run_{0}_{1}_{2}_single_exp_log.txt".format(self.runList[0], self.runList[1], self.runList[2])
                        elif self.aniso_double_exp:
                            out_file_name = "/data/exp_data/myeong0609/PAL-XFEL_20210514/analysis/minuit/run_{0}_{1}_{2}_double_exp_log.txt".format(self.runList[0], self.runList[1], self.runList[2])
                        elif self.aniso_stretched_exp:
                            out_file_name = "/data/exp_data/myeong0609/PAL-XFEL_20210514/analysis/minuit/run_{0}_{1}_{2}_stretched_exp_log.txt".format(self.runList[0], self.runList[1], self.runList[2])
                elif self.heating_process:
                    if len(self.runList) == 2:
                        out_file_name = "/data/exp_data/myeong0609/PAL-XFEL_20210514/analysis/minuit/run_{0}_{1}_heating_log.txt".format(self.runList[0], self.runList[1])

                    elif len(self.runList) == 3:
                        out_file_name = "/data/exp_data/myeong0609/PAL-XFEL_20210514/analysis/minuit/run_{0}_{1}_{2}_heating_log.txt".format(self.runList[0], self.runList[1], self.runList[2])
            else:
                if self.anisotropy_process:
                    if self.aniso_single_exp:
                        out_file_name = "/data/exp_data/myeong0609/PAL-XFEL_20210514/analysis/minuit/run_" + str(self.runNum) + "_single_exp_log.txt"
                    elif self.aniso_double_exp:
                        out_file_name = "/data/exp_data/myeong0609/PAL-XFEL_20210514/analysis/minuit/run_" + str(self.runNum) + "_double_exp_log.txt"
                    elif self.aniso_stretched_exp:
                        out_file_name = "/data/exp_data/myeong0609/PAL-XFEL_20210514/analysis/minuit/run_" + str(self.runNum) + "_stretched_exp_log.txt"

                elif self.heating_process:
                    out_file_name = "/data/exp_data/myeong0609/PAL-XFEL_20210514/analysis/minuit/run_" + str(self.runNum) + "_heating_log.txt"

            if num_of_random_initialization > 1000:
                if calc_idx == 99:
                    try:
                        log = open(out_file_name, 'r')
                        os.remove(out_file_name)
                        # if self.weighted_average:
                        #     if len(self.runList) == 2:
                        #         os.remove("/home/myeong0609/PAL-XFEL_20210514/analysis/minuit/run_{0}_{1}_log.txt".format(self.runList[0], self.runList[1]))
                        #     elif len(self.runList) == 3:
                        #         os.remove("/home/myeong0609/PAL-XFEL_20210514/analysis/minuit/run_{0}_{1}_{2}_log.txt".format(self.runList[0], self.runList[1], self.runList[2]))
                        # else:
                        #     os.remove("/home/myeong0609/PAL-XFEL_20210514/analysis/minuit/run_" + str(self.runNum) + "_log.txt")
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
                    if self.chi2_save_list:
                        contents = "now try : " + str(calc_idx + 1) + ", minimum chi2 : " + str(np.min(self.chi2_save_list)) + "\n"
                        log.write(contents)
                    else:
                        contents = "now try : " + str(calc_idx + 1) + ", minimum chi2 : none" + "\n"
                        log.write(contents)
                else:
                    log = open(out_file_name, 'a')
                    if self.chi2_save_list:
                        contents = "now try : " + str(calc_idx + 1) + ", minimum chi2 : " + str(np.min(self.chi2_save_list)) + "\n"
                        log.write(contents)
                    else:
                        contents = "now try : " + str(calc_idx + 1) + ", minimum chi2 : none" + "\n"
                        log.write(contents)
            else:
                if calc_idx == 4:
                    try:
                        log = open(out_file_name, 'r')
                        os.remove(out_file_name)
                        # if self.weighted_average:
                        #     if self.aniso_double_exp:
                        #         if len(self.runList) == 2:
                        #             os.remove("/home/myeong0609/PAL-XFEL_20210514/analysis/minuit/run_{0}_{1}_double_exp_log.txt".format(self.runList[0], self.runList[1]))
                        #         elif len(self.runList) == 3:
                        #             os.remove("/home/myeong0609/PAL-XFEL_20210514/analysis/minuit/run_{0}_{1}_{2}_double_exp_log.txt".format(self.runList[0], self.runList[1], self.runList[2]))
                        #     elif self.aniso_single_exp:
                        #         if len(self.runList) == 2:
                        #             os.remove("/home/myeong0609/PAL-XFEL_20210514/analysis/minuit/run_{0}_{1}_single_exp_log.txt".format(self.runList[0], self.runList[1]))
                        #         elif len(self.runList) == 3:
                        #             os.remove("/home/myeong0609/PAL-XFEL_20210514/analysis/minuit/run_{0}_{1}_{2}_single_exp_log.txt".format(self.runList[0], self.runList[1], self.runList[2]))
                        #     elif self.heating_process:
                        #         if len(self.runList) == 2:
                        #             os.remove("/home/myeong0609/PAL-XFEL_20210514/analysis/minuit/run_{0}_{1}_heating_log.txt".format(self.runList[0], self.runList[1]))
                        #         elif len(self.runList) == 3:
                        #             os.remove("/home/myeong0609/PAL-XFEL_20210514/analysis/minuit/run_{0}_{1}_{2}_heating_log.txt".format(self.runList[0], self.runList[1], self.runList[2]))
                        # else:
                        #     if self.aniso_double_exp:
                        #         os.remove("/home/myeong0609/PAL-XFEL_20210514/analysis/minuit/run_" + str(self.runNum) + "_double_exp_log.txt")
                        #     elif self.aniso_single_exp:
                        #         os.remove("/home/myeong0609/PAL-XFEL_20210514/analysis/minuit/run_" + str(self.runNum) + "_single_exp_log.txt")
                        #     elif self.heating_process:
                        #         os.remove("/home/myeong0609/PAL-XFEL_20210514/analysis/minuit/run_" + str(self.runNum) + "_heating_log.txt")

                    except:
                        pass
                    log = open(out_file_name, 'a')
                    if self.weighted_average:
                        run_num_info = "Status of anisotropic fitting analysis for run {0}&{1}".format(self.runList[0], self.runList[1]) + "\n"
                        timeStamp = "Time stamp of this run is " + self.now_time + "\n"

                    else:
                        run_num_info = "Status of anisotropic analysis for run " + str(self.runNum) + "\n"
                        timeStamp = "Time stamp of this run is " + self.now_time + "\n"
                    log.write(run_num_info)
                    log.write(timeStamp)
                    if self.chi2_save_list:
                        contents = "now try : " + str(calc_idx + 1) + ", minimum chi2 : " + str(np.min(self.chi2_save_list)) + "\n"
                        log.write(contents)
                    else:
                        contents = "now try : " + str(calc_idx + 1) + ", minimum chi2 : none" + "\n"
                        log.write(contents)
                else:
                    log = open(out_file_name, 'a')
                    if self.chi2_save_list:
                        contents = "now try : " + str(calc_idx + 1) + ", minimum chi2 : " + str(np.min(self.chi2_save_list)) + "\n"
                        log.write(contents)
                    else:
                        contents = "now try : " + str(calc_idx + 1) + ", minimum chi2 : none" + "\n"
                        log.write(contents)

        #
        now_delay_exp_data = self.input_raw_data # changed the codes as only saving what we want to fit
        common_time_delay = self.input_time_delay
        # convoluted_function = new_convolution_double_exp_to_final_fit(self.calculate_time_list, a1=1.08187147859314, t1=1.60110284481272, x0=-0.00622195396014221, y0=0.00362786725753946, a2=0.126356056181677, t2=11.5694739834075, FWHM=1.6)
        convoluted_function_1 = function_convolution_to_final_fit(self.calculate_time_list, a1=1.66213165878085, t1=1.00013954921794, x0=-0.731983295685843, y0=0, a2=0.136446943519626, t2=9.71403628455203, FWHM=2.21091553278284)
        old_convolution = function_convolution_to_final_fit(self.calculate_time_list,a1=1.49163476404242, t1=1.01191802268203,x0=-1.47143435476404, y0=-0.0034896646589035,a2=0.133983550686906,t2=7.20440096857635,FWHM=2.21091553278284)

        # convoluted_function = function_convolution_to_final_fit(self.calculate_time_list, a1=1.08187147859314, t1=1.6011  0284481272, x0=-0.00622195396014221, y0=0.00362786725753946, a2=0.126356056181677, t2=11.5694739834075, FWHM=1.6)
        common_time_delay = common_time_delay * 1E12
        # plt.plot(common_time_delay, now_delay_exp_data, label="run 9")
        # t_same = np.round(np.arange(-1, 3.001, 5E-3), 3)
        plt.plot(self.calculate_time_list, old_convolution, label="old convolution function")
        plt.plot(self.calculate_time_list, convoluted_function_1, label="original convolution function")
        #plt.plot(self.calculate_time_list, temp, label="without convolution")
        #plt.ylim(0,1.5)
        #plt.plot(self.calculate_time_list, convoluted_function, label="test fit", color='r')
        plt.legend()
        plt.show()
        # calculate_time_delay = self.calculate_time_list
        # data_yerr = 1
        # if self.runNum == 30 or self.runNum == 34:
        #     data_yerr = []
        #     for data_idx in range(len(self.input_raw_data)):
        #
        #         if 6 <= data_idx < 20:
        #             data_yerr.append(0.5)
        #         elif data_idx < 6:
        #             data_yerr.append(1.0)
        #         else:
        #             data_yerr.append(1.0)
        # elif self.runNum == 68 or self.runNum == 69:
        #     data_yerr = []
        #     for data_idx in range(len(self.input_raw_data)):
        #         if 7 < data_idx < 36:
        #             data_yerr.append(0.5)
        #         elif data_idx <= 7:
        #             data_yerr.append(1.0)
        #         else:
        #             data_yerr.append(1.0)
        #
        # # data_yerr = [1,1,1,1,1,0.01,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1] # for 46 pts
        # if random_initialization :
        #
        #     try:
        #         previous_result = np.load(self.previous_npy_load_path + ".npy")
        #         self.chi2_save_list.append(previous_result[0])
        #         self.min_validity_parameter_val_list.append(previous_result[1:])
        #     except:
        #         pass
        #
        #     if self.heating_process:
        #         num_of_parameter = 3
        #     elif self.anisotropy_process:
        #         if self.aniso_double_exp:
        #             num_of_parameter = 7
        #         elif self.aniso_single_exp :
        #             num_of_parameter = 5
        #         elif self.aniso_stretched_exp:
        #             num_of_parameter = 6
        #
        #     parameter_random_val_list = random_initialize(num_of_parameter, num_of_random_initialization)
        #
        #     for calc_idx in range(num_of_random_initialization):
        #
        #         if self.anisotropy_process:
        #             if self.aniso_double_exp:
        #                 a1_init = parameter_random_val_list[0][calc_idx]
        #                 t1_init = parameter_random_val_list[1][calc_idx]
        #                 x0_init = parameter_random_val_list[2][calc_idx]
        #                 y0_init = parameter_random_val_list[3][calc_idx]
        #                 a2_init = parameter_random_val_list[4][calc_idx]
        #                 t2_init = parameter_random_val_list[5][calc_idx]
        #                 # FWHM_init = 2.210915532782842 # FWHM from RT data
        #                 FWHM_init = parameter_random_val_list[6][calc_idx] #FWHM by random initialization
        #                 # FWHM_init = 1.89922266777435 # from run 9
        #                 # FWHM_init = 1.01556477029541 # from run 69s
        #                 # FWHM_init = 1.53210339379534
        #                 # FWHM_init = 1.58113033670129 #from run 71
        #
        #             elif self.aniso_single_exp:
        #                 a1_init = parameter_random_val_list[0][calc_idx]
        #                 y0_init = parameter_random_val_list[1][calc_idx]
        #                 t1_init = parameter_random_val_list[2][calc_idx] #1.6
        #                 x0_init = parameter_random_val_list[3][calc_idx]
        #                 FWHM_init = 2.210915532782842 # FWHM from RT data
        #                 #FWHM_init = parameter_random_val_list[4][calc_idx] #FWHM by random initialization
        #                 # FWHM_init = 1.53210339379534 # FWHM from day 5
        #                 #FWHM_init = 1.05046930445132 # FWHM from day 6
        #                 #FWHM_init = 1.82665039950606 #fwhm from run56&57&58
        #             elif self.aniso_stretched_exp:
        #                 a_init = parameter_random_val_list[0][calc_idx]
        #                 t_init = parameter_random_val_list[1][calc_idx]
        #                 beta_init = 0.6 #parameter_random_val_list[4][calc_idx]
        #                 x0_init = parameter_random_val_list[3][calc_idx]
        #                 y0_init = parameter_random_val_list[4][calc_idx]
        #                 FWHM_init = 2.210915532782842 #parameter_random_val_list[5][calc_idx] #FWHM by random initialization
        #         elif self.heating_process:
        #             #y0_init = parameter_random_val_list[0][calc_idx]
        #             a2_init = parameter_random_val_list[0][calc_idx]
        #             t2_init = parameter_random_val_list[1][calc_idx]
        #             x0_init = self.x0_from_anisotropy#parameter_random_val_list[2][calc_idx]#self.x0_from_anisotropy#parameter_random_val_list[1][calc_idx]
        #             #FWHM_init = 2.210915532782842  # FWHM from RT data
        #         if self.anisotropy_process:
        #             if self.aniso_single_exp:
        #                 single_exp_least_square = LeastSquares(common_time_delay, now_delay_exp_data, data_yerr, single_exponential_funciton_convolution)
        #                 m = Minuit(single_exp_least_square, a1=a1_init, t1=t1_init, x0=x0_init, y0=y0_init, FWHM=FWHM_init)
        #                 m.errors = (1E-5, 1E-5, 1E-5, 1E-5, 1E-5)
        #                 m.fixed['FWHM'] = True #set t1 as free
        #
        #             elif self.aniso_double_exp:
        #                 biexponential_least_square = LeastSquares(common_time_delay, now_delay_exp_data, data_yerr,new_convolution_double_exp)
        #                 m = Minuit(biexponential_least_square, a1=a1_init, t1=t1_init, a2=a2_init, t2=t2_init, x0=x0_init, y0=y0_init, FWHM=FWHM_init)
        #                 m.errors = (1E-5, 1E-5, 1E-5, 1E-5, 1E-5, 1E-5, 1E-5)
        #             #m.fixed['x0','FWHM', 't1'] = True # set t1 as fixed
        #                 m.fixed['FWHM'] = True #TODO
        #                 #m.fixed['FWHM', 'a1', 'a2', 't1', 't2', 'x0', 'y0'] = True #TODO
        #             #m.fixed['x0','t1'] = True #set FWHM as free
        #             elif self.aniso_stretched_exp:
        #                 stretched_exp_least_square = LeastSquares(common_time_delay, now_delay_exp_data, data_yerr,stretched_exponential_function_convolution)
        #                 m = Minuit(stretched_exp_least_square, a=a_init, t=t_init, beta=beta_init, x0=x0_init, y0=y0_init, FWHM=FWHM_init)
        #                 m.fixed['FWHM'] = True
        #                 m.errors = (1E-5, 1E-5, 1E-5, 1E-5, 1E-5, 1E-5)
        #             m.limits = self.param_limit  #TODO change a2 limits after test
        #         elif self.heating_process:
        #             exponential_least_square = LeastSquares(common_time_delay, now_delay_exp_data, data_yerr, exp_heating)
        #             m = Minuit(exponential_least_square, a2=a2_init, t2=t2_init, x0=x0_init)
        #             m.errors = (1E-5, 1E-5, 1E-5)
        #             m.limits = self.param_limit
        #             m.fixed['x0'] = True
        #
        #         if self.temporary_mask:
        #             biexponential_least_square.mask = (common_time_delay >= 0.5E-12)
        #             m.migrad()
        #             m.fixed['x0'] = False
        #             biexponential_least_square.mask = None
        #             self.temporary_mask = False
        #             m.fixed['a2', 't2'] = True
        #         else:
        #             m.migrad()
        #             #m.fixed['x0'] = False
        #         m.simplex()
        #         m.migrad()
        #         m.hesse()
        #         if m.valid:
        #             if self.anisotropy_process:
        #                 if self.aniso_double_exp:
        #                     # if (m.values[1] <= m.values[5]) and (m.values[0] <= 10 * m.values[4]) and (m.values[0] > m.values[4]):
        #                     if (m.values[1] <= m.values[5]) and (0.1 * m.values[0] <= m.values[4] <= 0.5 * m.values[0]) and (m.values[0] > m.values[4]):
        #                         self.chi2_save_list.append(m.fval)
        #                         self.min_validity_parameter_val_list.append(np.array(m.values))
        #                     else :
        #                         continue
        #                 elif self.aniso_single_exp:
        #                     self.chi2_save_list.append(m.fval)
        #                     self.min_validity_parameter_val_list.append(np.array(m.values))
        #                 elif self.aniso_stretched_exp:
        #                     self.chi2_save_list.append(m.fval)
        #                     self.min_validity_parameter_val_list.append(np.array(m.values))
        #             elif self.heating_process:
        #                 self.chi2_save_list.append(m.fval)
        #                 self.min_validity_parameter_val_list.append(np.array(m.values))
        #         if num_of_random_initialization > 1000:
        #             if calc_idx % 100 == 99:
        #                 write_log_info(calc_idx)
        #                 if self.chi2_save_list:
        #                     print("now try :", calc_idx + 1, "time" + ", minimum chi2 :", np.min(self.chi2_save_list))
        #                 else:
        #                     print("now try :", calc_idx + 1, "time" + ", minimum chi2 : None")
        #         else:
        #             if calc_idx %5 == 4:
        #                 write_log_info(calc_idx)
        #                 if self.chi2_save_list:
        #                     print("now try :", calc_idx + 1, "time" + ", minimum chi2 :", np.min(self.chi2_save_list))
        #                 else:
        #                     print("now try :", calc_idx + 1, "time" + ", minimum chi2 : None")
        #     if self.chi2_save_list:
        #         best_fit_idx = np.argmin(np.array(self.chi2_save_list))
        #     else:
        #         print("No valid minimum!")
        #         return
        #     # TODO :  save random result
        #     save_data_list = self.min_validity_parameter_val_list[best_fit_idx]
        #     if self.anisotropy_process:
        #         if self.aniso_double_exp:
        #             save_fit_val = new_convolution_double_exp_to_final_fit(self.calculate_time_list, *save_data_list)
        #         elif self.aniso_single_exp:
        #             save_fit_val = single_exponential_function_convolution_to_final_fit(self.calculate_time_list, *save_data_list)
        #         elif self.aniso_stretched_exp:
        #             save_fit_val = stretched_exponential_function_convolution_to_final_fit(self.calculate_time_list, *save_data_list)
        #     elif self.heating_process:
        #         save_fit_val = exp_heating(self.calculate_time_list, *save_data_list)
        #     save_fit_result_as_dat(save_data_list, save_fit_val, now_delay_exp_data, parameter_random_val_list, best_fit_idx)
        #
        # else:
        #     if self.anisotropy_process:
        #         a1_init = 1.5
        #         y0_init = 0
        #         t1_init = 1
        #         x0_init = -0.03 #0.030063757635259397  #for day3
        #         a2_init = 0.5
        #         t2_init = 8
        #         FWHM_init = 2.210915532782842  # 1.9
        #         biexponential_least_square = LeastSquares(common_time_delay, now_delay_exp_data, data_yerr, double_exponential_function_convolution)
        #         m = Minuit(biexponential_least_square, a1=a1_init, t1=t1_init, a2=a2_init, t2=t2_init, x0=x0_init, y0=y0_init, FWHM=FWHM_init)
        #         m.errors = (1E-5, 1E-5, 1E-5, 1E-5, 1E-5, 1E-5, 1E-5)
        #         m.limits = [(0, 3), (0, 3), (-1, 1), (0, 5), (0, 3), (6, 12), (0, 2.5)]
        #         #m.limits = [(0, None), (0, None), (None, None), (0, None), (0, None), (0, None), (0, None)]
        #         m.fixed['x0', 'FWHM'] = True
        #     elif self.heating_process:
        #         a2_init = -10
        #         t2_init = 13
        #         x0_init = 0.030063757635259397
        #         #y0_init = -1
        #         exponential_least_square = LeastSquares(common_time_delay, now_delay_exp_data, data_yerr, exp_heating)
        #         m = Minuit(exponential_least_square, a2=a2_init, t2=t2_init, x0=x0_init)
        #         m.errors = (1E-5, 1E-5, 1E-5)
        #         m.limits = [(None, 0), (10, 15), (None, None)]
        #         #m.fixed['x0'] = True
        #     # m2 = Minuit(biexponential_least_square, a1=a1_init, t1=t1_init, a2=a2_init, t2=t2_init, x0=x0_init, y0=y0_init,FWHM=FWHM_init)
        #     # m2.errors = (1E-5, 1E-5, 1E-5, 1E-5, 1E-5, 1E-5, 1E-5)
        #     # m2.limits = [(1, 2), (1, 3), (-1, 1), (0, 5), (0, 3), (6, 12),(0, 2.5)]
        #     # m2.fixed['t1'] = True
        #     print(m.params)
        #     #plt.plot(common_time_delay, function_convolution(common_time_delay, *m.values), label="simplex-fit", color='y')
        #     if self.temporary_mask:
        #         biexponential_least_square.mask = (common_time_delay <= 0.5E-12)
        #         m.simplex()
        #         m.migrad()
        #         m.fixed['x0'] = False
        #         biexponential_least_square.mask = None
        #         self.temporary_mask = False
        #         m.fixed['a1'] = True
        #         m.simplex()
        #     else:
        #         m.simplex()
        #         m.migrad()
        #         m.fixed['x0'] = False
        #     m.simplex()
        #     m.migrad()
        #     m.hesse()
        #     #m2.migrad
        #     #m2.fixed['t1', 'FWHM'] = True
        #     #m2.migrad()
        #     #m2.hesse()
        #     print(m.params)
        #     print(m.values)
        #     # print(m2.params)
        #     # print(m2.values)
        #     plt.plot(common_time_delay, now_delay_exp_data, label="data")
        #     if self.anisotropy_process:
        #         plt.plot(self.calculate_time_list, function_convolution_to_final_fit(self.calculate_time_list, *m.values), label="simplex->migrad fit", color='r')
        #     elif self.heating_process:
        #         plt.plot(self.calculate_time_list, exp_heating(self.calculate_time_list, *m.values), label="simplex->migrad fit", color='r')
        #     #plt.plot(self.calculate_time_list, function_convolution_to_final_fit(self.calculate_time_list, *m2.values), label="migrad only fit", color='black')
        #
        #     # display legend with some fit info
        #     fit_info = [
        #         f"$\\chi^2$ / $n_\\mathrm{{dof}}$ = {m.fval:.1f} / {len(common_time_delay) - m.nfit}",
        #     ]
        #     for p, v, e in zip(m.parameters, m.values, m.errors):
        #         fit_info.append(f"{p} = ${v:.3f} \\pm {e:.3f}$")
        #
        #     plt.legend(title="\n".join(fit_info))
        #     # plt.ylim((-50, 50))
        #     #plt.axvline(x=0, color='b')
        #     plt.show()
        #
        #     if m.valid:
        #         self.chi2_save_list.append(m.fval)
        #         temp_parameter_val_list = []
        #         if self.heating_process:
        #             num_of_parameter = 3
        #         elif self.anisotropy_process:
        #             num_of_parameter = 7
        #         for para_idx in range(num_of_parameter + 1):
        #             if para_idx == 0:
        #                 temp_parameter_val_list.append(m.fval)
        #             else:
        #                 temp_parameter_val_list.append(m.values[para_idx - 1])
        #         self.min_validity_parameter_val_list = (np.array(temp_parameter_val_list))
        #     else:
        #         print("No valid minimum!")
        #         return
        #
        #     if self.heating_process:
        #         save_fit_val = exp_heating(self.calculate_time_list, self.min_validity_parameter_val_list[1], self.min_validity_parameter_val_list[2], self.min_validity_parameter_val_list[3])#, self.min_validity_parameter_val_list[4])
        #
        #     elif self.anisotropy_process:
        #         save_fit_val = function_convolution_to_final_fit(self.calculate_time_list, self.min_validity_parameter_val_list[1], self.min_validity_parameter_val_list[2], self.min_validity_parameter_val_list[3], self.min_validity_parameter_val_list[4], self.min_validity_parameter_val_list[5], self.min_validity_parameter_val_list[6], self.min_validity_parameter_val_list[7])
        #     parameter_random_val_list = []
        #     calc_idx = 0
        #     save_fit_result_as_dat(self.min_validity_parameter_val_list, save_fit_val, now_delay_exp_data, parameter_random_val_list, calc_idx)

