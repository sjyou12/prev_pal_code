# from palxfel_scatter.diff_pair_1dcurve.PulseDataLight import PulseDataLight
from palxfel_scatter.diff_pair_1dcurve.Tth2qConvert import Tth2qConvert
import numpy as np
import math
import scipy.fft as fft
import scipy.stats as stats
import matplotlib.pyplot as plt
import os
import h5py as h5
import re
from codes.PDF.ff_class import Atom

class pulse_info:
    is_normalized = False
    norm_range_sum = 0

    def __init__(self, int_val, data_key, norm_start_idx, norm_end_idx):
        self.intensity_val = int_val
        self.key = data_key
        self.norm_start_idx = norm_start_idx
        self.norm_end_idx = norm_end_idx
        self.norm_range_sum = sum(self.intensity_val[self.norm_start_idx:self.norm_end_idx])

    def norm_given_range(self):
        self.is_normalized = True
        try:
            self.intensity_val = list((np.array(self.intensity_val) / self.norm_range_sum) * 1E7)
            #return self.intensity_val
        except:
            print("normalization error ")
            print(self.intensity_val)
            print(self.norm_range_sum)
            self.is_normalized = False

class pdfConverter:
    FT_start_q_idx = None
    FT_end_q_idx = None
    q_val = []
    twotheta_file_name = None
    twotheta_val = []
    now_run_delay_num = 0
    intensity_file_names = []
    number_density = 33.3679 * 1E-3 # number density per A^3
    x_ray_energy = 20 # todo
    norm_q_range_start_idx = None
    norm_q_range_after_idx = None

    tth_to_q_cvt = None

    def __init__(self, dat_common_root, run_num):
        self.FileCommonRoot = dat_common_root
        self.each_run_file_dir = []
        self.run_num = run_num
        self.intensity_files = []
        self.each_delay_int_list = []
        self.ref_q_val = []
        self.ref_structure_factor = []
        self.ref_r_arr = []
        self.ref_FT_results = []
        self.q_min_idx = None
        self.r_arr = []
        self.FT_result = []
        self.q_max_arr = []

        self.pair_info = []

        self.laser_on_int_list = []
        self.laser_off_int_list = []
        self.laser_on_key_list = []
        self.laser_off_key_list = []
        self.tot_diff_curve = []

        self.max_q_dependent_diff_PDF = []
        self.each_delay_diff_PDF = []
        self.ref_data = []

        self.O_ff = []
        self.H_ff = []
        self.I_at = [] # sum of square of each form factor value (check SI of Nature Chemistry 11, 504-509 (2019))

    def norm_range_q_idx_calc(self):
        # set idx range for normalization
        NormStartQ = 1.5
        NormEndQ = 3.5
        # TODO : make new method for set norm / pairing range
        if len(self.q_val) == 0:
            print("no q value now!")
        self.norm_q_range_start_idx = int(np.where(self.q_val >= NormStartQ)[0][0])
        # this index is not included in water q range!!!
        self.norm_q_range_after_idx = int(np.where(self.q_val > NormEndQ)[0][0])

        print("( normalization] {0} is in {1}th index ~ {2} is in {3}th index )".format(self.q_val[self.norm_q_range_start_idx],
                                                                         self.norm_q_range_start_idx,
                                                                         self.q_val[self.norm_q_range_after_idx],
                                                                         self.norm_q_range_after_idx))

    def read_twotheta_value(self):
        print("read tth value from file")

        self.tth_to_q_cvt = Tth2qConvert(self.x_ray_energy)

        # now_tth_path = self.FileCommonRoot + "run" + str(self.runList[0]) + "/"
        now_tth_path = self.each_run_file_dir[0] + self.twotheta_file_name + ".h5"
        twotheta_file = h5.File(now_tth_path, 'r')
        twotheta_keys = list(twotheta_file.keys())
        # print(len(twotheta_keys), "key values, head : ", twotheta_keys[0], "tail : ", twotheta_keys[-1])

        now_tth_obj_name = twotheta_keys[0]
        self.twotheta_val = np.array(twotheta_file[now_tth_obj_name])
        print("read fixed 2theta value end. shape of value : ", self.twotheta_val.shape)
        self.q_val = np.array(self.tth_to_q_cvt.tth_to_q(self.twotheta_val))
        print("now q values : from ", self.q_val[0], "to", self.q_val[-1])

    def linear_regrssion(self):
        left_part_q_val = (0.8, 1.2)
        # left_part_q_val = 1.2
        left_q_mask = (self.q_val >= left_part_q_val[0]) & (self.q_val <= left_part_q_val[1])
        left_q_val = self.q_val[left_q_mask]
        right_part_start_q_val =

    def load_1d(self):

        self.pair_info = np.load("../../results/anisotropy/run" + str(self.run_num) + "_pairinfo.npy", allow_pickle=True)

        dat_input_path = self.FileCommonRoot + "run_{0:05d}_DIR/eh1rayMXAI_int/".format(self.run_num)
        delay_names = os.listdir(dat_input_path)
        print("now delay num : ", len(delay_names))
        self.now_run_delay_num = len(delay_names)
        self.now_run_delay_num = 1

        for idx in range(self.now_run_delay_num):
            temp_name_int = "eh1rayMXAI_int/001_001_%03d" % (idx + 1)
            self.intensity_file_names.append(temp_name_int)
        # open only one twotheta data (since all value is same)
        self.twotheta_file_name = 'eh1rayMXAI_tth/001_001_001'

        now_file_dir = self.FileCommonRoot + "run_{0:05d}_DIR/".format(self.run_num)
        self.each_run_file_dir.append(now_file_dir)
        temp_int_files = []
        for idx in range(self.now_run_delay_num):
            if idx != 0:
                continue
            now_int_path = now_file_dir + self.intensity_file_names[idx] + ".h5"
            temp_int_file = h5.File(now_int_path, 'r')
            temp_int_files.append(temp_int_file)
        self.intensity_files = temp_int_files
        self.read_twotheta_value()
        self.norm_range_q_idx_calc()

        self.check_laser_on_off()
        self.make_on_off_pairing_1d()
        self.calc_ff()

    def make_on_off_pairing_1d(self):
        for delay_idx in range(self.now_run_delay_num):
            on_key_list = self.laser_on_key_list[delay_idx]
            off_key_list = self.laser_off_key_list[delay_idx]
            on_curve_list = self.laser_on_int_list[delay_idx]
            off_curve_list = self.laser_off_int_list[delay_idx]

            each_delay_diff = []
            for idx, each_pair_data in enumerate(self.pair_info[delay_idx]):
                on_key = each_pair_data[0]
                off_key = each_pair_data[1]
                try:
                    on_img_idx = (np.where(on_key_list == on_key))[0][0]
                    off_img_idx = (np.where(off_key_list == off_key))[0][0]
                except:
                    print("error key : {}(on) / {}(off)".format(on_key, off_key))
                on_curve = np.array(on_curve_list[on_img_idx].intensity_val)
                off_curve = np.array(off_curve_list[off_img_idx].intensity_val)
                diff_curve = on_curve - off_curve
                each_delay_diff.append(diff_curve)
            self.tot_diff_curve.append(np.average(each_delay_diff, axis=0))

    def load_ref_data(self, Q_max_arr, Q_min):
        dat_input_root = "/data/exp_data/myeong0609/gromacs/PDF/input_Iq.csv"
        self.ref_data = np.loadtxt(dat_input_root, skiprows=1, delimiter=',')
        self.ref_q_val = self.ref_data[:, 0]
        self.ref_structure_factor = self.ref_data[:, 5]

        a = 2.8
        b = 0.5
        total_FT_result = []
        total_r_arr = []

        if Q_min == 0:
            q_min_idx = 1
        else:
            q_min_idx = np.where(self.ref_q_val == Q_min)[0][0]

        for max_q in Q_max_arr:
            q_max = max_q
            points_space = np.round(self.ref_q_val[1] - self.ref_q_val[0], 10)  # gap of original space
            q_max_idx = np.where(self.ref_q_val == q_max)[0][0]
            sliced_q_val = self.ref_q_val[q_min_idx:q_max_idx + 1]
            r_arr = fft.fftfreq(len(sliced_q_val), points_space)[:len(sliced_q_val) // 2]
            total_r_arr.append(r_arr)
            modified_fxn = []
            delta_r_arr = (math.pi / q_max) * (1 - np.exp(-(np.abs(r_arr - a) / b)))
            for delta_r in delta_r_arr:
                modified_fxn.append(np.sin(sliced_q_val * delta_r) / (sliced_q_val * delta_r))
            FT_result = []
            for r_idx, r in enumerate(r_arr):
                temp_sum_arr = []
                for q_idx, q in enumerate(sliced_q_val):
                    temp_sum_arr.append(1 + 1 / (2 * math.pi * self.number_density * r) * (
                                modified_fxn[r_idx][q_idx] * q * (self.ref_structure_factor[q_idx] - 1)) * (np.sin(q * r)))
                FT_result.append(np.sum(temp_sum_arr))
            total_FT_result.append(FT_result)

        self.ref_FT_results = total_FT_result
        self.ref_r_arr = total_r_arr

    def check_laser_on_off(self):
        for delay_idx, each_delay_int in enumerate(self.intensity_files):
            now_delay_keys = list(each_delay_int.keys())
            temp_delay_on_int_list = []
            temp_delay_on_key_list = []
            temp_delay_off_int_list = []
            temp_delay_off_key_list = []

            each_delay_pass_key = self.pair_info[delay_idx].flatten()
            for each_key in now_delay_keys:
                if each_key not in each_delay_pass_key:
                    continue
                now_int_val = np.array(each_delay_int[each_key])
                now_pulseID = re.findall("(.*)\.(.*)_(.*)", each_key)[0][2]
                now_pulseID = int(now_pulseID)
                now_int_obj = pulse_info(now_int_val, each_key, self.norm_q_range_start_idx, self.norm_q_range_after_idx)
                if not now_int_obj.is_normalized:
                    now_int_obj.norm_given_range()
                if now_pulseID % 24 == 0:
                    temp_delay_on_int_list.append(now_int_obj)
                    temp_delay_on_key_list.append(each_key)
                else:
                    temp_delay_off_int_list.append(now_int_obj)
                    temp_delay_off_key_list.append(each_key)

            self.laser_on_int_list.append(np.array(temp_delay_on_int_list))
            self.laser_on_key_list.append(np.array(temp_delay_on_key_list))
            self.laser_off_int_list.append(np.array(temp_delay_off_int_list))
            self.laser_off_key_list.append(np.array(temp_delay_off_key_list))

    def calc_ff(self):
        oxygen = Atom('O')
        self.O_ff = oxygen.form_factor(oxygen, self.q_val)
        hydrogen = Atom('H')
        self.H_ff = hydrogen.form_factor(hydrogen, self.q_val)

        square_O_ff = self.O_ff**2
        square_H_ff = self.H_ff**2
        self.I_at = square_O_ff + 2*square_H_ff

    def calc_FT(self):
        a = 2.8
        b = 0.5
        total_FT_result = []
        total_r_arr = []

        print("q range for FT : {0} ~ {1}".format(self.q_val[0], self.q_val[-1]))

        self.calc_ff()

        self.q_max_arr = [11.5, self.q_val[-1]]
        self.q_min_idx = 0
        temp_int = []
        for idx, each_int in enumerate(self.laser_off_int_list[0]):
            temp_int.append(each_int.intensity_val)
        off_avg_int = np.average(temp_int, axis=0)
        data = off_avg_int/self.I_at
        for q_max in self.q_max_arr:
            points_space = np.round(self.q_val[1] - self.q_val[0], 10)  # gap of original space
            q_max_idx = np.where(self.q_val >= q_max)[0][0]
            if q_max_idx == len(self.q_val)-1:
                sliced_q_val = self.q_val[self.q_min_idx:]
            else:
                sliced_q_val = self.q_val[self.q_min_idx:q_max_idx + 1]
            r_arr = fft.fftfreq(len(sliced_q_val), points_space)[:len(sliced_q_val) // 2]
            total_r_arr.append(r_arr)
            modified_fxn = []
            delta_r_arr = (math.pi / q_max) * (1 - np.exp(-(np.abs(r_arr - a) / b)))
            temp_FT = []
            for delta_r in delta_r_arr:
                modified_fxn.append(np.sin(sliced_q_val * delta_r) / (sliced_q_val * delta_r))
            # for each_delay_int in self.laser_on_int_list:
            # for each_delay_int in self.tot_diff_curve:
            #     for r_idx, r in enumerate(r_arr):
            #         temp_sum_arr = []
            #         for q_idx, q in enumerate(sliced_q_val):
            #             temp_sum_arr.append(1 + 1 / (2 * math.pi * self.number_density * r) * (modified_fxn[r_idx][q_idx] * q * (each_delay_int[q_idx] - 1)) * (np.sin(q * r)))
            #         temp_FT.append(np.sum(temp_sum_arr))
            #     total_FT_result.append(temp_FT)
            for r_idx, r in enumerate(r_arr):
                temp_sum_arr = []
                if r_idx == 0:
                    temp_FT.append(0)
                else:
                    for q_idx, q in enumerate(sliced_q_val):
                        temp_sum_arr.append(1 + 1 / (2 * math.pi * self.number_density * r) * (modified_fxn[r_idx][q_idx] * q * (data[q_idx] - 1)) * (np.sin(q * r)))
                    temp_FT.append(np.sum(temp_sum_arr))
            total_FT_result.append(temp_FT)

        self.r_arr = total_r_arr
        self.FT_result = total_FT_result

    def FT_science_paper(self):

        q_max = 4.5
        points_space = np.round(self.q_val[1] - self.q_val[0], 10)  # gap of original space
        q_max_idx = np.where(self.q_val > q_max)[0][0]
        if q_max_idx == len(self.q_val) - 1:
            sliced_q_val = self.q_val[self.q_min_idx:]
        else:
            sliced_q_val = self.q_val[self.q_min_idx:q_max_idx + 1]
        r_arr = fft.fftfreq(len(sliced_q_val), points_space)[:len(sliced_q_val) // 2]
        self.r_arr = (r_arr)

        modified_scat = (self.tot_diff_curve/self.I_at)*self.q_val

        # for idx, q_max in enumerate(self.q_max_arr):
        for each_delay_modified_scat in modified_scat:
            temp_sum = []
            for r in self.r_arr:
                # temp_val = []
                # for q_idx, q in enumerate(self.q_val):
                temp_sin_val = np.sin(self.q_val * r)
                temp_exp_val = np.exp(-0.02 * self.q_val**2)
                temp_sum.append(np.sum(each_delay_modified_scat * temp_sin_val * temp_exp_val))
            self.each_delay_diff_PDF.append(temp_sum)
        self.max_q_dependent_diff_PDF.append(self.each_delay_diff_PDF)

    def plt_each_PDF(self):
        for delay_idx, each_FT_result in enumerate(self.FT_result):
            one_graph_plot = 1
            # plt.plot(self.r_arr[max_q_idx], each_FT_result / each_FT_result[-1],label="Q max = {}".format(self.q_max_arr[max_q_idx]))
            plt.plot(self.r_arr[delay_idx], each_FT_result / each_FT_result[-1])#,label="{} th delay".format(delay_idx + 1))
            if delay_idx % one_graph_plot == (one_graph_plot-1):
                plt.title("Pair distribution function with Q max {}".format(self.q_max_arr[delay_idx]))
                plt.xlim(0, 6)
                plt.legend()
                plt.show()

        # plt.title("Pair distribution function with Q min {}".format(self.q_val[0]))
        # plt.xlim(0, 6)
        # plt.legend()
        # plt.plot(self.ref_r_arr[-1], self.ref_FT_results[-1] / self.ref_FT_results[-1][-1], label="Q max = 26")
        # plt.show()

        # for max_q_idx, each_FT_result in enumerate(self.FT_result):
        #     plt.plot(self.r_arr[max_q_idx], each_FT_result / each_FT_result[-1],
        #              label="Q max = {}".format(self.q_max_arr[max_q_idx]))
        # plt.title("Pair distribution function with Q min {}".format(self.q_val[0]))
        # # plt.xlim(0, 6)
        # plt.legend()
        # plt.show()

    def plot_contour(self):
        # plot_limit = np.max(np.abs(self.each_delay_diff_PDF))
        plot_limit = np.max(np.abs(self.FT_result[0]))
        plot_levels = np.arange(-plot_limit, plot_limit + plot_limit * 2 / 50, plot_limit * 2 / 50)

        now_delays = np.arange(-1, 3.1, 0.1)

        plt.contour(self.r_arr[0:45], now_delays,np.array(self.FT_result[0])[:, 0:45], cmap="seismic", vmin=-plot_limit, vmax=plot_limit)
        cntr = plt.contourf(self.r_arr[0:45], now_delays , np.array(self.FT_result[0])[:, 0:45],levels=plot_levels, cmap="seismic")
        plt.title("difference PDF")
        plt.xlabel("r (Å)")
        plt.ylabel("time (ps)")
        plt.colorbar(cntr)
        plt.show()

        # plt.contour(self.q_val, range(self.now_run_delay_num),self.exp_qdS[self.plot_start_t_idx:self.plot_end_t_idx, :], cmap="seismic", vmin=-plot_limit, vmax=plot_limit)

