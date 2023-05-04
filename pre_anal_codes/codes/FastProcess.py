import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
import configparser
import math
import re
import ast

from codes.PulseData import PulseData, WaterType, IceType
from codes.old_SVDCalc import SVDCalc
from codes.DataClasses import tth_to_q, find_neareast_pair

FileCommonRoot = "/xfel/ffs/dat/scan/"

config = configparser.ConfigParser()
config.read('anal_cond.cfg')

# read constant from config file
IcePeakStartQ = float(config['DEFAULT']['IcePeakStartQ'])
IcePeakEndQ = float(config['DEFAULT']['IcePeakEndQ'])
WaterPeakStartQ = float(config['DEFAULT']['WaterPeakStartQ'])
WaterPeakEndQ = float(config['DEFAULT']['WaterPeakEndQ'])
LowerBoundOfIce = float(config['DEFAULT']['LowerBoundOfIce'])
UpperBoundOfNotIce = float(config['DEFAULT']['UpperBoundOfNotIce'])
LowerBoundOfWater = float(config['DEFAULT']['LowerBoundOfWater'])
UpperBoundOfNotWater = float(config['DEFAULT']['UpperBoundOfNotWater'])
NormStartQ = float(config['DEFAULT']['NormStartQ'])
NormEndQ = float(config['DEFAULT']['NormEndQ'])
PlotStartIdx = int(config['DEFAULT']['PlotStartIdx'])
PlotEndIdx = int(config['DEFAULT']['PlotEndIdx'])
WaterOutlierLowerBound = float(config['DEFAULT']['WaterOutlierLowerBound'])

svd_start_idx = 93
svd_end_idx = -180

# common constant for calculating q values
XrayEnergy = 14  # keV unit
XrayWavelength = 12.3984 / XrayEnergy  # Angstrom unit (10^-10 m)
QCoefficient = 4 * math.pi / XrayWavelength
NormFactor = 100000  # Normalization factor (sum of all integration)


class FastProcessDataSet:
    # common constant for main file
    NUM_MOTOR_VAL = 3
    NUM_PULSE = 30
    NUM_THETA_BIN = 1000

    file_family_name = None
    file_index = None
    file_main_name = None
    file_dir = None

    # if convinced two theta is all equal, remove file member and only save one value index
    twotheta_file_name = None  # need to remove
    twotheta_file = None  # need to remove
    twotheta_keys = None  # need to remove

    num_of_delay = 3  # default value, number of motor value
    num_data_per_each_delay = 0
    delay_idx = ['001', '002', '003']
    out_file_pointers = {"each-delay-diff": None}

    def __init__(self):
        self.intensity_file_names = []
        self.intenisty_files = []
        self.intensity_keys_list = []
        self.intensity_data_list = []

        # if convinced two theta is all equal, remove file member and only save one value index
        self.twotheta_val = []
        self.q_val = []

        self.int_pair_diff_save = []
        self.neg_delay_avg_diff_list = []

        self.I0_file_names = []
        self.I0_files = []
        self.cutted_after_data_len = []
        self.neg_diff_data_len = []

    def __del__(self):
        self.close_all_files()

    def close_all_files(self):
        if self.intenisty_files:
            print(len(self.intenisty_files), "int files to close")
            for each_int_file in self.intenisty_files:
                each_int_file.close()
            self.intenisty_files = []
            print("close intensity files")
        if self.twotheta_file is not None:
            self.twotheta_file.close()
            self.twotheta_file = None
            print("close twotheta files")
        if self.I0_files:
            print(len(self.I0_files), "int files to close")
            for each_file in self.I0_files:
                each_file.close()
            self.I0_files = []
            print("close I0 files")

    def set_file_name(self, now_family_name, now_index):
        global FileCommonRoot
        self.file_family_name = now_family_name
        self.file_index = now_index
        self.file_main_name = now_family_name + "_" + now_index
        self.file_dir = FileCommonRoot + self.file_main_name + "_DIR/"

    def set_delay_num(self, new_delay_num, new_delay_idx):
        self.num_of_delay = new_delay_num
        self.delay_idx = new_delay_idx
        for idx in range(new_delay_num):
            self.int_pair_diff_save.append([])

    def load_each_file(self):
        for idx in self.delay_idx:
            temp_name_int = 'eh2rayMXAI_int/001_' + idx
            self.intensity_file_names.append(temp_name_int)
            temp_name_I0 = 'eh1qbpm1_totalsum/001_' + idx
            self.I0_file_names.append(temp_name_I0)
        # open only one twotheta data (since all value is same)
        self.twotheta_file_name = 'eh2rayMXAI_tth/001_001_001'

        for idx in range(self.num_of_delay):
            now_int_path = self.file_dir + self.intensity_file_names[idx] + ".h5"
            now_I0_path = self.file_dir + self.I0_file_names[idx] + ".h5"
            try:
                temp_int_file = h5.File(now_int_path, 'r')
                self.intenisty_files.append(temp_int_file)
                self.intensity_keys_list.append(list(temp_int_file.keys()))
            except:
                print("file open error in ", now_int_path)
            try:
                temp_I0_file = h5.File(now_I0_path, 'r')
                self.I0_files.append(temp_I0_file)
            except:
                print("file open error in ", now_I0_path)
        now_tth_path = self.file_dir + self.twotheta_file_name + ".h5"
        self.twotheta_file = h5.File(now_tth_path, 'r')
        self.twotheta_keys = list(self.twotheta_file.keys())
        print(self.twotheta_keys)
        self.read_twotheta_value()
        # self.read_all_intensity_value()

    def read_twotheta_value(self):
        now_tth_obj_name = self.twotheta_keys[0]
        self.twotheta_val = np.array(self.twotheta_file[now_tth_obj_name])
        print("read fixed 2theta value end")
        print(self.twotheta_val.shape)
        self.q_val = tth_to_q(self.twotheta_val)
        print("now q values : ", self.q_val)

    def read_all_intensity_value(self, norm_with_I0=False):
        '''
        auxilary function for read all value at same time
        if you want to read necessary value only, do not use this function.
        it will take huge memory space.
        '''
        laser_off_cnt = 0
        for motor_idx in range(self.num_of_delay):
            now_motor_int_val_list = []
            for idx_in_file in range(len(self.intensity_keys_list[motor_idx])):
                # print(self.intensity_file_names[motor_idx])
                now_int_obj_name = self.intensity_keys_list[motor_idx][idx_in_file]
                if norm_with_I0:
                    # try:
                    now_I0_val = float(self.I0_files[motor_idx][now_int_obj_name][()])
                    # except:
                    #     print("error in delay", motor_idx, "obj name", now_int_obj_name)
                    now_int_val = PulseData(np.array(self.intenisty_files[motor_idx][now_int_obj_name]),
                                            I0_val=now_I0_val)
                else:
                    now_int_val = PulseData(np.array(self.intenisty_files[motor_idx][now_int_obj_name]))
                now_pulseID = re.findall("(.*)\.(.*)_(.*)", now_int_obj_name)[0][2]
                now_pulseID = int(now_pulseID)
                now_int_val.check_laser_onoff(now_pulseID)
                if not now_int_val.laser_is_on:
                    laser_off_cnt += 1
                    random_number = np.random.rand(1)
                    if (random_number[0] > 0.5) == 1:
                        now_int_val.neg_delay_laser_on = True
                    else:
                        now_int_val.neg_delay_laser_on = False

                now_motor_int_val_list.append(now_int_val)
            self.intensity_data_list.append(now_motor_int_val_list)
        print("read intensity value end")
        print(len(self.intensity_data_list))
        self.num_data_per_each_delay = len(self.intensity_data_list[0])

    @staticmethod
    def plot_sum_for_criteria(data, graph_title, v_line_1=0.0, v_line_2=0.0, v_line_3=0.0):
        """
        plot ice sum / water sum tendency for decide criteria of water/ice data separation
        """
        data = np.array(data)
        data = data[data>0]
        plt.hist(data, bins=200, log=True)
        # plt.hist(data, bins=200)
        plt.title(graph_title)
        plt.xlabel("integration value")
        plt.ylabel("frequency")
        if v_line_1 != 0.0:
            plt.axvline(x=v_line_1, color='r')
        if v_line_2 != 0.0:
            plt.axvline(x=v_line_2, color='r')
        if v_line_3 != 0.0:
            plt.axvline(x=v_line_3, color='g')
        plt.show()

    def plot_avg_each_class(self, data_list, show_ice_plot=False):
        ice_counter = 0
        gray_ice_counter = 0
        not_ice_counter = 0
        water_counter = 0
        gray_water_counter = 0
        not_water_counter = 0
        ice_sum = np.zeros_like(data_list[0].intensity_val)
        gray_ice_sum = np.zeros_like(data_list[0].intensity_val)
        not_ice_sum = np.zeros_like(data_list[0].intensity_val)
        water_sum = np.zeros_like(data_list[0].intensity_val)
        gray_water_sum = np.zeros_like(data_list[0].intensity_val)
        not_water_sum = np.zeros_like(data_list[0].intensity_val)
        for each_data in data_list:
            if each_data.ice_type == IceType.ICE:
                ice_counter += 1
                ice_sum += np.array(each_data.intensity_val)
            elif each_data.ice_type == IceType.GRAY_ICE:
                gray_ice_counter += 1
                gray_ice_sum += np.array(each_data.intensity_val)
            else:
                not_ice_counter += 1
                not_ice_sum += np.array(each_data.intensity_val)

            if each_data.water_type == WaterType.WATER:
                water_counter += 1
                water_sum += np.array(each_data.intensity_val)
            elif each_data.water_type == WaterType.GRAY_WATER:
                gray_water_counter += 1
                gray_water_sum += np.array(each_data.intensity_val)
            elif each_data.water_type == WaterType.NOT_WATER:
                not_water_counter += 1
                not_water_sum += np.array(each_data.intensity_val)

        print("print statistic")
        print("ice :", ice_counter, "gray :", gray_ice_counter, "not ice : ", not_ice_counter)
        print("water :", water_counter, "gray :", gray_water_counter, "not water : ", not_water_counter)
        total_ice = ice_counter + gray_ice_counter + not_ice_counter
        total_water = water_counter + gray_water_counter + not_water_counter
        print("total number check (ice)", total_ice, "(water)", total_water)

        division_list = [(ice_sum, ice_counter), (gray_ice_sum, gray_ice_counter), (not_ice_sum, not_ice_counter),
                         (water_sum, water_counter), (gray_water_sum, gray_water_counter),
                         (not_water_sum, not_water_counter)]
        avg_list = []
        for now_sum, now_counter in division_list:
            try:
                now_avg = now_sum / now_counter
            except:
                now_avg = now_sum
            avg_list.append(now_avg)

        ice_avg = avg_list[0]
        gray_ice_avg = avg_list[1]
        not_ice_avg = avg_list[2]
        water_avg = avg_list[3]
        gray_water_avg = avg_list[4]
        not_water_avg = avg_list[5]

        if show_ice_plot:
            plt.plot(self.q_val, ice_avg, label="ice avg")
            plt.plot(self.q_val, gray_ice_avg, label="gray avg")
            plt.plot(self.q_val, not_ice_avg, label="not ice avg")
            plt.axvline(x=IcePeakStartQ, color='r')
            plt.axvline(x=IcePeakEndQ, color='r')
            plt.title('average intensity for each ice class')
            plt.xlabel('q value')
            plt.ylabel("average intensity")
            plt.legend()
            plt.show()

        plt.plot(self.q_val, water_avg, label="water avg")
        plt.plot(self.q_val, gray_water_avg, label="gray avg")
        plt.plot(self.q_val, not_water_avg, label="not water avg")
        plt.axvline(x=WaterPeakStartQ, color='r')
        plt.axvline(x=WaterPeakEndQ, color='r')
        plt.title('average intensity for each water class')
        plt.xlabel('q value')
        plt.ylabel("average intensity")
        plt.legend()
        plt.show()

    def decide_criteria(self, show_ice_plot=False):
        all_ice_sum_list = []
        all_water_sum_list = []
        for each_delay in range(self.num_of_delay):
            for data_idx in range(self.num_data_per_each_delay):
                now_ice_sum, now_water_sum = self.intensity_data_list[each_delay][data_idx].classify_data(self.q_val)
                all_ice_sum_list.append(now_ice_sum)
                all_water_sum_list.append(now_water_sum)

        self.plot_sum_for_criteria(all_water_sum_list, "water sum view", LowerBoundOfWater, UpperBoundOfNotWater,
                                   WaterOutlierLowerBound)
        if show_ice_plot:
            self.plot_sum_for_criteria(all_ice_sum_list, "ice sum view", LowerBoundOfIce, UpperBoundOfNotIce)

        concat_all_delay = []
        for each_delay in range(self.num_of_delay):
            concat_all_delay.extend(self.intensity_data_list[each_delay])
        self.plot_avg_each_class(concat_all_delay, show_ice_plot=show_ice_plot)

    @staticmethod
    def extract_laser_on_off_water_list(data_list):
        laser_on_water_idx = []
        laser_off_water_idx = []
        for data_idx, each_data in enumerate(data_list):
            if each_data.water_type == WaterType.WATER:
                if each_data.laser_is_on:
                    laser_on_water_idx.append(data_idx)
                else:
                    laser_off_water_idx.append(data_idx)

        print("laser on droplet : ", len(laser_on_water_idx), "laser off : ", len(laser_off_water_idx))
        return laser_on_water_idx, laser_off_water_idx

    @staticmethod
    def extract_neg_delay_laser_on_off_water_list(data_list, laser_off_idx):
        neg_laser_on_water_idx = []
        neg_laser_off_water_idx = []
        for each_off_idx in laser_off_idx:
            now_data = data_list[each_off_idx]
            if now_data.laser_is_on:
                print("error! laser is on!!")
            else:
                if now_data.water_type == WaterType.WATER:
                    if now_data.neg_delay_laser_on:
                        neg_laser_on_water_idx.append(each_off_idx)
                    else:
                        neg_laser_off_water_idx.append(each_off_idx)

        print("(neg delay) temporal laser on droplet : ", len(neg_laser_on_water_idx), "laser off : ",
              len(neg_laser_off_water_idx))
        return neg_laser_on_water_idx, neg_laser_off_water_idx

    @staticmethod
    def match_water_near_intensity_pair(data_list, laser_on_water_idx, laser_off_water_idx):
        compare_longer_one = None
        compare_shoter_one = None
        if len(laser_on_water_idx) > len(laser_off_water_idx):
            compare_longer_one = laser_on_water_idx
            compare_shoter_one = laser_off_water_idx
        else:
            compare_longer_one = laser_off_water_idx
            compare_shoter_one = laser_on_water_idx

        cmp_long_norm_sum = []
        cmp_short_norm_sum = []
        for each_idx in compare_longer_one:
            cmp_long_norm_sum.append(data_list[each_idx].pair_range_sum)
        for each_idx in compare_shoter_one:
            cmp_short_norm_sum.append(data_list[each_idx].pair_range_sum)

        neareast_int_pair = []
        for int_idx, each_int_sum in enumerate(cmp_long_norm_sum):
            most_similar_sum = find_neareast_pair(cmp_short_norm_sum, each_int_sum)
            sim_sum_idx = cmp_short_norm_sum.index(most_similar_sum)
            neareast_int_pair.append((compare_longer_one[int_idx], compare_shoter_one[sim_sum_idx]))

        print("pairing with nearest integrated intensity logic")
        return neareast_int_pair

    @staticmethod
    def match_water_near_idx_pair(data_list, laser_on_water_idx, laser_off_water_idx):
        compare_longer_one = None
        compare_shoter_one = None
        if len(laser_on_water_idx) > len(laser_off_water_idx):
            compare_longer_one = laser_on_water_idx
            compare_shoter_one = laser_off_water_idx
        else:
            compare_longer_one = laser_off_water_idx
            compare_shoter_one = laser_on_water_idx

        neareast_idx_pair = []
        for data_idx in compare_longer_one:
            nearest_idx = find_neareast_pair(compare_shoter_one, data_idx)
            neareast_idx_pair.append((data_idx, nearest_idx))

        print("pairing with nearest index (time order) logic")
        return neareast_idx_pair

    def plot_near_int_diff(self, data_list, near_int_pair, delay_idx, draw_each_graph=False,
                           each_avg_plot=True, plot_start_idx=0, plot_end_idx=-1):
        global svd_start_idx
        global svd_end_idx
        diff_int_arr = []
        for each_idx_a, each_idx_b in near_int_pair:
            data_a = data_list[each_idx_a]
            data_b = data_list[each_idx_b]

            laser_on_data = None
            laser_off_data = None
            if data_a.laser_is_on:
                laser_on_data = data_a
                laser_off_data = data_b
            else:
                laser_on_data = data_b
                laser_off_data = data_a

            diff_int = np.array(laser_on_data.intensity_val) - np.array(laser_off_data.intensity_val)
            self.int_pair_diff_save[delay_idx].append(diff_int[svd_start_idx:svd_end_idx])
            diff_int_arr.append(diff_int)

        print(len(diff_int_arr))
        if draw_each_graph:
            num_draw_front_diff = 9
            num_data_in_one_graph = 3
            for diff_idx in range(num_draw_front_diff):
                plt.plot(self.q_val[plot_start_idx:plot_end_idx],
                         diff_int_arr[diff_idx][plot_start_idx:plot_end_idx],
                         label=(str(diff_idx + 1) + 'th diff'))
                if ((diff_idx + 1) % num_data_in_one_graph) == 0:
                    plt.title("draw each diff - nearest intensity")
                    plt.xlabel('q value')
                    plt.ylabel("intensity")
                    plt.legend()
                    plt.show()
        try:
            avg_diff = np.average(diff_int_arr, axis=0)
            if each_avg_plot:
                plt.plot(self.q_val[plot_start_idx:plot_end_idx], avg_diff[plot_start_idx:plot_end_idx])
                plt.title("avg normalized diff - similar integration on/off pair (" + str(delay_idx) + "-th delay)")
                plt.xlabel('q value')
                plt.ylabel("intensity")
                plt.show()
        except:
            avg_diff = np.zeros_like(self.q_val)
            print("no pairing graph in delay", delay_idx)
        return avg_diff

    def plot_neg_delay_near_int_diff(self, data_list, near_int_pair, delay_idx, draw_each_graph=False,
                                     plot_start_idx=0, plot_end_idx=-1, plot=False):
        # global svd_start_idx
        # global svd_end_idx
        diff_int_arr = []
        for each_idx_a, each_idx_b in near_int_pair:
            data_a = data_list[each_idx_a]
            data_b = data_list[each_idx_b]

            laser_on_data = None
            laser_off_data = None
            if data_a.neg_delay_laser_on:
                laser_on_data = data_a
                laser_off_data = data_b
            else:
                laser_on_data = data_b
                laser_off_data = data_a

            diff_int = np.array(laser_on_data.intensity_val) - np.array(laser_off_data.intensity_val)
            diff_int_arr.append(diff_int)

        print("negative delay pair num : ", len(diff_int_arr))
        if draw_each_graph:
            num_draw_front_diff = 9
            num_data_in_one_graph = 3
            for diff_idx in range(num_draw_front_diff):
                plt.plot(self.q_val[plot_start_idx:plot_end_idx],
                         diff_int_arr[diff_idx][plot_start_idx:plot_end_idx],
                         label=(str(diff_idx + 1) + 'th diff'))
                if ((diff_idx + 1) % num_data_in_one_graph) == 0:
                    plt.title("draw each diff - nearest intensity")
                    plt.xlabel('q value')
                    plt.ylabel("intensity")
                    plt.legend()
                    plt.show()
        now_data_len = len(diff_int_arr)
        avg_diff = np.average(diff_int_arr, axis=0)
        if plot:
            plt.plot(self.q_val[plot_start_idx:plot_end_idx], avg_diff[plot_start_idx:plot_end_idx])
            plt.title("negative delay avg norm diff - int on/off pair (from" + str(delay_idx) + "-th delay)")
            plt.xlabel('q value')
            plt.ylabel("intensity")
            plt.show()
        return avg_diff, now_data_len

    def pairwise_diff_calc(self, file_out=False, show_cmp=True, show_neg_pair=False, plot_each_avg=False):
        global PlotStartIdx
        global PlotEndIdx
        plot_start_idx = PlotStartIdx
        plot_end_idx = PlotEndIdx
        print("plot_start / end", plot_start_idx, plot_end_idx)

        int_diff_each_delay = []

        for each_delay in range(self.num_of_delay):
            water_laser_on_idx, water_laser_off_idx = self.extract_laser_on_off_water_list(
                self.intensity_data_list[each_delay])
            water_nearest_int_pair = self.match_water_near_intensity_pair(self.intensity_data_list[each_delay],
                                                                          water_laser_on_idx, water_laser_off_idx)
            neg_water_on_idx, neg_water_off_idx = self.extract_neg_delay_laser_on_off_water_list(
                self.intensity_data_list[each_delay], water_laser_off_idx)

            neg_delay_int_pair = self.match_water_near_intensity_pair(self.intensity_data_list[each_delay],
                                                                      neg_water_on_idx, neg_water_off_idx)
            print("negative delay pairing end")

            for each_data in self.intensity_data_list[each_delay]:
                each_data.norm_given_range()

            near_int_diff = self.plot_near_int_diff(self.intensity_data_list[each_delay],
                                                    water_nearest_int_pair, each_delay, draw_each_graph=False,
                                                    plot_start_idx=plot_start_idx, plot_end_idx=plot_end_idx,
                                                    each_avg_plot=plot_each_avg)
            neg_int_diff, neg_diff_len = self.plot_neg_delay_near_int_diff(self.intensity_data_list[each_delay],
                                                             neg_delay_int_pair, each_delay,
                                                             plot_start_idx=plot_start_idx, plot_end_idx=plot_end_idx,
                                                             plot=show_neg_pair)
            int_diff_each_delay.append(near_int_diff)
            self.neg_delay_avg_diff_list.append(neg_int_diff)
            self.neg_diff_data_len.append(neg_diff_len)

            if show_cmp:
                plt.plot(self.q_val[plot_start_idx:plot_end_idx], near_int_diff[plot_start_idx:plot_end_idx],
                         label="intensity pair")
                plt.plot(self.q_val[plot_start_idx:plot_end_idx], neg_int_diff[plot_start_idx:plot_end_idx],
                         label="negative_delay")

                plt.title("comparison of average normalized diff (" + str(each_delay) + "-th delay)")
                plt.xlabel('q value')
                plt.ylabel("intensity")
                plt.legend()
                plt.show()

        for delay_idx, diff_data in enumerate(int_diff_each_delay):
            try:
                plt.plot(self.q_val[plot_start_idx:plot_end_idx],
                         diff_data[plot_start_idx:plot_end_idx],
                         label=str(delay_idx) + "th delay")
            except:
                print("no diff graph in delay", delay_idx)
        # plt.ylim(-0.5, 0.5)
        plt.title("delay comparison of int pair diff")
        plt.xlabel('q value')
        plt.ylabel("intensity")
        plt.legend()
        plt.show()

        if file_out:
            self.file_out_delay_compare(int_diff_each_delay)

    def calc_SVD_and_plot(self, file_out=False, avg_SVD=False):
        concat_diff_arr = []
        each_delay_count = []
        for delay_idx in range(self.num_of_delay):
            concat_diff_arr.extend(self.int_pair_diff_save[delay_idx])
            each_delay_count.append(len(self.int_pair_diff_save[delay_idx]))
        tp_concat_diff = np.transpose(concat_diff_arr)
        concat_diff_arr = None
        diffSVD = SVDCalc(tp_concat_diff)
        diffSingVal = diffSVD.calc_svd()

        singular_show_num = 50
        print(diffSingVal[:singular_show_num])
        singular_data_y = diffSingVal[:singular_show_num]
        singular_data_y_log = np.log(singular_data_y)
        singular_data_x = range(1, len(singular_data_y) + 1)

        def plot_singular_value(data_x, data_y, data_y_log):
            color_r = 'tab:red'
            fig, ax1 = plt.subplots()
            ax1.set_xlabel("index of singular value")
            ax1.set_ylabel("singular value", color=color_r)
            ax1.scatter(data_x, data_y, color=color_r)
            ax1.plot(data_x, data_y, color=color_r)
            ax1.tick_params(axis='y', labelcolor=color_r)

            ax2 = ax1.twinx()
            color_b = 'tab:blue'
            ax2.set_ylabel("log scale singular value", color=color_b)
            ax2.scatter(data_x, data_y_log, color=color_b)
            ax2.plot(data_x, data_y_log, color=color_b)
            ax2.tick_params(axis='y', labelcolor=color_b)

            fig.tight_layout()
            plt.show()

        plot_singular_value(singular_data_x, singular_data_y, singular_data_y_log)

        singular_cut_num = 5
        bigSingVal = diffSingVal[:singular_cut_num]
        print(bigSingVal)

        print("left", diffSVD.leftVec.shape)
        print("right", diffSVD.rightVecTrans.shape)
        diffSVD.pick_meaningful_data(singular_cut_num)
        print("left", diffSVD.meanLeftVec.shape)
        print("right", diffSVD.meanRightVec.shape)
        diffSVD.plot_left_Vec()
        # delay_1_start = each_delay_count[0]
        # delay_2_start = each_delay_count[0] + each_delay_count[1]
        delay_1_start = 2130
        delay_2_start = 2130 + 2037
        diffSVD.plot_right_Vec(v_line_1=delay_1_start, v_line_2=delay_2_start)

        if file_out:
            left_file_name = "../results/svd/" + self.file_main_name + "_LSV.dat"
            right_file_name = "../results/svd/" + self.file_main_name + "_RSV.dat"
            singVal_file_name = "../results/svd/" + self.file_main_name + "_SingVal.dat"
            leftFp = open(left_file_name, 'w')
            rightFp = open(right_file_name, 'w')
            svalFp = open(singVal_file_name, 'w')
            diffSVD.file_output_singular_vectors(leftFp, rightFp)
            print("LSV, RSV file print")
            diffSVD.file_output_singular_value(svalFp)
            leftFp.close()
            rightFp.close()
            svalFp.close()

    def set_file_pointer(self, dict_keyword, file):
        keyword_list = ['each-delay-diff', 'time-delay-avg-diff', 'cut-each-delay-diff', 'cut-time-delay-avg-diff',
                        'laser-off-avg-np', 'delay-group-avg-np', 'each-delay-avg-np', 'off-delay-avg-np']
        if dict_keyword not in keyword_list:
            raise ValueError("not defined file type keyword")
        self.out_file_pointers[dict_keyword] = file

    def file_out_delay_compare(self, data):
        outFp = self.out_file_pointers['each-delay-diff']
        outFp.write("difference of each delay - analysis result of " + self.file_main_name + "\n")
        outFp.write("q-value\t")
        for delay_idx in range(self.num_of_delay):
            outFp.write(str(delay_idx) + "th delay\t")
        outFp.write("\n")
        for data_idx in range(len(self.q_val)):
            outFp.write("%.5f\t" % self.q_val[data_idx])
            for delay_idx in range(self.num_of_delay):
                outFp.write("%.5f\t" % data[delay_idx][data_idx])
            outFp.write("\n")

    def additional_process_diff(self, show_before_cutted=True, show_after_cutted=False, file_out=False,
                                svd_with_cut=False, weighted_group_avg=False, sep_np_save=False):
        global PlotStartIdx
        global PlotEndIdx
        global svd_start_idx
        global svd_end_idx
        plot_start_idx = PlotStartIdx
        plot_end_idx = PlotEndIdx

        intg_start_idx = 93
        intg_end_idx = -180
        # cutoff_criteria = [2.2E5, 2.3E5, 2.7E5, 3.4E5,
        #                    2.5E5, 2.3E5, 1.8E5, 1.9E5,
        #                    2.7E5, 2.5E5, 3.25E5, 2E5,
        #                    2E5, 2E5, 1.7E5, 2E5]
        cutoff_criteria = [2E5 for _ in range(self.num_of_delay)]
        cutted_diff_list = []
        cutted_avg = []

        for each_delay in range(self.num_of_delay):
            # plot first histogram & criteria
            sum_list = []
            for diff_data in self.int_pair_diff_save[each_delay]:
                now_sum = np.sum(np.abs(diff_data[intg_start_idx:intg_end_idx]))
                sum_list.append(now_sum)
            # print(sum_list)
            sum_list = np.array(sum_list)
            if show_before_cutted:
                now_graph_title = "integration hist of " + str(each_delay) + "-th delay"
                self.plot_sum_for_criteria(sum_list, now_graph_title, v_line_1=cutoff_criteria[each_delay])
                print(now_graph_title)

            # left only diff pair with small integration value
            # and calculated average of them
            cut_diff = np.array(self.int_pair_diff_save[each_delay])[sum_list < cutoff_criteria[each_delay]]

            print("remove ",
                  len(np.array(self.int_pair_diff_save[each_delay])[sum_list >= cutoff_criteria[each_delay]]),
                  " in", each_delay, "-th delay")
            self.cutted_after_data_len.append(len(cut_diff))
            # now_cutted_q = np.array(self.q_val)
            cutted_diff_list.append(cut_diff)
            avg_diff = np.average(cut_diff, axis=0)
            if avg_diff.shape is np.nan:
                avg_diff = np.zeros_like(self.q_val[svd_start_idx:svd_end_idx])
            cutted_avg.append(avg_diff)

            if svd_with_cut:
                self.int_pair_diff_save[each_delay] = cut_diff

            if show_after_cutted:
                # plot second (after cut) histogram
                cutted_sum_list = []
                for diff_data in cut_diff:
                    now_sum = np.sum(np.abs(diff_data[intg_start_idx:intg_end_idx]))
                    cutted_sum_list.append(now_sum)
                cutted_sum_list = np.array(cutted_sum_list)
                now_graph_title = "cutted integration hist of " + str(each_delay) + "-th delay"
                self.plot_sum_for_criteria(cutted_sum_list, now_graph_title, v_line_1=cutoff_criteria[each_delay])
                print(now_graph_title)

        if svd_with_cut:
            num_delay_in_one_group = 8
            group_data_len = []
            for each_delay in range(self.num_of_delay):
                if (each_delay % num_delay_in_one_group) == 0:
                    group_data_len.append(self.cutted_after_data_len[each_delay])
                else:
                    group_data_len[-1] += self.cutted_after_data_len[each_delay]
            print("data num in each group : ", group_data_len)
            print("each delay data num : ", self.cutted_after_data_len)

        if sep_np_save:
            np.save(self.out_file_pointers['each-delay-avg-np'], cutted_avg)


        for delay_idx, avg_diff in enumerate(cutted_avg):
            avg_diff = np.array(avg_diff)
            try:
                plt.plot(self.q_val[svd_start_idx:svd_end_idx],
                         avg_diff,
                         label=str(delay_idx) + "th delay cutted")
            except:
                avg_diff = np.zeros_like(self.q_val[svd_start_idx:svd_end_idx])
                plt.plot(self.q_val[svd_start_idx:svd_end_idx], avg_diff, label=str(delay_idx) + "th delay cutted")
                print("additional process avg error in delay", delay_idx)
        plt.title("delay comparison of cutted int pair diff ")
        plt.xlabel('q value')
        plt.ylabel("intensity")
        plt.legend()
        plt.show()

        if weighted_group_avg:
            # group 1 : delay 0 ~ 7
            # group 2 : delay 8 ~ 15
            group_1_avg = np.average(cutted_avg[:8], axis=0, weights=self.cutted_after_data_len[:8])
            group_2_avg = np.average(cutted_avg[8:], axis=0, weights=self.cutted_after_data_len[8:])
            plt.plot(self.q_val[svd_start_idx:svd_end_idx], group_1_avg, label="group 1 avg")
            plt.plot(self.q_val[svd_start_idx:svd_end_idx], group_2_avg, label="group 2 avg")
            plt.title("cutted group average compare")
            plt.xlabel('q value')
            plt.ylabel("intensity")
            plt.legend()
            plt.show()

            delay_group_avg = np.stack([group_1_avg, group_2_avg])
            np.save(self.out_file_pointers['delay-group-avg-np'], delay_group_avg)

        if file_out:
            self.file_out_cutted_delay_compare(cutted_avg)

    def process_laser_off(self, save_all_avg=False, save_delay_avg=False):
        global PlotStartIdx
        global PlotEndIdx
        plot_start_idx = PlotStartIdx
        plot_end_idx = PlotEndIdx
        print(len(self.neg_delay_avg_diff_list))
        # each delay average data save
        if save_delay_avg:
            np.save(self.out_file_pointers['off-delay-avg-np'], self.neg_delay_avg_diff_list)
        neg_delay_whole_avg = np.average(self.neg_delay_avg_diff_list, axis=0)
        print("weight : ", self.neg_diff_data_len)
        neg_weighted_avg = np.average(self.neg_delay_avg_diff_list, axis=0, weights=self.neg_diff_data_len)
        plt.plot(self.q_val[plot_start_idx:plot_end_idx],
                 neg_delay_whole_avg[plot_start_idx:plot_end_idx], label="no weight avg")
        plt.plot(self.q_val[plot_start_idx:plot_end_idx],
                 neg_weighted_avg[plot_start_idx:plot_end_idx], label="weighted avg")
        plt.title("negative delay avg diff - same delay paring")
        plt.xlabel('q value')
        plt.ylabel("intensity")
        plt.legend()
        plt.show()
        # delay group weighted average save.
        if save_all_avg:
            np.save(self.out_file_pointers['laser-off-avg-np'], neg_weighted_avg)


    def file_out_cutted_delay_compare(self, data):
        global svd_start_idx
        global svd_end_idx
        outFp = self.out_file_pointers['cut-each-delay-diff']
        outFp.write("difference of each delay - analysis result of " + self.file_main_name + "\n")
        outFp.write("q-value\t")
        for delay_idx in range(self.num_of_delay):
            outFp.write(str(delay_idx) + "th delay\t")
        outFp.write("\n")
        cutted_q = self.q_val[svd_start_idx:svd_end_idx]
        print("cutted_q len : ", len(cutted_q), "value len : ", len(data[0]))
        for data_idx in range(len(cutted_q)):
            outFp.write("%.5f\t" % cutted_q[data_idx])
            for delay_idx in range(self.num_of_delay):
                outFp.write("%.5f\t" % data[delay_idx][data_idx])
            outFp.write("\n")



