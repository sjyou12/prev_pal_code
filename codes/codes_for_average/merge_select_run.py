from palxfel_scatter.diff_pair_1dcurve.MultiRunProc import MultiRunProc
from palxfel_scatter.diff_pair_1dcurve.DataClasses import plot_sum_for_criteria, match_water_near_intensity_pair
from palxfel_scatter.diff_pair_1dcurve.PulseDataLight import PulseDataLight
import numpy as np
import matplotlib.pyplot as plt
import datetime
import re
import scipy.signal as scipy


common_file_root = "/home/myeong0609/PAL-XFEL_20210514/analysis/results/"
#watersum_pass_int_file_common_root = common_file_root + "watersum/"
avg_run_list = [11, 12]
right_time_delay_list = [-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3]
now_file_family_name = "run11-12-avg"


def find_peak_pattern_multirun(self, incr_dist_plot=False, plot_outlier=False, sum_file_out=False): #without small ice test
    plot_within_range = False
    rm_ice_within_range = True
    plot_each_outlier = False
    plot_incr_test_pass = False

    print("going to find strange peak pattern")
    start_time = datetime.datetime.now()
    print(start_time)
    # TODO : need to change
    inc_max_outlier_boundary = 100
    inc_range_min = 50
    inc_range_max = 100
    each_outlier_plot_list = [12, 16]
    incr_test_pass_info = []
    incr_test_pass_int = []
    incr_max_within_range_info = []
    incr_max_within_range_int = []
    print_criteria = 10

    water_range_int_sum_list = []
    incr_max_list = []
    incr_outlier_int = []
    incr_outlier_info = []
    for idx_run, each_run_int_files in enumerate(self.intensity_files):
        now_int_file_num = len(each_run_int_files)
        now_water_sum_list = []
        now_fileout_list = []
        for idx_file, each_int_file in enumerate(each_run_int_files):
            now_int_keys = list(each_int_file.keys())
            # num_small_ice_peak_each_delay = 0
            for each_key in now_int_keys:
                now_int_val = np.array(each_int_file[each_key])
                now_int_water_sum = sum(now_int_val[self.water_q_range_start_idx:self.water_q_range_after_idx])
                now_int_next = np.roll(now_int_val, 1)
                now_int_next[0] = now_int_next[1]  # remove last element
                now_int_incr = np.abs(now_int_val - now_int_next)
                now_incr_max = np.max(now_int_incr[self.water_q_range_start_idx:self.water_q_range_after_idx])
                incr_max_list.append(now_incr_max)
                if rm_ice_within_range:
                    if now_incr_max > inc_max_outlier_boundary:
                        incr_outlier_int.append(now_int_val)
                        incr_outlier_info.append([self.runList[idx_run], idx_file, each_key])
                        self.strange_peak_key_blacklist.append(each_key)
                        if inc_range_max > now_incr_max > inc_range_min:
                            incr_max_within_range_int.append(now_int_val)
                            incr_max_within_range_info.append([self.runList[idx_run], idx_file, each_key])

                    else:
                        # now_pulseID = re.findall("(.*)\.(.*)_(.*)", each_key)[0][2]
                        # now_pulseID = int(now_pulseID)
                        incr_test_pass_int.append(now_int_val)
                        incr_test_pass_info.append([self.runList[idx_run], idx_file, each_key])

                else:
                    if now_incr_max > inc_max_outlier_boundary:
                        incr_outlier_int.append(now_int_val)
                        incr_outlier_info.append([self.runList[idx_run], idx_file, each_key])
                        self.strange_peak_key_blacklist.append(each_key)

                    else:
                        # now_pulseID = re.findall("(.*)\.(.*)_(.*)", each_key)[0][2]
                        # now_pulseID = int(now_pulseID)
                        incr_test_pass_int.append(now_int_val)
                        incr_test_pass_info.append([self.runList[idx_run], idx_file, each_key])

                now_water_sum_list.append(now_int_water_sum)
                now_file_out = [each_key, now_int_water_sum]
                now_fileout_list.append(now_file_out)
            if (idx_file + 1) % print_criteria == 0:
                print("read {0} / {1} file".format(idx_file + 1, now_int_file_num))

        water_range_int_sum_list.append(now_water_sum_list)
        # TODO : add ice range plot
        print("end for run{} files".format(self.runList[idx_run]))

        if sum_file_out:
            now_fileout_list = np.array(now_fileout_list)
            now_save_file_root = "../results/anisotropy/run" + str(self.runList[idx_run]) + "_watersum"
            np.save(now_save_file_root, now_fileout_list)
            print("successful file out :", now_save_file_root)
    finish_time = datetime.datetime.now()
    print(finish_time)
    merge_water_sum_list = np.array(water_range_int_sum_list).reshape(-1)
    print(merge_water_sum_list.shape)
    # plot_sum_for_criteria(merge_water_sum_list, "water range sum view of all run")
    hist_water_sum, bin_edges = np.histogram(water_range_int_sum_list, bins=200)
    self.hist_fileout(hist_water_sum, bin_edges, file_name="../results/water_sum_hist.dat", runList=self.runList)

    # temp value
    self.UpperBoundOfNotWater = self.eachRunInfo[0][1]
    self.LowerBoundOfWater = self.eachRunInfo[0][2]
    self.WaterOutlierLowerBound = self.eachRunInfo[0][3]

    plot_sum_for_criteria(merge_water_sum_list, "water range sum view of all run with criteria", self.LowerBoundOfWater, self.UpperBoundOfNotWater, self.WaterOutlierLowerBound)

    if len(incr_outlier_info) != 0:
        if incr_dist_plot:
            plot_sum_for_criteria(incr_max_list, "incr max value of run {}".format(incr_outlier_info[0][0]),
                                  inc_max_outlier_boundary)

        if plot_outlier:
            one_graph_plot = 10
            if plot_within_range:
                for idx_outlier, each_int in enumerate(incr_max_within_range_int):
                    now_label = "run{0}-delay{1}-{2}".format(incr_max_within_range_info[idx_outlier][0], incr_max_within_range_info[idx_outlier][1] + 1, idx_outlier)
                    print("{0}-th outlier : run{1}-delay{2}-key{3}".format(idx_outlier, incr_max_within_range_info[idx_outlier][0], incr_max_within_range_info[idx_outlier][1] + 1, incr_max_within_range_info[idx_outlier][2]))
                    # plt.plot(each_int, label=now_label)
                    plt.plot(each_int, marker='.', markersize=1, label=now_label)
                    if idx_outlier % one_graph_plot == (one_graph_plot - 1):
                        plt.title("outlier of run" + str(incr_max_within_range_info[0][0]))
                        plt.legend()
                        plt.show()
                plt.title("outlier of run" + str(incr_max_within_range_info[0][0]))
                plt.legend()
                plt.show()

            else:
                for idx_outlier, each_int in enumerate(incr_outlier_int):
                    now_label = "run{0}-delay{1}-{2}".format(incr_outlier_info[idx_outlier][0],
                                                             incr_outlier_info[idx_outlier][1] + 1, idx_outlier)
                    print(
                        "{0}-th outlier : run{1}-delay{2}-key{3}".format(idx_outlier, incr_outlier_info[idx_outlier][0], incr_outlier_info[idx_outlier][1] + 1, incr_outlier_info[idx_outlier][2]))
                    # plt.plot(each_int, label=now_label)
                    plt.plot(each_int, marker='.', markersize=1, label=now_label)
                    if idx_outlier % one_graph_plot == (one_graph_plot - 1):
                        plt.title("outlier of run" + str(incr_outlier_info[0][0]))
                        plt.legend()
                        plt.show()
                plt.title("outlier of run" + str(incr_outlier_info[0][0]))
                plt.legend()
                plt.show()
        elif plot_each_outlier:
            idx_each_outlier = 0
            for idx_outlier, each_int in enumerate(incr_outlier_int):
                try:
                    if idx_outlier == each_outlier_plot_list[idx_each_outlier]:
                        now_label = "run{0}-delay{1}-{2}".format(incr_outlier_info[idx_outlier][0], incr_outlier_info[idx_outlier][1] + 1, idx_outlier)
                        print("{0}-th outlier : run{1}-delay{2}-key{3}".format(idx_outlier,incr_outlier_info[idx_outlier][0], incr_outlier_info[idx_outlier][1] + 1, incr_outlier_info[idx_outlier][2]))
                        # plt.plot(each_int, label=now_label)
                        plt.plot(each_int, marker='.', markersize=1, label=now_label)
                        plt.title("outlier of run" + str(incr_outlier_info[0][0]))
                        plt.legend()
                        plt.show()
                        idx_each_outlier += 1
                    else:
                        continue
                except:
                    continue

        elif plot_incr_test_pass:
            one_graph_plot = 10
            for idx_pass, each_int in enumerate(incr_test_pass_int):
                # now_label = "run{0}-delay{1}-{2}-q_idx {3}-key {4}".format(diff_test_fail_info[idx_outlier][0],diff_test_fail_info[idx_outlier][1] + 1, idx_outlier, where_max_diff_test_fail[idx_outlier],diff_test_fail_info[idx_outlier][2])
                now_label = "run{0}-delay{1}-{2}".format(incr_test_pass_info[idx_pass][0],
                                                         incr_test_pass_info[idx_pass][1] + 1, idx_pass)
                # print("{0}-th outlier : run{1}-delay{2}-key{3}".format(idx_outlier, diff_test_fail_info[idx_outlier][0], diff_test_fail_info[idx_outlier][1] + 1,diff_test_fail_info[idx_outlier][2]))
                # plt.plot(each_int, label=now_label)
                plt.plot(each_int, marker='.', markersize=1, label=now_label)
                if idx_pass % one_graph_plot == (one_graph_plot - 1):
                    plt.title("Increment test pass of run" + str(incr_test_pass_info[0][0]))
                    plt.legend()
                    plt.show()
            plt.title("Increment test pass of run" + str(incr_test_pass_info[0][0]))
            plt.legend()
            plt.show()

        print("now remove {0} strange peak at run{1}".format(len(incr_outlier_int), incr_outlier_info[0][0]))
        # np.savetxt('Difference test failure of run 50.txt', diff_test_fail_info, fmt='%s', delimiter='/')

    def read_intensity_only_water_multirun(self, np_file_out=False, rm_vapor=False, plot_watersum_pass=False):
        """
        execute one of (this function and plot_water_sum_dist)
        two function have overlapped feature
        """
        print("read intensity files and save water range data")

        print_criteria = 10
        watersum_test_pass_info = []
        watersum_test_pass_int_all_delay = []
        num_watersum_pass_each_delay = []
        num_watersum_pass_in_each_delay = 0

        self.UpperBoundOfNotWater = self.eachRunInfo[0][1]
        self.LowerBoundOfWater = self.eachRunInfo[0][2]
        self.WaterOutlierLowerBound = self.eachRunInfo[0][3]

        if len(self.strange_peak_key_blacklist) == 0:
            print("strange peak blacklist is empty!")

        for idx_run, each_run_int_files in enumerate(self.intensity_files):
            now_int_file_num = len(each_run_int_files)
            now_run_int_val_list = []
            for idx_file, each_int_file in enumerate(each_run_int_files):
                now_int_keys = list(each_int_file.keys())
                now_delay_watersum_pass_list =[]
                now_delay_int_val_list = []
                for each_key in now_int_keys:
                    if each_key in self.strange_peak_key_blacklist:
                        # print("now key is skipped")
                        continue
                    now_int_val = np.array(each_int_file[each_key])
                    now_int_sum = sum(now_int_val[self.water_q_range_start_idx:self.water_q_range_after_idx])
                    if self.LowerBoundOfWater < now_int_sum < self.WaterOutlierLowerBound:
                        if rm_vapor:
                            now_I0_val = self.all_run_I0_dict_list[idx_run][idx_file][each_key]
                            now_vapor_int = np.multiply(self.norm_vapor_int, now_I0_val)
                            now_int_val = now_int_val - now_vapor_int
                        now_int_obj = PulseDataLight(now_int_val, each_key,
                                                     self.water_q_range_start_idx, self.water_q_range_after_idx,
                                                     self.norm_q_range_start_idx, self.norm_q_range_after_idx,
                                                     self.pair_q_range_start_idx, self.pair_q_range_after_idx)
                        now_pulseID = re.findall("(.*)\.(.*)_(.*)", each_key)[0][2]
                        now_pulseID = int(now_pulseID)
                        now_int_obj.check_laser_onoff(now_pulseID)
                        now_delay_int_val_list.append(now_int_obj)
                        watersum_test_pass_info.append([self.runList[idx_run], idx_file, now_pulseID, num_watersum_pass_in_each_delay])
                        now_delay_watersum_pass_list.append(now_int_val)
                        num_watersum_pass_in_each_delay += 1
                watersum_test_pass_int_all_delay.append(now_delay_watersum_pass_list)
                num_watersum_pass_each_delay.append(num_watersum_pass_in_each_delay)
                now_run_int_val_list.append(now_delay_int_val_list)
                if (idx_file + 1) % print_criteria == 0:
                    print("read {0} / {1} file".format(idx_file + 1, now_int_file_num))
            #self.each_run_int_val_list.append(now_run_int_val_list)
            self.each_run_int_val_list.extend(now_run_int_val_list)
            print("end intensity read for run{} files".format(self.runList[idx_run]))

        if plot_watersum_pass:
            one_graph_plot = 10
            for idx_delay in range(len(watersum_test_pass_int_all_delay)):
                try:
                    tot_num_early_delay = num_watersum_pass_each_delay[idx_delay - 1]

                except:
                    continue

                for num_watersum_pass_idx in range(10):
                    if idx_delay == 0:
                        now_label = "run{0}-delay{1}-{2}".format(watersum_test_pass_info[idx_delay][0], idx_delay + 1, watersum_test_pass_info[idx_delay+num_watersum_pass_idx][3])

                    else:
                        now_label = "run{0}-delay{1}-{2}".format(watersum_test_pass_info[idx_delay][0], idx_delay + 1, watersum_test_pass_info[tot_num_early_delay+num_watersum_pass_idx][3])

                    plt.plot(watersum_test_pass_int_all_delay[idx_delay][num_watersum_pass_idx], marker='.', markersize=1, label=now_label)

                    if num_watersum_pass_idx % one_graph_plot == (one_graph_plot - 1) :
                        plt.title("Shots of pass watersum test of run" + str(watersum_test_pass_info[0][0]))
                        plt.legend()
                        #plt.xlim(130, 350)
                        plt.show()
                    # plt.title("Shots of pass watersum test of run" + str(watersum_test_pass_info[0][0]))

        np.savetxt("Information of watersum test passed shots.txt", watersum_test_pass_info, fmt='%s', delimiter='/')
        print("now {} shots remain after watersum test at run{}".format(len(watersum_test_pass_info), watersum_test_pass_info[0][0]))
        if np_file_out:
            # save each_run_int_val_list numpy array
            temp_save_name = "../results/whole_run_int/whole_run_int_" + self.file_common_name + ".npy"
            print("save as :", temp_save_name)
            np.save(temp_save_name, self.each_run_int_val_list)
            #np.savetxt('run_50_watersum.txt', self.each_run_int_val_list, fmt='%s', delimiter=' ')

            # save q_val numpy array
            temp_save_name = "../results/q_val_" + self.file_common_name + ".npy"
            print("save as :", temp_save_name)
            np.save(temp_save_name, self.q_val)

def small_ice_test(self, run_list_to_merge =[], plot_outlier=False, sum_file_out=False):
    #incr_test_pass_int_file_common_root = common_file_root + "_watersum/"

    rm_ice_within_range = True
    plot_each_outlier = False
    plot_small_ice_test_fail = False
    plot_incr_test_pass = False
    small_ice_test = True

    print("going to find small ice peak pattern")
    start_time = datetime.datetime.now()
    print(start_time)
    # TODO : need to change

    prominence_max_val = 10
    prominence_min_val = 3.5
    each_outlier_plot_list = [12, 16]
    incr_test_pass_info = []
    incr_test_pass_int = []
    small_ice_test_fail_int = []
    small_ice_test_fail_info = []
    small_ice_test_fail_all_delay = []
    small_ice_plot_start_delay = 30
    small_ice_plot_end_delay = 47
    num_small_ice_peaks_all_delay = []
    num_small_ice_peak_each_delay = 0
    print_criteria = 10

    temp_int_list = []
    whole_run_int_files = []

    water_range_int_sum_list = []
    incr_max_list = []
    incr_outlier_int = []
    incr_outlier_info = []

    for list_idx, idx_run in enumerate(run_list_to_merge):
        now_watersum_test_pass_save_file_root = "../results/each_run_watersum_int/" + str(self.runList[idx_run]) + "_watersum"
        temp_int_list = np.load(now_watersum_test_pass_save_file_root)
        whole_run_int_files.extend(temp_int_list)

    for idx_run, each_run_int_files in enumerate(whole_run_int_files):
        now_int_file_num = len(each_run_int_files)
        now_water_sum_list = []
        now_fileout_list = []
        for idx_file, each_int_file in enumerate(each_run_int_files):
            now_int_keys = list(each_int_file.keys())
            # num_small_ice_peak_each_delay = 0
            now_delay_small_ice_fail_int = []
            for each_key in now_int_keys:
                now_int_val = np.array(each_int_file[each_key])
                now_int_water_sum = sum(now_int_val[self.water_q_range_start_idx:self.water_q_range_after_idx])
                now_int_next = np.roll(now_int_val, 1)
                now_int_next[0] = now_int_next[1]  # remove last element
                now_int_incr = np.abs(now_int_val - now_int_next)
                now_incr_max = np.max(now_int_incr[self.water_q_range_start_idx:self.water_q_range_after_idx])
                incr_max_list.append(now_incr_max)
                if small_ice_test:
                    num_small_ice_peaks, properties = scipy.find_peaks(now_int_incr[self.water_q_range_start_idx:self.water_q_range_after_idx], prominence=(prominence_min_val, prominence_max_val))
                    if len(num_small_ice_peaks) != 0:
                        incr_outlier_int.append(now_int_val)
                        incr_outlier_info.append([self.runList[idx_run], idx_file, each_key])
                        self.strange_peak_key_blacklist.append(each_key)
                        now_pulseID = re.findall("(.*)\.(.*)_(.*)", each_key)[0][2]
                        now_pulseID = int(now_pulseID)
                        small_ice_test_fail_int.append(now_int_val)
                        small_ice_test_fail_info.append([self.runList[idx_run], idx_file, now_pulseID])
                        now_delay_small_ice_fail_int.append(now_int_val)
                        num_small_ice_peak_each_delay += 1

                    else:
                        incr_test_pass_int.append(now_int_val)
                        incr_test_pass_info.append([self.runList[idx_run], idx_file, each_key])
                        # diff_test_fail_info.append([self.runList[idx_run], idx_file, now_pulseID, temp_q_idx_of_diff_max])
                    now_water_sum_list.append(now_int_water_sum)
                    now_file_out = [each_key, now_int_water_sum]
                    now_fileout_list.append(now_file_out)
                if (idx_file + 1) % print_criteria == 0:
                    print("read {0} / {1} file".format(idx_file + 1, now_int_file_num))
                small_ice_test_fail_all_delay.append(now_delay_small_ice_fail_int)
                num_small_ice_peaks_all_delay.append(num_small_ice_peak_each_delay)
            water_range_int_sum_list.append(now_water_sum_list)
            # TODO : add ice range plot
            print("end for run{} files".format(self.runList[idx_run]))

            if sum_file_out:
                now_fileout_list = np.array(now_fileout_list)
                for idx in range(len(run_list_to_merge)):
                    if idx != (len(run_list_to_merge) - 1):
                        tot_run_name = str(run_list_to_merge[idx]) + "+"

                    else:
                        tot_run_name = str(run_list_to_merge[idx])
                now_save_file_root = "../results/merge_run_small_ice_pass/run" + str(tot_run_name) + "_small_ice_test"
                np.save(now_save_file_root, now_fileout_list)
                print("successful file out :", now_save_file_root)

def pairwise_diff_calc(self, merge_int_val_list = [], test_plot=False, test_plot_num=5, fileout_pair_info=True):
    plot_start_idx = 10
    plot_end_idx = -120
    print("plot_start / end", plot_start_idx, plot_end_idx)

    #for each_run_int_list in self.each_run_int_val_list:
    for each_run_int_list in merge_int_val_list:
        now_run_diff_list = []
        now_pair_arr = []
        for each_delay_int_list in each_run_int_list:
            num_each_delay_laser_on = []
            num_each_delay_laser_off = []
            laser_on_idx, laser_off_idx = extract_laser_on_off_list_only_water(each_delay_int_list)
            num_each_delay_laser_off.append(laser_off_idx)
            num_each_delay_laser_on.append(laser_on_idx)
            print("pairing with nearest integrated intensity logic")
            nearest_int_pair = match_water_near_intensity_pair(each_delay_int_list, laser_on_idx, laser_off_idx)
            if fileout_pair_info:
                pair_pulseID_arr = self.make_pulseID_array(each_delay_int_list, nearest_int_pair)
                now_pair_arr.append(pair_pulseID_arr)
            for each_data in each_delay_int_list:
                each_data.norm_given_range()
            near_int_pair_diff = self.calc_near_int_pair_diff(each_delay_int_list, nearest_int_pair)
            now_run_diff_list.append(near_int_pair_diff)

        self.whole_run_diff_list.append(now_run_diff_list)
        if fileout_pair_info:
            self.whole_run_pair_list.append(now_pair_arr)

        if test_plot:
            for delay_idx in range(test_plot_num):
                for idx in range(test_plot_num):
                    plt.plot(self.q_val, now_run_diff_list[delay_idx][idx], label=(str(idx + 1) + 'th diff '))
                plt.title("test draw each diff - nearest intensity of delay" + str(delay_idx + 1))
                plt.xlabel('q value')
                plt.ylabel("intensity")
                plt.legend()
                plt.show()

    self.each_run_int_val_list = []
    print("remove each_run_int_val_list. It is already saved as file")

    # save whole_run_diff_list numpy array
    temp_save_name = "../results/whole_run_diff/whole_run_diff_" + self.file_common_name + ".npy"
    print("save as :", temp_save_name)
    np.save(temp_save_name, self.whole_run_diff_list)

def extract_laser_on_off_list_only_water(data_list):
    laser_on_water_idx = []
    laser_off_water_idx = []

    for data_idx, each_data in enumerate(data_list):
        if each_data.laser_is_on:
            laser_on_water_idx.append(data_idx)
        else:
            laser_off_water_idx.append(data_idx)

    print("laser on droplet : ", len(laser_on_water_idx), "laser off : ", len(laser_off_water_idx))
    return laser_on_water_idx, laser_off_water_idx

