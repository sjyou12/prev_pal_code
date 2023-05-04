from palxfel_scatter.diff_pair_1dcurve.DataClasses import plot_sum_for_criteria, match_water_near_intensity_pair
from palxfel_scatter.diff_pair_1dcurve.Tth2qConvert import Tth2qConvert
from palxfel_scatter.diff_pair_1dcurve.PulseDataLight import PulseDataLight
import h5py as h5
import numpy as np
import re
import matplotlib.pyplot as plt
import os
import datetime
import scipy.signal as scipy
import scipy.interpolate as interpolate

class MultiRunProc:
    each_run_int_val_list = []
    twotheta_file_name = None
    twotheta_val = []
    q_val = []
    water_q_range_start_idx = None
    water_q_range_after_idx = None
    norm_q_range_start_idx = None
    norm_q_range_after_idx = None
    pair_q_range_start_idx = None
    pair_q_range_after_idx = None
    ice_q_range_start_idx = None
    ice_q_range_after_idx = None

    UpperBoundOfNotWater = 0
    LowerBoundOfWater = 0
    WaterOutlierLowerBound = 0

    tth_to_q_cvt = None
    each_run_file_dir = []
    whole_run_pair_list = []
    now_run_delay_num = 0
    strange_peak_key_blacklist = []

    shorten_img = False
    num_shorten_img = None
    on_on_off_test = False

    def __init__(self, each_run_info):
        self.watersum_pass_negative_delay = []
        self.intensity_file_names = []
        self.intensity_files = []
        self.whole_run_diff_list = []
        self.whole_run_cutted_diff_list = []
        self.eachRunInfo = each_run_info  # database format
        self.runList = []
        for run_info in each_run_info:
            self.runList.append(run_info[0])
        self.numOfRun = len(self.runList)
        run_name_conc = ("run" + str(self.runList[0]) + "-")
        for run_num in self.runList[1:]:
            run_name_conc += (str(run_num) + "-")
        run_name_conc = run_name_conc[:-1]
        self.file_common_name = run_name_conc
        print("now file common name : ", run_name_conc)
        self.norm_vapor_int = []
        self.all_run_I0_dict_list = []
        self.runNum = each_run_info[0][0]

    def common_variables(self, x_ray_energy, file_common_root):
        self.tth_to_q_cvt = Tth2qConvert(x_ray_energy)
        self.FileCommonRoot = file_common_root

    def read_twotheta_value(self):
        print("read tth value from file")

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

    def water_range_q_idx_calc(self):
        # set water droplet amorphous property signal range
        WaterPeakStartQ = 1.5
        WaterPeakEndQ = 3.5
        if len(self.q_val) == 0:
            print("no q value now!")
        self.water_q_range_start_idx = int(np.where(self.q_val >= WaterPeakStartQ)[0][0])
        # this index is not included in water q range!!!
        self.water_q_range_after_idx = int(np.where(self.q_val > WaterPeakEndQ)[0][0])

        print("( water] {0} is in {1}th index ~ {2} is in {3}th index )".format(self.q_val[self.water_q_range_start_idx],
                                                                         self.water_q_range_start_idx,
                                                                         self.q_val[self.water_q_range_after_idx],
                                                                         self.water_q_range_after_idx))

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

    def pair_range_q_idx_calc(self):
        # set idx range for pairing
        PairStartQ = 1.5
        PairEndQ = 3.5
        if len(self.q_val) == 0:
            print("no q value now!")
        self.pair_q_range_start_idx = int(np.where(self.q_val >= PairStartQ)[0][0])
        # this index is not included in water q range!!!
        self.pair_q_range_after_idx = int(np.where(self.q_val > PairEndQ)[0][0])

        print("( pairing] {0} is in {1}th index ~ {2} is in {3}th index )".format(self.q_val[self.pair_q_range_start_idx],
                                                                         self.pair_q_range_start_idx,
                                                                         self.q_val[self.pair_q_range_after_idx],
                                                                         self.pair_q_range_after_idx))

    def ice_range_q_idx_calc(self):
        # set idx range for pairing
        IcePeakStartQ = 1.64
        IcePeakEndQ = 1.66
        if len(self.q_val) == 0:
            print("no q value now!")
        self.ice_q_range_start_idx = int(np.where(self.q_val >= IcePeakStartQ)[0][0])
        # this index is not included in water q range!!!
        self.ice_q_range_after_idx = int(np.where(self.q_val > IcePeakEndQ)[0][0])

        print("( ice peak] {0} is in {1}th index ~ {2} is in {3}th index )".format(self.q_val[self.ice_q_range_start_idx],
                                                                         self.ice_q_range_start_idx,
                                                                         self.q_val[self.ice_q_range_after_idx],
                                                                         self.ice_q_range_after_idx))

    def set_file_name_and_read_tth(self):
        print("now run list ", self.runList)

        first_run_img_file_dir = self.FileCommonRoot + "run_{0:05d}_DIR/eh1rayMXAI_int/".format(self.runList[0])
        delay_names = os.listdir(first_run_img_file_dir)
        print("now delay num : ", len(delay_names))
        now_delay_num = len(delay_names)
        self.now_run_delay_num = now_delay_num

        print("set file name")
        # set file name
        for idx in range(now_delay_num):
            temp_name_int = "eh1rayMXAI_int/001_001_%03d" % (idx + 1)
            self.intensity_file_names.append(temp_name_int)
        # open only one twotheta data (since all value is same)
        self.twotheta_file_name = 'eh1rayMXAI_tth/001_001_001'
        # for idx in range(len(self.q_val)):
        #     print(self.q_val[idx])

        for each_run_num in self.runList:
            now_file_dir = self.FileCommonRoot + "run_{0:05d}_DIR/".format(each_run_num)
            self.each_run_file_dir.append(now_file_dir)
            temp_int_files = []
            for idx in range(now_delay_num):
                now_int_path = now_file_dir + self.intensity_file_names[idx] + ".h5"
                temp_int_file = h5.File(now_int_path, 'r')
                temp_int_files.append(temp_int_file)
            self.intensity_files.append(temp_int_files)

        self.read_twotheta_value()
        self.water_range_q_idx_calc()
        self.norm_range_q_idx_calc()
        self.pair_range_q_idx_calc()
        self.ice_range_q_idx_calc()

    def plot_water_sum_dist(self, each_run_plot=False, sum_file_out=False):
        print("going to plot water sum dist")

        print_criteria = 10

        water_range_int_sum_list = []
        ice_range_int_sum_list = []
        for idx_run, each_run_int_files in enumerate(self.intensity_files):
            now_int_file_num = len(each_run_int_files)
            now_water_sum_list = []
            now_ice_sum_list = []
            now_fileout_list = []
            for idx_file, each_int_file in enumerate(each_run_int_files):
                now_int_keys = list(each_int_file.keys())
                for each_key in now_int_keys:
                    now_int_val = np.array(each_int_file[each_key])
                    now_int_water_sum = sum(now_int_val[self.water_q_range_start_idx:self.water_q_range_after_idx])
                    now_int_ice_sum = sum(now_int_val[self.ice_q_range_start_idx:self.ice_q_range_after_idx])
                    now_water_sum_list.append(now_int_water_sum)
                    now_ice_sum_list.append(now_int_ice_sum)
                    now_file_out = [each_key, now_int_water_sum, now_int_ice_sum]
                    now_fileout_list.append(now_file_out)
                if (idx_file + 1) % print_criteria == 0:
                    print("read {0} / {1} file".format(idx_file + 1, now_int_file_num))
            water_range_int_sum_list.append(now_water_sum_list)
            ice_range_int_sum_list.append(now_ice_sum_list)
            # TODO : add ice range plot
            print("end for run{} files".format(self.runList[idx_run]))

            if sum_file_out:
                now_fileout_list = np.array(now_fileout_list)
                now_save_file_root = "../results/anisotropy/run" + str(self.runList[idx_run]) + "_watersum"
                np.save(now_save_file_root, now_fileout_list)
                print("successful file out :", now_save_file_root)

        merge_water_sum_list = np.array(water_range_int_sum_list).reshape(-1)
        merge_ice_sum_list = np.array(ice_range_int_sum_list).reshape(-1)
        print(merge_water_sum_list.shape, merge_ice_sum_list.shape)
        # plot_sum_for_criteria(merge_water_sum_list, "water range sum view of all run")
        hist_water_sum, bin_edges = np.histogram(water_range_int_sum_list, bins=200)
        hist_ice_sum, ice_bin_edges = np.histogram(ice_range_int_sum_list, bins=200)
        self.hist_fileout(hist_water_sum, bin_edges, file_name="../results/water_sum_hist.dat", runList=self.runList)
        self.hist_fileout(hist_ice_sum, ice_bin_edges, file_name="../results/ice_sum_hist.dat", runList=self.runList)

        # temp value
        self.UpperBoundOfNotWater = self.eachRunInfo[0][1]
        self.LowerBoundOfWater = self.eachRunInfo[0][2]
        self.WaterOutlierLowerBound = self.eachRunInfo[0][3]

        plot_sum_for_criteria(merge_water_sum_list, "water range sum view of all run with criteria", self.LowerBoundOfWater,self.UpperBoundOfNotWater, self.WaterOutlierLowerBound)

        LowerBoundOfIce = 150
        UpperBoundOfNotIce = 120
        IceOutlierLowerBound = 450
        plot_sum_for_criteria(merge_ice_sum_list, "ice peak sum view of all run with criteria",
                              LowerBoundOfIce, UpperBoundOfNotIce, IceOutlierLowerBound)

        if each_run_plot:
            for idx, each_sum_list in enumerate(water_range_int_sum_list):
                run_name = "run" + str(self.runList[idx])
                plot_sum_for_criteria(each_sum_list, "water range sum view of " + run_name + " with criteria",
                                      self.LowerBoundOfWater, self.UpperBoundOfNotWater, self.WaterOutlierLowerBound)

    def find_peak_pattern(self, incr_dist_plot=False, plot_outlier=False, sum_file_out=False):

        plot_within_range = False
        rm_ice_within_range = True
        plot_each_outlier = False
        plot_small_ice_test_fail = False
        plot_incr_test_pass = False

        print("going to find strange peak pattern")
        start_time = datetime.datetime.now()
        print(start_time)
        # TODO : need to change
        inc_max_outlier_boundary = 100
        inc_range_min = 100
        inc_range_max = 20
        prominence_max_val = 0
        prominence_min_val = 0
        each_outlier_plot_list = [12, 16]
        incr_test_pass_info = []
        incr_test_pass_int = []
        incr_max_within_range_info = []
        incr_max_within_range_int = []
        small_ice_test_fail_int = []
        small_ice_test_fail_info = []
        small_ice_test_fail_all_delay = []
        small_ice_plot_start_delay = 1
        small_ice_plot_end_delay = 10
        num_small_ice_peaks_all_delay = []
        num_small_ice_peak_each_delay = 0
        print_criteria = 10
        outlier_find_q_start = 100
        outlier_find_q_finish = 750

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
                if self.shorten_img:
                    now_int_keys = now_int_keys[0:self.num_shorten_img]
                #num_small_ice_peak_each_delay = 0
                now_delay_small_ice_fail_int = []
                for each_key in now_int_keys:
                    # if each_key == '1621421649.9924772_9120' and self.runNum == 69:
                    #     continue
                    now_int_diff_avg = []
                    now_int_val = np.array(each_int_file[each_key])
                    now_int_water_sum = sum(now_int_val[self.water_q_range_start_idx:self.water_q_range_after_idx])
                    now_int_next = np.roll(now_int_val, 1)
                    now_int_next[0] = now_int_next[1] # remove last element
                    now_int_incr = np.abs(now_int_val - now_int_next)
                    now_incr_max = np.max(now_int_incr[outlier_find_q_start:outlier_find_q_finish])
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
                            num_small_ice_peaks, properties = scipy.find_peaks(now_int_incr[self.water_q_range_start_idx:self.water_q_range_after_idx], prominence= (prominence_min_val, prominence_max_val))
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
                                #diff_test_fail_info.append([self.runList[idx_run], idx_file, now_pulseID, temp_q_idx_of_diff_max])
                    else:
                        if now_incr_max > inc_max_outlier_boundary:
                            incr_outlier_int.append(now_int_val)
                            incr_outlier_info.append([self.runList[idx_run], idx_file, each_key])
                            self.strange_peak_key_blacklist.append(each_key)
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

        plot_sum_for_criteria(merge_water_sum_list, "water range sum view of all run with criteria", self.LowerBoundOfWater,
                              self.UpperBoundOfNotWater, self.WaterOutlierLowerBound)

        if len(incr_outlier_info) != 0:
            if incr_dist_plot:
                plot_sum_for_criteria(incr_max_list, "incr max value of run {}".format(incr_outlier_info[0][0]), inc_max_outlier_boundary)

            if plot_outlier:
                one_graph_plot = 10
                if plot_within_range:
                    for idx_outlier, each_int in enumerate(incr_max_within_range_int):
                        now_label = "run{0}-delay{1}-{2}".format(incr_max_within_range_info[idx_outlier][0], incr_max_within_range_info[idx_outlier][1] + 1,idx_outlier)
                        print("{0}-th outlier : run{1}-delay{2}-key{3}".format(idx_outlier, incr_max_within_range_info[idx_outlier][0], incr_max_within_range_info[idx_outlier][1] + 1, incr_max_within_range_info[idx_outlier][2]))
                        #plt.plot(each_int, label=now_label)
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
                        now_label = "run{0}-delay{1}-{2}".format(incr_outlier_info[idx_outlier][0], incr_outlier_info[idx_outlier][1] + 1,idx_outlier)
                        print("{0}-th outlier : run{1}-delay{2}-key{3}".format(idx_outlier, incr_outlier_info[idx_outlier][0], incr_outlier_info[idx_outlier][1] + 1, incr_outlier_info[idx_outlier][2]))
                        #plt.plot(each_int, label=now_label)
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
                            now_label = "run{0}-delay{1}-{2}".format(incr_outlier_info[idx_outlier][0],incr_outlier_info[idx_outlier][1] + 1, idx_outlier)
                            print("{0}-th outlier : run{1}-delay{2}-key{3}".format(idx_outlier, incr_outlier_info[idx_outlier][0],incr_outlier_info[idx_outlier][1] + 1,incr_outlier_info[idx_outlier][2]))
                            #plt.plot(each_int, label=now_label)
                            plt.plot(each_int, marker='.', markersize=1, label=now_label)
                            plt.title("outlier of run" + str(incr_outlier_info[0][0]))
                            plt.legend()
                            plt.show()
                            idx_each_outlier += 1
                        else:
                            continue
                    except:
                        continue

            elif plot_small_ice_test_fail:
                one_graph_plot = 10
                tot_num_small_ice_outlier_early_delay = 0
                for idx_delay in range(small_ice_plot_end_delay - small_ice_plot_start_delay):
                    idx_delay = idx_delay + small_ice_plot_start_delay
                    if idx_delay != 0:
                        tot_num_small_ice_outlier_early_delay = num_small_ice_peaks_all_delay[idx_delay -1]
                        for idx_outlier in range(num_small_ice_peaks_all_delay[idx_delay] - num_small_ice_peaks_all_delay[idx_delay - 1]):
                            now_label = "run{0}-delay{1}-{2}".format(small_ice_test_fail_info[idx_delay][0],small_ice_test_fail_info[tot_num_small_ice_outlier_early_delay][1] + 1, tot_num_small_ice_outlier_early_delay + idx_outlier)
                            plt.plot(small_ice_test_fail_all_delay[idx_delay][idx_outlier], marker='.', markersize=1, label=now_label)
                            if idx_outlier % one_graph_plot == (one_graph_plot - 1):
                                plt.title("Small ice test failure of run" + str(small_ice_test_fail_info[0][0]))
                                plt.legend()
                                plt.show()
                        if (num_small_ice_peaks_all_delay[idx_delay] - num_small_ice_peaks_all_delay[idx_delay - 1]) % one_graph_plot != 0:
                            plt.title("Small ice test failure of run" + str(small_ice_test_fail_info[0][0]))
                            plt.legend()
                            plt.show()

                    else:
                        for idx_outlier in range(num_small_ice_peaks_all_delay[idx_delay]):
                            now_label = "run{0}-delay{1}-{2}".format(small_ice_test_fail_info[idx_delay][0],small_ice_test_fail_info[tot_num_small_ice_outlier_early_delay][1] + 1, tot_num_small_ice_outlier_early_delay + idx_outlier)
                            plt.plot(small_ice_test_fail_all_delay[idx_delay][idx_outlier], marker='.', markersize=1, label=now_label)
                            if idx_outlier % one_graph_plot == (one_graph_plot - 1):
                                plt.title("Small ice test failure of run" + str(small_ice_test_fail_info[0][0]))
                                plt.legend()
                                plt.show()
                        if (num_small_ice_peaks_all_delay[idx_delay] - num_small_ice_peaks_all_delay[idx_delay - 1]) % one_graph_plot != 0:
                            plt.title("Small ice test failure of run" + str(small_ice_test_fail_info[0][0]))
                            plt.legend()
                            plt.show()

            elif plot_incr_test_pass:
                one_graph_plot = 10
                for idx_pass, each_int in enumerate(incr_test_pass_int):
                    #now_label = "run{0}-delay{1}-{2}-q_idx {3}-key {4}".format(diff_test_fail_info[idx_outlier][0],diff_test_fail_info[idx_outlier][1] + 1, idx_outlier, where_max_diff_test_fail[idx_outlier],diff_test_fail_info[idx_outlier][2])
                    now_label = "run{0}-delay{1}-{2}".format(incr_test_pass_info[idx_pass][0],incr_test_pass_info[idx_pass][1] + 1, idx_pass)
                    #print("{0}-th outlier : run{1}-delay{2}-key{3}".format(idx_outlier, diff_test_fail_info[idx_outlier][0], diff_test_fail_info[idx_outlier][1] + 1,diff_test_fail_info[idx_outlier][2]))
                    # plt.plot(each_int, label=now_label)
                    plt.plot(each_int, marker='.', markersize=1, label=now_label)
                    if idx_pass % one_graph_plot == (one_graph_plot - 1):
                        plt.title("Increment test pass of run" + str(incr_test_pass_info[0][0]))
                        plt.legend()
                        plt.show()
                plt.title("Increment test pass of run" + str(incr_test_pass_info[0][0]))
                plt.legend()
                plt.show()

            print("now remove {0} strange peak at run{1}, {2} shots are removed by small ice test".format(len(incr_outlier_int), incr_outlier_info[0][0], len(small_ice_test_fail_int)))
            #np.savetxt('Difference test failure of run 50.txt', diff_test_fail_info, fmt='%s', delimiter='/')

    @staticmethod
    def hist_fileout(hist, bin_edges, file_name, runList):
        outFp = open(file_name, 'w')
        outFp.write("run List : " + str(runList) + "\n")
        outFp.write("bin_edge_left\tfrequency\n")
        for idx, hist_val in enumerate(hist):
            outFp.write("{}\t{}\n".format(bin_edges[idx], hist_val))
        outFp.close()
        print("histogram file out : ", file_name)

    def read_intensity_only_water(self, num_negative_delay = 5, np_file_out=False, rm_vapor=False, plot_watersum_pass=False, I0_normalize=False):
        """
        execute one of (this function and plot_water_sum_dist)
        two function have overlapped feature
        """
        print("read intensity files and save water range data")
        self.each_run_int_val_list = []
        print_criteria = 10
        watersum_test_pass_info = []
        watersum_test_pass_int_all_delay = []
        watersum_test_pass_info_negative_delay = []
        num_watersum_pass_each_delay = []
        num_watersum_pass_in_each_delay = 0
        watersum_file_out = True
        now_norm_delay_int_sum = []
        now_delay_int_sum = []
        temp_I0_dist = []

        self.UpperBoundOfNotWater = self.eachRunInfo[0][1]
        self.LowerBoundOfWater = self.eachRunInfo[0][2]
        self.WaterOutlierLowerBound = self.eachRunInfo[0][3]

        if len(self.strange_peak_key_blacklist) == 0:
            print("strange peak blacklist is empty!")

        if I0_normalize:
            temp_now_run_I0 = []
            now_run_I0 = []
            # now_I0_common_path = "/data/exp_data/PAL-XFEL_20210514/rawdata/run_{0:05d}_00001_DIR/eh2qbpm1_totalsum/".format(self.runNum)
            now_I0_common_path = "/data/exp_data/PAL-XFEL_20210514/rawdata/run_{0:05d}_DIR/ohqbpm2_totalsum/".format(self.runNum)
            for delay_file_name in self.intensity_file_names:
                now_delay_I0_path = now_I0_common_path + delay_file_name + ".h5"
                temp_I0_file = h5.File(now_delay_I0_path, 'r')
                temp_now_run_I0.append(temp_I0_file)
            for idx_delay, each_delay_key_file in enumerate(temp_now_run_I0):
                now_run_I0.append(each_delay_key_file)

        for idx_run, each_run_int_files in enumerate(self.intensity_files):
            now_int_file_num = len(each_run_int_files)
            now_run_int_val_list = []
            temp_int_arr = []
            for idx_file, each_int_file in enumerate(each_run_int_files):
                if I0_normalize:
                    now_delay_I0 = now_run_I0[idx_file]
                now_int_keys = list(each_int_file.keys())
                if self.shorten_img:
                    now_int_keys = now_int_keys[0:self.num_shorten_img]
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
                        if I0_normalize:
                            now_I0 = now_delay_I0[each_key]
                            now_I0 = now_I0[()]*0.66
                            temp_I0_dist.append(now_I0[()])
                        now_int_obj = PulseDataLight(now_int_val, each_key,
                                                     self.water_q_range_start_idx, self.water_q_range_after_idx,
                                                     self.norm_q_range_start_idx, self.norm_q_range_after_idx,
                                                     self.pair_q_range_start_idx, self.pair_q_range_after_idx)
                        now_pulseID = re.findall("(.*)\.(.*)_(.*)", each_key)[0][2]
                        now_pulseID = int(now_pulseID)
                        now_int_obj.check_laser_onoff(now_pulseID)
                        now_delay_int_val_list.append(now_int_obj)
                        temp_obj = now_int_obj
                        temp_obj.norm_given_range()
                        temp_int_arr.append(temp_obj.intensity_val)
                        # self.q_interpolate(1.5, 3.5, self.q_val, I0_normalized_int)
                        watersum_test_pass_info.append([self.runList[idx_run], idx_file, each_key, num_watersum_pass_in_each_delay])
                        if idx_file < num_negative_delay:
                            watersum_test_pass_info_negative_delay.append([self.runList[idx_run], idx_file, each_key, num_watersum_pass_in_each_delay])
                        # now_delay_watersum_pass_list.append(np.sum(now_temp_obj.intensity_val))
                        # now_delay_watersum_pass_list.append(norm_range_sum)
                        if I0_normalize:
                            I0_normalized_int = (now_int_val / now_I0) / 1E7
                            now_norm_delay_int_sum.append(I0_normalized_int)
                            now_delay_watersum_pass_list.append(I0_normalized_int)
                        else:
                            # temp_obj.norm_given_range()
                            # now_norm_delay_int_sum.append(np.sum(temp_obj.intensity_val[self.water_q_range_start_idx:self.water_q_range_after_idx]))  # for q 1.5~3.5 sum
                            # now_delay_int_sum.append(np.sum(now_int_sum))
                            now_delay_watersum_pass_list.append(temp_obj.intensity_val)
                        num_watersum_pass_in_each_delay += 1
                watersum_test_pass_int_all_delay.append(now_delay_watersum_pass_list)
                num_watersum_pass_each_delay.append(num_watersum_pass_in_each_delay)
                now_run_int_val_list.append(now_delay_int_val_list)
                if (idx_file + 1) % print_criteria == 0:
                    print("read {0} / {1} file".format(idx_file + 1, now_int_file_num))
            self.each_run_int_val_list.append(now_run_int_val_list)
            # plot_sum_for_criteria(temp_int_arr, "water range sum view of all run with criteria",self.LowerBoundOfWater, self.UpperBoundOfNotWater, self.WaterOutlierLowerBound)
            # np.save("/data/exp_data/myeong0609/azi_intg_test/watersum_test_run{}.npy".format(self.runNum), watersum_test_pass_int_all_delay)
            if I0_normalize:
                normalized_int_avg = np.average(now_norm_delay_int_sum, axis=0)
                hit_int_mean = np.sum(normalized_int_avg[self.water_q_range_start_idx:self.water_q_range_after_idx])
                # hit_int_mean = np.average(now_delay_int_sum)
                # hit_int_median = np.median(now_delay_int_sum)
                print("Average intensity after I0 normalization : {0}".format(hit_int_mean))
                # print("mean intensity : {0}, median value : {1}".format(hit_int_mean, hit_int_median))
            # else:
                # normalized_int_avg = np.average(now_norm_delay_int_sum, axis=0)
                # before_normalize_int_avg = np.average(now_delay_int_sum)
                # print("Before normalization intensity : {0}, after normalization : {1}".format(before_normalize_int_avg,normalized_int_avg ))

            #self.each_run_int_val_list.extend(now_run_int_val_list)
            print("end intensity read for run{} files".format(self.runList[idx_run]))

        if plot_watersum_pass:
            one_graph_plot = 10
            num_plot_one_delay = 0
            idx_pass_previous = 0
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
                    plt.plot(self.q_val, watersum_test_pass_int_all_delay[idx_delay][num_watersum_pass_idx], marker='.', markersize=1, label=now_label)
                    if num_watersum_pass_idx % one_graph_plot == (one_graph_plot - 1) :
                        plt.title("Shots of pass watersum test of run")# + str(watersum_test_pass_info[0][0]))
                        #plt.legend()
                        plt.xlim(0.3, 4.5)
                        plt.show()
                    # plt.title("Shots of pass watersum test of run" + str(watersum_test_pass_info[0][0]))
                    # plt.legend()
                    # plt.xlim(130, 350)
                    # plt.show()

                # plt.title("Shots of pass watersum test of run" + str(watersum_test_pass_info[0][0]))
                # plt.legend()
                # plt.show()

        print("now {} shots remain after watersum test at run{}".format(len(watersum_test_pass_info), watersum_test_pass_info[0][0]))
        if watersum_file_out:
            temp_save_name_negative_delay = "../results/each_run_watersum_int/watersum_test_pass_negative_delay_pulse_info_" + self.file_common_name + ".npy"
            print("save as :", temp_save_name_negative_delay)
            np.save(temp_save_name_negative_delay, watersum_test_pass_info_negative_delay)
            self.watersum_pass_negative_delay = watersum_test_pass_info_negative_delay
            temp_save_name_watersum_pass = "../results/each_run_watersum_int/watersum_test_pass_" + self.file_common_name + ".npy"
            # print("save as :", temp_save_name_watersum_pass)
            # np.save(temp_save_name_watersum_pass, self.each_run_int_val_list[0], allow_pickle=True)


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

    def pairwise_diff_calc(self, num_real_neg_delay = 5, run_list_to_merge=[], merge_multi_run=False, expand_negative_pool=False, test_plot=False, test_plot_num=5, fileout_pair_info=True):
        plot_start_idx = 10
        plot_end_idx = -120
        print("plot_start / end", plot_start_idx, plot_end_idx)

        laser_off_shot_int_list = []
        laser_off_shot_all_delay = []
        laser_off_shot_info_list = []
        num_laser_off_each_delay = []

        if merge_multi_run:
            for delay_idx, each_run_int_list in enumerate(self.each_run_int_val_list):
                now_run_diff_list = []
                now_pair_arr = []
                if expand_negative_pool:
                    for delay_idx, each_delay_int_list in enumerate(each_run_int_list):
                        len_of_laser_off_all_delay = 0
                        laser_on_idx, laser_off_idx = extract_laser_on_off_list_only_water(each_delay_int_list,len_of_laser_off_all_delay, expand_negative_pool)
                        for pulse_idx in laser_off_idx:
                            laser_off_shot_int_list.append(each_delay_int_list[pulse_idx])
                            laser_off_shot_all_delay.append(each_delay_int_list[pulse_idx].key)
                            laser_off_shot_info_list.append([delay_idx, each_delay_int_list[pulse_idx].key])
                    len_of_laser_off_all_delay = len(laser_off_shot_int_list)
                    for each_delay_int_list in each_run_int_list:
                        early_delay_key_list = []
                        num_each_delay_laser_on = []
                        num_each_delay_laser_off = []
                        temp_each_delay_int_list = []
                        laser_on_idx, laser_off_idx = extract_laser_on_off_list_only_water(each_delay_int_list,len_of_laser_off_all_delay, expand_negative_pool)
                        laser_off_idx = []
                        laser_off_idx = np.array(range(len(each_delay_int_list), len_of_laser_off_all_delay))
                        num_each_delay_laser_off.append(laser_off_idx)
                        num_each_delay_laser_on.append(laser_on_idx)
                        for idx in range(len_of_laser_off_all_delay):
                            each_delay_int_list.append(laser_off_shot_int_list[idx])
                        print("pairing with nearest integrated intensity logic")
                        nearest_int_pair = match_water_near_intensity_pair(each_delay_int_list, laser_on_idx, laser_off_idx, expand_negative_pool)
                        print(nearest_int_pair)
                        if fileout_pair_info:
                            pair_pulseID_arr = self.make_pulseID_array(each_delay_int_list, nearest_int_pair)
                            now_pair_arr.append(pair_pulseID_arr)
                        for each_data in each_delay_int_list:
                            if not each_data.is_normalized:
                                each_data.norm_given_range()
                        near_int_pair_diff = self.calc_near_int_pair_diff(each_delay_int_list, nearest_int_pair)
                        now_run_diff_list.append(near_int_pair_diff)
                else:
                    for idx, each_delay_int_list in enumerate(each_run_int_list):
                        num_each_delay_laser_on = []
                        num_each_delay_laser_off = []
                        len_of_original_each_dealy_int_list = 0
                        laser_on_idx, laser_off_idx = extract_laser_on_off_list_only_water(each_delay_int_list, len_of_original_each_dealy_int_list, expand_negative_pool)
                        num_each_delay_laser_off.append(laser_off_idx)
                        num_each_delay_laser_on.append(laser_on_idx)
                        print("pairing with nearest integrated intensity logic")
                        nearest_int_pair = match_water_near_intensity_pair(each_delay_int_list, laser_on_idx, laser_off_idx)
                        #print(nearest_int_pair)

                        if fileout_pair_info:
                            pair_pulseID_arr = self.make_pulseID_array(each_delay_int_list, nearest_int_pair)
                            now_pair_arr.append(pair_pulseID_arr)
                        for each_data in each_delay_int_list:
                            if not each_data.is_normalized:
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
        else:
            for each_run_int_list in self.each_run_int_val_list:
                now_run_diff_list = []
                now_pair_arr = []
                temp_each_delay_int_list = []
                if expand_negative_pool:
                    for delay_idx, each_delay_int_list in enumerate(each_run_int_list):
                        len_of_laser_off_all_delay = 0
                        laser_on_idx, laser_off_idx = extract_laser_on_off_list_only_water(each_delay_int_list,len_of_laser_off_all_delay, expand_negative_pool)
                        for pulse_idx in laser_off_idx:
                            laser_off_shot_int_list.append(each_delay_int_list[pulse_idx])
                            laser_off_shot_all_delay.append(each_delay_int_list[pulse_idx].key)
                            laser_off_shot_info_list.append([self.runList[0], delay_idx, each_delay_int_list[pulse_idx].key])
                    len_of_laser_off_all_delay = len(laser_off_shot_int_list)
                    for each_delay_int_list in each_run_int_list:
                        early_delay_key_list = []
                        num_each_delay_laser_on = []
                        num_each_delay_laser_off = []
                        temp_each_delay_int_list = []
                        laser_on_idx, laser_off_idx = extract_laser_on_off_list_only_water(each_delay_int_list, len_of_laser_off_all_delay, expand_negative_pool)
                        laser_off_idx = []
                        #temp_laser_off_idx = np.array(range(len_of_laser_off_all_delay))
                        laser_off_idx = np.array(range(len(each_delay_int_list), len_of_laser_off_all_delay))
                        num_each_delay_laser_off.append(laser_off_idx)
                        num_each_delay_laser_on.append(laser_on_idx)
                        for idx in range(len_of_laser_off_all_delay):
                            each_delay_int_list.append(laser_off_shot_int_list[idx])
                        print("pairing with nearest integrated intensity logic")
                        nearest_int_pair = match_water_near_intensity_pair(each_delay_int_list,laser_on_idx,laser_off_idx,expand_negative_pool)
                        print(nearest_int_pair)
                        if fileout_pair_info:
                            pair_pulseID_arr = self.make_pulseID_array(each_delay_int_list, nearest_int_pair)
                            now_pair_arr.append(pair_pulseID_arr)
                        for each_data in each_delay_int_list:
                            if not each_data.is_normalized:
                                each_data.norm_given_range()
                        near_int_pair_diff = self.calc_near_int_pair_diff(each_delay_int_list, nearest_int_pair)
                        now_run_diff_list.append(near_int_pair_diff)
                else:
                    norm_off_int_list = []
                    norm_off_avg_int_list = []
                    for idx, each_delay_int_list in enumerate(each_run_int_list):
                        # each_delay_int_list = each
                        num_each_delay_laser_on = []
                        num_each_delay_laser_off = []
                        len_of_original_each_dealy_int_list = 0
                        laser_on_idx, laser_off_idx = extract_laser_on_off_list_only_water(each_delay_int_list,len_of_original_each_dealy_int_list ,expand_negative_pool)
                        num_each_delay_laser_off.append(laser_off_idx)
                        num_each_delay_laser_on.append(laser_on_idx)

                        for pulse_idx in laser_off_idx:
                            laser_off_shot_int_list.append(each_delay_int_list[pulse_idx].intensity_val)
                        print("pairing with nearest integrated intensity logic")
                        if self.on_on_off_test:
                            target_idx = int(np.round(len(laser_off_idx)/2))
                            laser_off_idx = laser_off_idx[0:target_idx]
                        nearest_int_pair = match_water_near_intensity_pair(each_delay_int_list, laser_on_idx, laser_off_idx)
                        if fileout_pair_info:
                            pair_pulseID_arr = self.make_pulseID_array(each_delay_int_list, nearest_int_pair)
                            now_pair_arr.append(pair_pulseID_arr)
                        for each_data in each_delay_int_list:
                            if not each_data.is_normalized:
                                each_data.norm_given_range()
                        for off_idx in laser_off_idx:
                            norm_off_int_list.append(each_delay_int_list[off_idx].norm_range_sum)
                        norm_off_avg_int_list.append(np.average(norm_off_int_list))
                        near_int_pair_diff = self.calc_near_int_pair_diff(each_delay_int_list, nearest_int_pair)
                        now_run_diff_list.append(near_int_pair_diff)
                print("This run's average normalized range sum of laser off shots is : {0}".format(np.average(norm_off_avg_int_list)))
                self.whole_run_diff_list.append(now_run_diff_list)
                if fileout_pair_info:
                    self.whole_run_pair_list.append(now_pair_arr)
                # temp_int_arr = []
                # for each_int in each_delay_int_list:
                #     temp_int_arr.append(each_int.intensity_val)
                # plot_sum_for_criteria(temp_int_arr, "water range sum view of all run with criteria",self.LowerBoundOfWater, self.UpperBoundOfNotWater, self.WaterOutlierLowerBound)

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
        if merge_multi_run:
            if expand_negative_pool:
                temp_save_name = "../results/each_run_watersum_int/laser_off_all_delay_list_run{0}_{1}.npy".format(run_list_to_merge[0],run_list_to_merge[1])  # TODO prepare for more than two runs
                print("save as :", temp_save_name)
                np.save(temp_save_name, laser_off_shot_info_list)
            else:
                temp_save_name = "../results/whole_run_diff/whole_run_diff_run{0}_{1}.npy".format(run_list_to_merge[0], run_list_to_merge[1]) #TODO prepare for more than two runs
                print("save as :", temp_save_name)
                np.save(temp_save_name, self.whole_run_diff_list)
        elif expand_negative_pool:
            temp_save_name = "../results/each_run_watersum_int/laser_off_all_delay_list_" + self.file_common_name + ".npy" #TODO prepare for more than two runs
            print("save as :", temp_save_name)
            np.save(temp_save_name, laser_off_shot_info_list)
            avg_all_off_shots = "../results/each_run_watersum_int/laser_off_all_delay_list_" + self.file_common_name + ".dat"
            print("save as : ", avg_all_off_shots)
            np.savetxt(avg_all_off_shots, avg_all_off_shots)
        else:
            temp_save_name = "../results/whole_run_diff/whole_run_diff_" + self.file_common_name + ".npy"
            print("save as :", temp_save_name)
            # np.save(temp_save_name, self.whole_run_diff_list)
            avg_all_off_shots_file_name = "../results/each_run_watersum_int/laser_off_all_delay_list_" + self.file_common_name + ".dat"
            avg_all_off_shots = np.average(laser_off_shot_int_list, axis=0)
            np.savetxt(avg_all_off_shots_file_name, avg_all_off_shots)
            np.savetxt("../results/each_run_watersum_int/q_val_" + self.file_common_name + ".dat", self.q_val)

    def fileout_pair_info_only(self, pair_file_out=False, diff_calc_test=False, test_plot=False, test_plot_num=5):
        for idx_run, each_run_int_list in enumerate(self.each_run_int_val_list):
            now_run_diff_list = []
            now_pair_arr = []
            for each_delay_int_list in each_run_int_list:
                laser_on_idx, laser_off_idx = extract_laser_on_off_list_only_water(each_delay_int_list)
                print("pairing with nearest integrated intensity logic")
                nearest_int_pair = match_water_near_intensity_pair(each_delay_int_list, laser_on_idx, laser_off_idx)
                pair_pulseID_arr = self.make_pulseID_array(each_delay_int_list, nearest_int_pair)
                now_pair_arr.append(pair_pulseID_arr)
                if diff_calc_test:
                    for each_data in each_delay_int_list:
                      each_data.norm_given_range()
                    near_int_pair_diff = self.calc_near_int_pair_diff(each_delay_int_list, nearest_int_pair)
                    now_run_diff_list.append(near_int_pair_diff)

            if pair_file_out:
                now_pair_arr = np.array(now_pair_arr, dtype=object)
                now_save_file_root = "../results/anisotropy/run" + str(self.runList[idx_run]) + "_pairinfo"
                np.save(now_save_file_root, now_pair_arr)
                print("successful file out :", now_save_file_root)

            if diff_calc_test:
                self.whole_run_diff_list.append(now_run_diff_list)

            if diff_calc_test and test_plot:
                # plot_start_idx = 10
                # plot_end_idx = -120
                # print("plot_start / end", plot_start_idx, plot_end_idx)

                for delay_idx in range(test_plot_num):
                    for idx in range(test_plot_num):
                        plt.plot(self.q_val, now_run_diff_list[delay_idx][idx], label=(str(idx + 1) + 'th diff '))
                    plt.title("test draw each diff - nearest intensity of delay" + str(delay_idx + 1))
                    plt.xlabel('q value')
                    plt.ylabel("intensity")
                    plt.legend()
                    plt.show()

        self.each_run_int_val_list = []
        print("remove each_run_int_val_list.")

    def additional_process_diff(self, merge_run_list=[], merge_multi_run=False, show_before_cutted=False, show_after_cutted=False, file_out=False, svd_with_cut=False, fileout_pair_info=False, plot_difference=False, plot_azimuthal=False):
        intg_start_idx = 140
        intg_end_idx = 348
        cutoff_criteria = [1E5 for _ in range(100)]

        #z = 1.65 #z for 95% cutoff
        z = 1.88 # z for standard normal distribution for 97% cutoff
        #z = 4.0 for 100%
        #z = 2.33 #for 99% cutoff

        test_plot_num = 60

        whole_run_sum_list = []
        whole_run_cutted_avg = []
        now_run_pair_list = []
        for run_idx, each_run_diff_list in enumerate(self.whole_run_diff_list):
            now_run_sum_list = []
            now_run_cutted_avg_list = []
            now_run_cutted_diff = []
            now_run_cutted_pair = []
            if fileout_pair_info:
                now_run_pair_list = self.whole_run_pair_list[run_idx]
            for delay_idx, each_delay_diff_list in enumerate(each_run_diff_list):
                each_delay_diff_list = np.array(each_delay_diff_list)
                now_delay_sum_list = []
                for diff_data in each_delay_diff_list:
                    now_sum = np.sum(np.abs(diff_data[intg_start_idx:intg_end_idx]))
                    now_delay_sum_list.append(now_sum)
                now_delay_sum_list = np.array(now_delay_sum_list)
                avg_now_delay_sum = np.mean(now_delay_sum_list)
                std_now_delay_sum = np.std(now_delay_sum_list)
                now_run_sum_list.append(now_delay_sum_list)
                cutoff_criteria[delay_idx] = z * std_now_delay_sum + avg_now_delay_sum

                if show_before_cutted:
                    if merge_multi_run:
                        if delay_idx < test_plot_num:
                            now_graph_title = "integration hist of run{0}&{1}".format(merge_run_list[0], merge_run_list[1])  + "//" + str(delay_idx + 1) + "-th delay"
                            plot_sum_for_criteria(now_delay_sum_list, now_graph_title, v_line_1=cutoff_criteria[delay_idx])
                            print(now_graph_title)
                    else:
                        if delay_idx < test_plot_num:
                            now_graph_title = "integration hist of run" + str(self.runList[run_idx]) + "//" + str(delay_idx + 1) + "-th delay"
                            plot_sum_for_criteria(now_delay_sum_list, now_graph_title, v_line_1=cutoff_criteria[delay_idx])
                            print(now_graph_title)
                            now_graph_title = "Difference of each shot in " + str(delay_idx + 1) + "-th delay"
                            plt.plot()
                now_delay_cutted_diff = each_delay_diff_list[now_delay_sum_list < cutoff_criteria[delay_idx]]
                now_run_cutted_diff.append(np.array(now_delay_cutted_diff, dtype='object'))
                if plot_difference:
                    for pulse_idx in range(len(now_delay_cutted_diff)):
                        one_graph_plot = 10
                        if merge_multi_run:
                            now_label = "run{0}&{1}-delay{2}-{3}".format(merge_run_list[0], merge_run_list[1], delay_idx + 1, pulse_idx)
                        else:
                            now_label = "run{0}-delay{1}-{2}".format(self.runList[run_idx], delay_idx + 1, pulse_idx)
                        plt.plot(now_delay_cutted_diff[pulse_idx], label = now_label)
                        if pulse_idx % one_graph_plot == (one_graph_plot - 1):
                            plt.title("Difference of each shot in " + str(delay_idx + 1) + "-th delay")
                            plt.legend()
                            plt.xlim(75, 400)
                            plt.ylim(-5000, 5000)
                            plt.show()
                    plt.title("Difference of each shot in " + str(delay_idx + 1) + "-th delay")
                    plt.legend()
                    plt.xlim(75, 400)
                    plt.show()


                print("remove ", len(each_delay_diff_list[now_delay_sum_list >= cutoff_criteria[delay_idx]])," in", str(delay_idx + 1), "-th delay")

                if fileout_pair_info:
                    # 2d -> 1d index problem
                    now_delay_pair_list = now_run_pair_list[delay_idx]
                    now_delay_pair_list = np.transpose(now_delay_pair_list)
                    whole_cut = []
                    for each_list in now_delay_pair_list:
                        now_list_cut = each_list[now_delay_sum_list < cutoff_criteria[delay_idx]]
                        whole_cut.append(now_list_cut)
                    now_delay_left_pair_list = np.transpose(whole_cut)
                    now_run_cutted_pair.append(now_delay_left_pair_list)
                    print("left pair : ", len(now_delay_left_pair_list), " in", str(delay_idx + 1), "-th delay")

                now_delay_avg = np.average(now_delay_cutted_diff, axis=0)
                if now_delay_avg.shape is np.nan:
                    now_delay_avg = np.zeros_like(self.q_val)
                now_run_cutted_avg_list.append(now_delay_avg)

            whole_run_sum_list.append(now_run_sum_list)
            whole_run_cutted_avg.append(np.array(now_run_cutted_avg_list, dtype='object'))
            self.whole_run_cutted_diff_list.append(np.array(now_run_cutted_diff, dtype='object'))

            if plot_azimuthal:
                for delay_idx in range(len(now_run_cutted_avg_list)):
                    one_graph_plot = 10
                    if merge_multi_run:
                        now_label = "run{0}&{1}-delay{2}".format(merge_run_list[0], merge_run_list[1], delay_idx + 1)
                    else:
                        now_label = "run{0}-delay{1}".format(self.runList[run_idx], delay_idx + 1)
                    plt.plot(now_run_cutted_avg_list[delay_idx], label=now_label)
                    if delay_idx % one_graph_plot == (one_graph_plot - 1):
                        plt.title("Average of difference of each delay")
                        # plt.figsize(10,6)
                        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                        plt.axvline(x=50)
                        plt.axvline(x=430)
                        plt.tight_layout()
                        #plt.xlim(75, 400)
                        plt.figure(figsize=(8, 10))
                        plt.show()
                plt.title("Average of difference of each delay")
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                plt.axvline(x=50)
                plt.axvline(x=430)
                plt.tight_layout()
                #plt.xlim(75, 400)
                plt.show()

            if fileout_pair_info:
                if merge_multi_run:
                    now_pair_arr = np.array(now_run_cutted_pair, dtype=object)
                    now_save_file_root = "../results/anisotropy/run{0}_{1}".format(merge_run_list[0], merge_run_list[1]) + "_pairinfo"
                    np.save(now_save_file_root, now_pair_arr)
                    print("successful file out :", now_save_file_root)
                else:
                    now_pair_arr = np.array(now_run_cutted_pair, dtype=object)
                    now_save_file_root = "../results/anisotropy/run" + str(self.runList[run_idx]) + "_pairinfo"
                    np.save(now_save_file_root, now_pair_arr)
                    print("successful file out :", now_save_file_root)
        if merge_multi_run:
            # save whole_run_cutted_diff_list numpy array
            temp_save_name = "../results/whole_run_diff/whole_run_cutted_diff_run{0}_{1}".format(merge_run_list[0], merge_run_list[1]) + ".npy"
            print("save as :", temp_save_name)
            np.save(temp_save_name, self.whole_run_cutted_diff_list)
            self.whole_run_diff_list = []

            # save whole_run_cutted_avg numpy array
            temp_save_name = "../results/whole_run_avg/whole_run_cutted_avg_run{0}_{1}".format(merge_run_list[0], merge_run_list[1]) + ".npy"
            print("save as :", temp_save_name)
            np.save(temp_save_name, whole_run_cutted_avg)
        else:
            # save whole_run_cutted_diff_list numpy array
            temp_save_name = "../results/whole_run_diff/whole_run_cutted_diff_" + self.file_common_name + ".npy"
            # print("save as :", temp_save_name)
            np.save(temp_save_name, self.whole_run_cutted_diff_list)
            self.whole_run_diff_list = []

            # save whole_run_cutted_avg numpy array
            temp_save_name = "../results/whole_run_avg/whole_run_cutted_avg_" + self.file_common_name + ".npy"
            print("save as :", temp_save_name)
            np.save(temp_save_name, whole_run_cutted_avg)
            np.save("../results/whole_run_avg/q_val_" + self.file_common_name + ".npy", self.q_val)

    def read_I0_value(self):
        all_run_I0_dict_list = []
        for idx_run, _ in enumerate(self.runList):
            now_delay_num = len(self.intensity_files[idx_run])
            now_run_I0_dict_list = []
            for idx_delay in range(now_delay_num):
                now_delay_I0_dict = {}
                now_file_name_I0 = 'eh1qbpm1_totalsum/001_001_%03d' % (idx_delay + 1)
                # now_I0_path = self.each_run_file_dir[idx_run] + now_file_name_I0 + ".h5"
                now_I0_path = '/data/exp_data/PAL-XFEL_20210514/rawdata/' + 'run_{0:05d}_DIR/'.format(self.runList[0]) + now_file_name_I0 + ".h5"
                now_I0_file = h5.File(now_I0_path, 'r')
                now_I0_keys = now_I0_file.keys()
                for each_key in now_I0_keys:
                    now_I0_val = float(now_I0_file[each_key][()])
                    now_delay_I0_dict[each_key] = now_I0_val
                    # temp_pair = [each_key, now_I0_val]
                    # now_delay_I0_pair_list.append(temp_pair)
                now_run_I0_dict_list.append(now_delay_I0_dict)
            all_run_I0_dict_list.append(now_run_I0_dict_list)
        self.all_run_I0_dict_list = all_run_I0_dict_list

    def vapor_anal(self, incr_dist_plot=False, plot_outlier=False, sum_file_out=False, plot_vapor_avg=False):
        print("going to find strange peak pattern")
        inc_max_outlier_boundary = 50
        vapor_range_start = 5000
        vapor_range_end = 23000
        #vapor_range_end = 12000
        outlier_find_q_start = 50
        outlier_find_q_finish = 450

        print_criteria = 10
        self.read_I0_value()

        water_range_int_sum_list = []
        incr_max_list = []
        incr_outlier_int = []
        incr_outlier_info = []
        vapor_norm_int = []
        vapor_info = []
        for idx_run, each_run_int_files in enumerate(self.intensity_files):
            now_int_file_num = len(each_run_int_files)
            now_water_sum_list = []
            now_fileout_list = []
            for idx_file, each_int_file in enumerate(each_run_int_files):
                now_I0_dict = self.all_run_I0_dict_list[idx_run][idx_file]
                now_int_keys = list(each_int_file.keys())
                for each_key in now_int_keys:
                    now_int_val = np.array(each_int_file[each_key])
                    now_int_water_sum = sum(now_int_val[self.water_q_range_start_idx:self.water_q_range_after_idx])
                    now_int_next = np.roll(now_int_val, 1)
                    now_int_next[0] = 0  # remove last element
                    now_int_incr = np.abs(now_int_val - now_int_next)
                    now_incr_max = np.max(now_int_incr)
                    now_incr_max = np.max(now_int_incr[outlier_find_q_start:outlier_find_q_finish])
                    # now_incr_max = np.max(now_int_incr[self.water_q_range_start_idx:self.water_q_range_after_idx])
                    incr_max_list.append(now_incr_max)
                    if now_incr_max > inc_max_outlier_boundary:
                        incr_outlier_int.append(now_int_val)
                        incr_outlier_info.append([self.runList[idx_run], idx_file, each_key])
                        self.strange_peak_key_blacklist.append(each_key)
                    if vapor_range_start < now_int_water_sum < vapor_range_end:
                        # when data is only vapor signal
                        now_vapor_I0 = now_I0_dict[each_key]
                        try:
                            norm_vapor_int = np.divide(now_int_val, now_vapor_I0)
                        except ZeroDivisionError:
                            print("I0 is zero at run{} delay{} key{}".format(self.runList[idx_run], idx_file + 1, each_key))
                            norm_vapor_int = now_vapor_I0
                        vapor_norm_int.append(norm_vapor_int)
                        vapor_info.append([self.runList[idx_run], idx_file, each_key, now_vapor_I0])
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
            now_vapor_save_file_root = "../results/vapor_signal/run" + str(self.runList[idx_run]) + "_vapor.npy"
            np.save(now_vapor_save_file_root, vapor_info)
            print("successful file out of vapor signal:", now_vapor_save_file_root)


        merge_water_sum_list = np.array(water_range_int_sum_list).reshape(-1)
        print(merge_water_sum_list.shape)
        # plot_sum_for_criteria(merge_water_sum_list, "water range sum view of all run")
        hist_water_sum, bin_edges = np.histogram(water_range_int_sum_list, bins=200)
        self.hist_fileout(hist_water_sum, bin_edges, file_name="../results/water_sum_hist.dat", runList=self.runList)

        # temp value
        self.UpperBoundOfNotWater = self.eachRunInfo[0][1]
        self.LowerBoundOfWater = self.eachRunInfo[0][2]
        self.WaterOutlierLowerBound = self.eachRunInfo[0][3]

        plot_sum_for_criteria(merge_water_sum_list, "water range sum view of all run with criteria",self.LowerBoundOfWater, self.UpperBoundOfNotWater, self.WaterOutlierLowerBound, vapor_range_start, vapor_range_end)

        if len(self.strange_peak_key_blacklist) != 0:
            if incr_dist_plot:
                plot_sum_for_criteria(incr_max_list, "incr max value of run {}".format(incr_outlier_info[0][0]), inc_max_outlier_boundary)

            if plot_outlier:
                one_graph_plot = 10
                for idx_outlier, each_int in enumerate(incr_outlier_int):
                    now_label = "run{0}-delay{1}-{2}".format(incr_outlier_info[idx_outlier][0], incr_outlier_info[idx_outlier][1], idx_outlier)
                    print("{0}-th outlier : run{1}-delay{2}-key{3}".format(idx_outlier, incr_outlier_info[idx_outlier][0], incr_outlier_info[idx_outlier][1], incr_outlier_info[idx_outlier][2]))
                    plt.plot(each_int, label=now_label)
                    if idx_outlier % one_graph_plot == (one_graph_plot - 1):
                        plt.title("outlier of run" + str(incr_outlier_info[0][0]))
                        plt.legend()
                        plt.show()
                plt.title("outlier of run" + str(incr_outlier_info[0][0]))
                plt.legend()
                plt.show()
            print("now remove {} strange peak at run{}".format(len(incr_outlier_int), incr_outlier_info[0][0]))

        else:
            print("No strange peak!")
        avg_vapor_range = np.average(vapor_norm_int, axis=0)
        self.norm_vapor_int = avg_vapor_range
        print("{0} shots are collected as vapor".format(len(vapor_info)))
        if plot_vapor_avg:
            plt.title("I0 normalized vapor average!")
            plt.plot(self.q_val, avg_vapor_range)
            plt.show()

    @staticmethod
    def hist_fileout(hist, bin_edges, file_name, runList):
        outFp = open(file_name, 'w')
        outFp.write("run List : " + str(runList) + "\n")
        outFp.write("bin_edge_left\tfrequency\n")
        for idx, hist_val in enumerate(hist):
            outFp.write("{}\t{}\n".format(bin_edges[idx], hist_val))
        outFp.close()
        print("histogram file out : ", file_name)

    @staticmethod
    def calc_near_int_pair_diff(data_list, near_int_pair_idx):
        diff_int_arr = []
        for each_idx_a, each_idx_b in near_int_pair_idx:
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

            if not (laser_on_data.is_normalized and laser_off_data.is_normalized):
                # check whether both data is normalized
                print("normalization error! not normalized in index pair (", each_idx_a, ",", each_idx_b, ")")

            diff_int = np.array(laser_on_data.intensity_val) - np.array(laser_off_data.intensity_val)
            diff_int_arr.append(diff_int)

        return diff_int_arr

    @staticmethod
    def make_pulseID_array(data_list, near_int_pair_idx):
        pulseID_arr = []
        for each_idx_a, each_idx_b in near_int_pair_idx:
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

            now_data = [laser_on_data.key, laser_off_data.key]
            pulseID_arr.append(now_data)
        return pulseID_arr

    def find_peak_pattern_multirun(self, incr_dist_plot=False, plot_outlier=False, sum_file_out=False):  # without small ice test
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

        plot_sum_for_criteria(merge_water_sum_list, "water range sum view of all run with criteria",self.LowerBoundOfWater, self.UpperBoundOfNotWater, self.WaterOutlierLowerBound)

        if len(incr_outlier_info) != 0:
            if incr_dist_plot:
                plot_sum_for_criteria(incr_max_list, "incr max value of run {}".format(incr_outlier_info[0][0]),
                                      inc_max_outlier_boundary)

            if plot_outlier:
                one_graph_plot = 10
                if plot_within_range:
                    for idx_outlier, each_int in enumerate(incr_max_within_range_int):
                        now_label = "run{0}-delay{1}-{2}".format(incr_max_within_range_info[idx_outlier][0],
                                                                 incr_max_within_range_info[idx_outlier][1] + 1,
                                                                 idx_outlier)
                        print("{0}-th outlier : run{1}-delay{2}-key{3}".format(idx_outlier, incr_max_within_range_info[idx_outlier][0],incr_max_within_range_info[idx_outlier][1] + 1,incr_max_within_range_info[idx_outlier][2]))
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
                            print("{0}-th outlier : run{1}-delay{2}-key{3}".format(idx_outlier,incr_outlier_info[idx_outlier][0],incr_outlier_info[idx_outlier][1] + 1, incr_outlier_info[idx_outlier][2]))
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

    def small_ice_test(self, num_negative_delay, run_list_to_merge=[], expand_negative_pool=False, plot_outlier=False, sum_file_out=False):

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
        first_temp_int_list = []
        second_temp_int_list = []
        whole_run_int_files = []
        int_list_for_hist = []
        pulseData_each_delay_list = []
        pulseData_run_list = []
        save_pulse_list = []
        negative_delay_list = []

        water_range_int_sum_list = []
        incr_max_list = []
        incr_outlier_int = []
        incr_outlier_info = []

        for list_idx, idx_run in enumerate(run_list_to_merge):
            now_watersum_test_pass_save_file_root = "../results/each_run_watersum_int/watersum_test_pass_run" + str(run_list_to_merge[list_idx]) + ".npy"
            temp_int_list = np.load(now_watersum_test_pass_save_file_root, allow_pickle=True)
            if list_idx == 0:
                first_temp_int_list = np.array(temp_int_list)
            elif list_idx == (len(run_list_to_merge) - 1):
                second_temp_int_list = np.array(temp_int_list)
                for delay_idx in range(len(second_temp_int_list)):
                    temp_int_list = np.append(first_temp_int_list[delay_idx], second_temp_int_list[delay_idx])
                    whole_run_int_files.append(temp_int_list)
            else:
                second_temp_int_list = np.transpose(temp_int_list)
                for delay_idx in range(len(second_temp_int_list)):
                    temp_int_list = np.append(first_temp_int_list[delay_idx], second_temp_int_list[delay_idx])
                    first_temp_int_list.append(temp_int_list)

        for idx_delay, each_run_int_files in enumerate(whole_run_int_files):
            now_int_file_num = len(whole_run_int_files)
            now_water_sum_list = []
            now_fileout_list = np.array([])
            pulseData_each_delay_list = []
            for idx_file, each_int_file in enumerate(each_run_int_files):
                now_int_key = each_int_file.key
                # num_small_ice_peak_each_delay = 0
                now_delay_small_ice_fail_int = []
                now_int_val = np.array(each_int_file.intensity_val)
                now_int_water_sum = sum(now_int_val[self.water_q_range_start_idx:self.water_q_range_after_idx])
                now_int_next = np.roll(now_int_val, 1)
                now_int_next[0] = now_int_next[1]  # remove last element
                now_int_incr = np.abs(now_int_val - now_int_next)
                if small_ice_test:
                    num_small_ice_peaks, properties = scipy.find_peaks(now_int_incr[self.water_q_range_start_idx:self.water_q_range_after_idx],prominence=(prominence_min_val, prominence_max_val))
                    if len(num_small_ice_peaks) != 0:
                        incr_outlier_int.append(now_int_val)
                        incr_outlier_info.append([run_list_to_merge[:], idx_file, now_int_key])
                        self.strange_peak_key_blacklist.append(now_int_key)
                        now_pulseID = re.findall("(.*)\.(.*)_(.*)", now_int_key)[0][2]
                        now_pulseID = int(now_pulseID)
                        small_ice_test_fail_int.append(now_int_val)
                        small_ice_test_fail_info.append([run_list_to_merge[:], idx_file, now_pulseID])
                        now_delay_small_ice_fail_int.append(now_int_val)
                        num_small_ice_peak_each_delay += 1

                    else:
                        incr_test_pass_int.append(now_int_val)
                        incr_test_pass_info.append([run_list_to_merge[:], idx_delay, now_int_key])
                        pulseData_each_delay_list.append(each_int_file)
                        if idx_delay < num_negative_delay:
                            negative_delay_list.append([idx_delay, now_int_key])
                        # diff_test_fail_info.append([self.runList[idx_run], idx_file, now_pulseID, temp_q_idx_of_diff_max])
                    now_water_sum_list.append(now_int_water_sum)
                    now_file_out = [now_int_key, now_int_water_sum]
                    now_fileout_list = np.append(now_fileout_list, now_file_out)
            pulseData_run_list.append(pulseData_each_delay_list)
            if (idx_delay + 1) % print_criteria == 0:
                print("read {0} / {1} file".format(idx_delay + 1, now_int_file_num))
            small_ice_test_fail_all_delay.append(now_delay_small_ice_fail_int)
            num_small_ice_peaks_all_delay.append(num_small_ice_peak_each_delay)
            water_range_int_sum_list.append(now_water_sum_list)
                # TODO : add ice range plot
        print("end for run{0}&{1} files".format(run_list_to_merge[0], run_list_to_merge[1]))

        if sum_file_out:
            now_fileout_list = np.array(now_fileout_list)
            for idx in range(len(run_list_to_merge)):
                if idx != (len(run_list_to_merge) - 1):
                    tot_run_name = str(run_list_to_merge[idx]) + "+"
                else:
                    tot_run_name = str(run_list_to_merge[idx])
        now_save_file_root = "../results/merge_run_small_ice_pass/run{0}_{1}_small_ice_test.npy".format(run_list_to_merge[0], run_list_to_merge[1])#TODO: need to prepare for more than two runs
        #key & intensity of watersum
        np.save(now_save_file_root, now_fileout_list)
        print("successful file out :", now_save_file_root)
        merge_water_sum_list = np.array(water_range_int_sum_list).reshape(-1)
        print(merge_water_sum_list.shape)
        for idx in range(len(water_range_int_sum_list)):
            int_list_for_hist.extend(water_range_int_sum_list[idx])
        # plot_sum_for_criteria(merge_water_sum_list, "water range sum view of all run")
        hist_water_sum, bin_edges = np.histogram(int_list_for_hist, bins=200)
        self.hist_fileout(hist_water_sum, bin_edges, file_name="../results/water_sum_hist.dat",runList=self.runList)

        # temp value
        self.UpperBoundOfNotWater = self.eachRunInfo[0][1]
        self.LowerBoundOfWater = self.eachRunInfo[0][2]
        self.WaterOutlierLowerBound = self.eachRunInfo[0][3]

        plot_sum_for_criteria(merge_water_sum_list, "water range sum view of all run with criteria",self.LowerBoundOfWater,self.UpperBoundOfNotWater, self.WaterOutlierLowerBound)

        if len(incr_outlier_info) != 0:
            if plot_each_outlier:
                idx_each_outlier = 0
                for idx_outlier, each_int in enumerate(incr_outlier_int):
                    try:
                        if idx_outlier == each_outlier_plot_list[idx_each_outlier]:
                            now_label = "run{0}-delay{1}-{2}".format(incr_outlier_info[idx_outlier][0], incr_outlier_info[idx_outlier][1] + 1, idx_outlier)
                            print("{0}-th outlier : run{1}-delay{2}-key{3}".format(idx_outlier,incr_outlier_info[idx_outlier][0],incr_outlier_info[idx_outlier][1] + 1,incr_outlier_info[idx_outlier][2]))
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

            elif plot_small_ice_test_fail:
                one_graph_plot = 10
                tot_num_small_ice_outlier_early_delay = 0
                for idx_delay in range(small_ice_plot_end_delay - small_ice_plot_start_delay):
                    idx_delay = idx_delay + small_ice_plot_start_delay
                    if idx_delay != 0:
                        tot_num_small_ice_outlier_early_delay = num_small_ice_peaks_all_delay[idx_delay - 1]
                        for idx_outlier in range(
                                num_small_ice_peaks_all_delay[idx_delay] - num_small_ice_peaks_all_delay[idx_delay - 1]):
                            now_label = "run{0}-delay{1}-{2}".format(small_ice_test_fail_info[idx_delay][0],small_ice_test_fail_info[tot_num_small_ice_outlier_early_delay][1] + 1,tot_num_small_ice_outlier_early_delay + idx_outlier)
                            plt.plot(small_ice_test_fail_all_delay[idx_delay][idx_outlier], marker='.', markersize=1, label=now_label)
                            if idx_outlier % one_graph_plot == (one_graph_plot - 1):
                                plt.title("Small ice test failure of run" + str(small_ice_test_fail_info[0][0]))
                                plt.legend()
                                plt.show()
                        if (num_small_ice_peaks_all_delay[idx_delay] - num_small_ice_peaks_all_delay[idx_delay - 1]) % one_graph_plot != 0:
                            plt.title("Small ice test failure of run" + str(small_ice_test_fail_info[0][0]))
                            plt.legend()
                            plt.show()

                    else:
                        for idx_outlier in range(num_small_ice_peaks_all_delay[idx_delay]):
                            now_label = "run{0}-delay{1}-{2}".format(small_ice_test_fail_info[idx_delay][0],small_ice_test_fail_info[tot_num_small_ice_outlier_early_delay][1] + 1,tot_num_small_ice_outlier_early_delay + idx_outlier)
                            plt.plot(small_ice_test_fail_all_delay[idx_delay][idx_outlier], marker='.',markersize=1, label=now_label)
                            if idx_outlier % one_graph_plot == (one_graph_plot - 1):
                                plt.title("Small ice test failure of run" + str(small_ice_test_fail_info[0][0]))
                                plt.legend()
                                plt.show()
                        if (num_small_ice_peaks_all_delay[idx_delay] - num_small_ice_peaks_all_delay[idx_delay - 1]) % one_graph_plot != 0:
                            plt.title("Small ice test failure of run" + str(small_ice_test_fail_info[0][0]))
                            plt.legend()
                            plt.show()

            elif plot_incr_test_pass:
                one_graph_plot = 10
                for idx_pass, each_int in enumerate(incr_test_pass_int):
                    # now_label = "run{0}-delay{1}-{2}-q_idx {3}-key {4}".format(diff_test_fail_info[idx_outlier][0],diff_test_fail_info[idx_outlier][1] + 1, idx_outlier, where_max_diff_test_fail[idx_outlier],diff_test_fail_info[idx_outlier][2])
                    now_label = "run{0}-delay{1}-{2}".format(incr_test_pass_info[idx_pass][0], incr_test_pass_info[idx_pass][1] + 1, idx_pass)
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

            print("now remove {0} strange peak at run{1}, {2} shots are removed by small ice test".format(len(incr_outlier_int), incr_outlier_info[0][0], len(small_ice_test_fail_int)))
                # np.savetxt('Difference test failure of run 50.txt', diff_test_fail_info, fmt='%s', delimiter='/')
            small_ice_file_out_root = "/home/myeong0609/PAL-XFEL_20230427/analysis/results/merge_run_small_ice_pass/"
            np.save(small_ice_file_out_root + "small_ice_pass_of_run{0}_{1}.npy".format(run_list_to_merge[0], run_list_to_merge[1]), incr_test_pass_info) #TODO: need to change for many runs
            #np.save((small_ice_file_out_root + "small_ice_test_pass_negative_delay_pulse_info_run{0}_{1}.npy").format(run_list_to_merge[0], run_list_to_merge[1]), negative_delay_list)
            save_pulse_list.append(pulseData_run_list)
            self.each_run_int_val_list = save_pulse_list

    def q_interpolate(self,start_q, end_q, interp_q_val, original_q_val, interp_int):
        # start_q = 1
        # end_q = 6
        start_q_idx = np.argwhere(interp_q_val >= start_q)[0][0]
        end_q_idx = np.argwhere(interp_q_val < end_q)[-1][0]

        # dat = interp_int[self.water_q_range_start_idx: self.water_q_range_after_idx]
        interp_q_val = interp_q_val[start_q_idx - 1:end_q_idx + 1]
        dat = interp_int

        # plt.plot(original_q_val, dat)
        # plt.show()

        temp_f = interpolate.interp1d(original_q_val, dat, fill_value="extrapolate")
        interp_data = temp_f(interp_q_val)

        plt.plot(original_q_val, dat, label="org dat, {}".format(len(dat)))
        plt.plot(interp_q_val, interp_data, label="interp dat, {}".format(len(interp_data)))
        plt.legend()
        plt.show()
        same_len_with_OKE_sum = np.sum(interp_data)
        return same_len_with_OKE_sum

def extract_laser_on_off_list_only_water(data_list, len_laser_off_all_delay = 5, expand_negative_pool = False):
    laser_on_water_idx = []
    laser_off_water_idx = []

    if expand_negative_pool:
        for data_idx, each_data in enumerate(data_list):
            if each_data.laser_is_on:
                laser_on_water_idx.append(data_idx)
            else:
                laser_off_water_idx.append(data_idx)
    else:
        for data_idx, each_data in enumerate(data_list):
            if each_data.laser_is_on:
                laser_on_water_idx.append(data_idx)
            else:
                laser_off_water_idx.append(data_idx)

    if expand_negative_pool:
        if len_laser_off_all_delay == 0:
            pass
        else:
            print("laser on droplet : ", len(laser_on_water_idx), "laser off : ", len_laser_off_all_delay)
    else:
        print("laser on droplet : ", len(laser_on_water_idx), "laser off : ", len(laser_off_water_idx))
    return laser_on_water_idx, laser_off_water_idx



# TODO : cut front and back data is nessesary!!

