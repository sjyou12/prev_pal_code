from palxfel_scatter.diff_pair_1dcurve.DataClasses import plot_sum_for_criteria, match_water_near_intensity_pair
from palxfel_scatter.diff_pair_1dcurve.Tth2qConvert import Tth2qConvert
from palxfel_scatter.diff_pair_1dcurve.PulseDataLight import PulseDataLight
import h5py as h5
import numpy as np
import re
import matplotlib.pyplot as plt
import os

class MultiRunProc:
    intensity_file_names = []
    intenisty_files = []
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

    each_run_int_val_list = []
    whole_run_diff_list = []
    whole_run_cutted_diff_list = []

    tth_to_q_cvt = None
    each_run_file_dir = []
    whole_run_pair_list = []
    now_run_delay_num = 0
    strange_peak_key_blacklist = []

    def __init__(self, each_run_info):
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

        first_run_img_file_dir = self.FileCommonRoot + "run{0:04d}_00001_DIR/eh1rayMXAI_int/".format(self.runList[0])
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

        for each_run_num in self.runList:
            now_file_dir = self.FileCommonRoot + "run{0:04d}_00001_DIR/".format(each_run_num)
            self.each_run_file_dir.append(now_file_dir)
            temp_int_files = []
            for idx in range(now_delay_num):
                now_int_path = now_file_dir + self.intensity_file_names[idx] + ".h5"
                temp_int_file = h5.File(now_int_path, 'r')
                temp_int_files.append(temp_int_file)
            self.intenisty_files.append(temp_int_files)

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
        for idx_run, each_run_int_files in enumerate(self.intenisty_files):
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

        plot_sum_for_criteria(merge_water_sum_list, "water range sum view of all run with criteria", self.LowerBoundOfWater,
                              self.UpperBoundOfNotWater, self.WaterOutlierLowerBound)

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
        print("going to find strange peak pattern")
        # TODO : need to change
        inc_max_outlier_boundary = 1000

        print_criteria = 10

        water_range_int_sum_list = []
        incr_max_list = []
        incr_outlier_int = []
        incr_outlier_info = []
        for idx_run, each_run_int_files in enumerate(self.intenisty_files):
            now_int_file_num = len(each_run_int_files)
            now_water_sum_list = []
            now_fileout_list = []
            for idx_file, each_int_file in enumerate(each_run_int_files):
                now_int_keys = list(each_int_file.keys())
                for each_key in now_int_keys:
                    now_int_val = np.array(each_int_file[each_key])
                    now_int_water_sum = sum(now_int_val[self.water_q_range_start_idx:self.water_q_range_after_idx])
                    now_int_next = np.roll(now_int_val, 1)
                    now_int_next[0] = 0 # remove last element
                    now_int_incr = np.abs(now_int_val - now_int_next)
                    now_incr_max = np.max(now_int_incr)
                    incr_max_list.append(now_incr_max)
                    if now_incr_max > inc_max_outlier_boundary:
                        incr_outlier_int.append(now_int_val)
                        incr_outlier_info.append([self.runList[idx_run], idx_file, each_key])
                        self.strange_peak_key_blacklist.append(each_key)

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
                for idx_outlier, each_int in enumerate(incr_outlier_int):
                    now_label = "run{0}-delay{1}-{2}".format(incr_outlier_info[idx_outlier][0], incr_outlier_info[idx_outlier][1],idx_outlier)
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

    @staticmethod
    def hist_fileout(hist, bin_edges, file_name, runList):
        outFp = open(file_name, 'w')
        outFp.write("run List : " + str(runList) + "\n")
        outFp.write("bin_edge_left\tfrequency\n")
        for idx, hist_val in enumerate(hist):
            outFp.write("{}\t{}\n".format(bin_edges[idx], hist_val))
        outFp.close()
        print("histogram file out : ", file_name)

    def read_intensity_only_water(self, np_file_out=False, rm_vapor=False):
        """
        execute one of (this function and plot_water_sum_dist)
        two function have overlapped feature
        """
        print("read intensity files and save water range data")

        print_criteria = 10

        self.UpperBoundOfNotWater = self.eachRunInfo[0][1]
        self.LowerBoundOfWater = self.eachRunInfo[0][2]
        self.WaterOutlierLowerBound = self.eachRunInfo[0][3]

        if len(self.strange_peak_key_blacklist) == 0:
            print("strange peak blacklist is empty!")

        for idx_run, each_run_int_files in enumerate(self.intenisty_files):
            now_int_file_num = len(each_run_int_files)
            now_run_int_val_list = []
            for idx_file, each_int_file in enumerate(each_run_int_files):
                now_int_keys = list(each_int_file.keys())
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
                now_run_int_val_list.append(now_delay_int_val_list)
                if (idx_file + 1) % print_criteria == 0:
                    print("read {0} / {1} file".format(idx_file + 1, now_int_file_num))
            self.each_run_int_val_list.append(now_run_int_val_list)
            print("end intensity read for run{} files".format(self.runList[idx_run]))

        if np_file_out:
            # save each_run_int_val_list numpy array
            temp_save_name = "../results/whole_run_int/whole_run_int_" + self.file_common_name + ".npy"
            print("save as :", temp_save_name)
            np.save(temp_save_name, self.each_run_int_val_list)

            # save q_val numpy array
            temp_save_name = "../results/q_val_" + self.file_common_name + ".npy"
            print("save as :", temp_save_name)
            np.save(temp_save_name, self.q_val)

    def pairwise_diff_calc(self, test_plot=False, test_plot_num=5, fileout_pair_info=True):
        plot_start_idx = 10
        plot_end_idx = -120
        print("plot_start / end", plot_start_idx, plot_end_idx)

        for each_run_int_list in self.each_run_int_val_list:
            now_run_diff_list = []
            now_pair_arr = []
            for each_delay_int_list in each_run_int_list:
                laser_on_idx, laser_off_idx = extract_laser_on_off_list_only_water(each_delay_int_list)
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



    def fileout_pair_info_only(self, pair_file_out=False, diff_calc_test=False, test_plot=False, test_plot_num=5, ):
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

    def additional_process_diff(self, show_before_cutted=True, show_after_cutted=False, file_out=False,
                                svd_with_cut=False, fileout_pair_info=False):
        # global PlotStartIdx
        # global PlotEndIdx
        # global svd_start_idx
        # global svd_end_idx
        # plot_start_idx = PlotStartIdx
        # plot_end_idx = PlotEndIdx

        intg_start_idx = 140
        intg_end_idx = 348
        cutoff_criteria = [1E5 for _ in range(55)]
        cutted_diff_list = []
        cutted_avg = []

        test_plot_num = 10

        whole_run_sum_list = []
        whole_run_cutted_avg = []
        now_run_pair_list = []
        now_delay_pair_list = []
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
                now_run_sum_list.append(now_delay_sum_list)

                if show_before_cutted:
                    if delay_idx < test_plot_num:
                        now_graph_title = "integration hist of run" + str(self.runList[run_idx]) + "//" + str(
                            delay_idx) + "-th delay"
                        plot_sum_for_criteria(now_delay_sum_list, now_graph_title, v_line_1=cutoff_criteria[delay_idx])
                        print(now_graph_title)

                now_delay_cutted_diff = each_delay_diff_list[now_delay_sum_list < cutoff_criteria[delay_idx]]
                now_run_cutted_diff.append(now_delay_cutted_diff)

                print("remove ", len(each_delay_diff_list[now_delay_sum_list >= cutoff_criteria[delay_idx]]),
                      " in", delay_idx, "-th delay")

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
                    print("left pair : ", len(now_delay_left_pair_list), " in", delay_idx, "-th delay")

                now_delay_avg = np.average(now_delay_cutted_diff, axis=0)
                if now_delay_avg.shape is np.nan:
                    now_delay_avg = np.zeros_like(self.q_val)
                now_run_cutted_avg_list.append(now_delay_avg)

            whole_run_sum_list.append(now_run_sum_list)
            whole_run_cutted_avg.append(now_run_cutted_avg_list)
            self.whole_run_cutted_diff_list.append(now_run_cutted_diff)

            if fileout_pair_info:
                now_pair_arr = np.array(now_run_cutted_pair, dtype=object)
                now_save_file_root = "../results/anisotropy/run" + str(self.runList[run_idx]) + "_pairinfo"
                np.save(now_save_file_root, now_pair_arr)
                print("successful file out :", now_save_file_root)

        # save whole_run_cutted_diff_list numpy array
        temp_save_name = "../results/whole_run_diff/whole_run_cutted_diff_" + self.file_common_name + ".npy"
        print("save as :", temp_save_name)
        np.save(temp_save_name, self.whole_run_cutted_diff_list)
        self.whole_run_diff_list = []

        # save whole_run_cutted_avg numpy array
        temp_save_name = "../results/whole_run_avg/whole_run_cutted_avg_" + self.file_common_name + ".npy"
        print("save as :", temp_save_name)
        np.save(temp_save_name, whole_run_cutted_avg)

    def read_I0_value(self):
        all_run_I0_dict_list = []
        for idx_run, _ in enumerate(self.runList):
            now_delay_num = len(self.intenisty_files[idx_run])
            now_run_I0_dict_list = []
            for idx_delay in range(now_delay_num):
                now_delay_I0_dict = {}
                now_file_name_I0 = 'eh1qbpm1_totalsum/001_001_%03d' % (idx_delay + 1)
                now_I0_path = self.each_run_file_dir[idx_run] + now_file_name_I0 + ".h5"
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
        inc_max_outlier_boundary = 250
        vapor_range_start = 5000
        vapor_range_end = 12000

        print_criteria = 10
        self.read_I0_value()

        water_range_int_sum_list = []
        incr_max_list = []
        incr_outlier_int = []
        incr_outlier_info = []
        vapor_norm_int = []
        vapor_info = []
        for idx_run, each_run_int_files in enumerate(self.intenisty_files):
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

        merge_water_sum_list = np.array(water_range_int_sum_list).reshape(-1)
        print(merge_water_sum_list.shape)
        # plot_sum_for_criteria(merge_water_sum_list, "water range sum view of all run")
        hist_water_sum, bin_edges = np.histogram(water_range_int_sum_list, bins=200)
        self.hist_fileout(hist_water_sum, bin_edges, file_name="../results/water_sum_hist.dat", runList=self.runList)

        # temp value
        self.UpperBoundOfNotWater = self.eachRunInfo[0][1]
        self.LowerBoundOfWater = self.eachRunInfo[0][2]
        self.WaterOutlierLowerBound = self.eachRunInfo[0][3]

        plot_sum_for_criteria(merge_water_sum_list, "water range sum view of all run with criteria",
                              self.LowerBoundOfWater,
                              self.UpperBoundOfNotWater, self.WaterOutlierLowerBound)

        if incr_dist_plot:
            plot_sum_for_criteria(incr_max_list, "incr max value of run {}".format(incr_outlier_info[0][0]),
                                  inc_max_outlier_boundary)

        if plot_outlier:
            one_graph_plot = 10
            for idx_outlier, each_int in enumerate(incr_outlier_int):
                now_label = "run{0}-delay{1}-{2}".format(incr_outlier_info[idx_outlier][0],
                                                         incr_outlier_info[idx_outlier][1], idx_outlier)
                print("{0}-th outlier : run{1}-delay{2}-key{3}".format(idx_outlier, incr_outlier_info[idx_outlier][0],
                                                                       incr_outlier_info[idx_outlier][1],
                                                                       incr_outlier_info[idx_outlier][2]))
                plt.plot(each_int, label=now_label)
                if idx_outlier % one_graph_plot == (one_graph_plot - 1):
                    plt.title("outlier of run" + str(incr_outlier_info[0][0]))
                    plt.legend()
                    plt.show()
            plt.title("outlier of run" + str(incr_outlier_info[0][0]))
            plt.legend()
            plt.show()

        avg_vapor_range = np.average(vapor_norm_int, axis=0)
        self.norm_vapor_int = avg_vapor_range
        if plot_vapor_avg:
            plt.title("I0 normalized vapor average!")
            plt.plot(self.q_val, avg_vapor_range)
            plt.show()

        print("now remove {} strange peak at run{}".format(len(incr_outlier_int), incr_outlier_info[0][0]))

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

# TODO : cut front and back data is nessesary!!

