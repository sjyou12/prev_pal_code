import numpy as np
import os
from matplotlib import pyplot as plt

def plot_rsv_linear_n_log_x_scale(x_data, y_data, fluence_list, x_label, y_label, scale_cut_x_val, suptitle):
    x_data = np.array(x_data)
    rsv_arr = (np.array(y_data))
    # rsv_arr = y_data
    fig, axs = plt.subplots(ncols=2, figsize=(8, 4), gridspec_kw={'wspace':0})
    linear_ax = axs[0]
    log_ax = axs[1]
    for run_idx in range(len(rsv_arr)):
        y_data = np.array(rsv_arr[run_idx][:len(x_data)])
        linear_range_mask = (x_data <= scale_cut_x_val)
        log_range_mask = (x_data >= scale_cut_x_val)
        linear_x_data = x_data[linear_range_mask]
        linear_y_data = y_data[linear_range_mask]
        log_x_data = x_data[log_range_mask]
        log_y_data = y_data[log_range_mask]

        linear_ax.plot(linear_x_data, linear_y_data)
        log_ax.plot(log_x_data, log_y_data, label="{} TW/cm2".format(fluence_list[run_idx]))

        log_ax.sharey(linear_ax)
        log_ax.get_yaxis().set_visible(False)
        log_ax.set_xscale("log")
        log_ax.set_xlim((scale_cut_x_val, None))
        linear_ax.set_xlim((None, scale_cut_x_val))
        log_ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        linear_ax.set_ylabel(y_label)

    for each_ax in axs:
        each_ax.xaxis.grid(ls=':')

    linear_ax.set_xlabel(x_label)
    fig.suptitle(suptitle)
    fig.set_tight_layout(True)
    fig.show()


class data_info:
    azi_common_path = None
    svd_common_path = None
    run_list = []
    fluence_list = []
    aniso_singVal = []
    iso_singVal = []

    def __init__(self, azi_common_path, svd_common_path, run_list, fluence_list, aniso_target_sing_Num, iso_target_sing_Num, time_delay, plot_non_svd):
        self.azi_common_path = azi_common_path
        self.svd_common_path = svd_common_path
        self.run_list = run_list
        self.fluence_list = fluence_list
        self.aniso_target_sing_Num = aniso_target_sing_Num
        self.iso_target_sing_Num = iso_target_sing_Num

        self.azi_int = []
        self.all_run_iso = []
        self.all_run_aniso = []
        self.aniso_LSV_SVD_result = []
        self.aniso_RSV_SVD_result = []
        self.iso_LSV_SVD_result = []
        self.iso_RSV_SVD_result = []

        self.time_delay = time_delay
        self.full_q_val = []
        self.cut_q_val = []
        self.plot_non_svd = plot_non_svd

    def load_azi(self):
        for run_num in self.run_list:
            azi_dat_path = self.azi_common_path + "whole_run_cutted_avg_run{}.npy".format(run_num)
            now_run_azi = np.load(azi_dat_path, allow_pickle=True)[0]
            self.azi_int.append(now_run_azi)

    def load_each_delay_results(self):
        for run_num in self.run_list:
            now_run_iso = []
            now_run_aniso = []
            svd_dat_path = self.svd_common_path + "anal_result/run_{0:05d}/".format(run_num)
            for delay_idx, delay in enumerate(self.time_delay):
                if delay_idx == 0:
                    self.full_q_val = np.load(svd_dat_path + "run{0}_delay{1}_qval.npy".format(run_num, delay_idx+1))
                now_run_iso.append(np.load(svd_dat_path + "run{0}_delay{1}_iso.npy".format(run_num, delay_idx+1)))
                now_run_aniso.append(np.load(svd_dat_path + "run{0}_delay{1}_aniso.npy".format(run_num, delay_idx+1)))
            self.all_run_iso.append(now_run_iso)
            self.all_run_aniso.append(now_run_aniso)

    def load_SVD_result(self):
        now_run_iso = []
        now_run_aniso = []
        for run_idx, run_num in enumerate(self.run_list):
            svd_dat_path = self.svd_common_path + "svd/run{0:04d}/".format(run_num)
            cutted_aniso_LSV = np.loadtxt(svd_dat_path + "run{0}-aniso-cut_LSV.dat".format(run_num), skiprows=1)
            self.cut_q_val = cutted_aniso_LSV[:, 0]
            self.aniso_LSV_SVD_result.append(np.transpose(cutted_aniso_LSV[:, 1:]))
            cutted_aniso_RSV = np.loadtxt(svd_dat_path + "run{0}-aniso-cut_RSV.dat".format(run_num), skiprows=1)
            self.aniso_RSV_SVD_result.append(np.transpose(cutted_aniso_RSV[:, 1:]))
            cutted_aniso_singVal = np.loadtxt(svd_dat_path + "run{0}-aniso-cut_SingVal.dat".format(run_num), skiprows=1)
            self.aniso_singVal.append(cutted_aniso_singVal)

            cutted_iso_LSV = np.loadtxt(svd_dat_path + "run{0}-iso-cut_LSV.dat".format(run_num), skiprows=1)
            self.iso_LSV_SVD_result.append(np.transpose(cutted_iso_LSV[:, 1:]))
            cutted_iso_RSV = np.loadtxt(svd_dat_path + "run{0}-iso-cut_RSV.dat".format(run_num), skiprows=1)
            self.iso_RSV_SVD_result.append(np.transpose(cutted_iso_RSV[:, 1:]))
            cutted_iso_singVal = np.loadtxt(svd_dat_path + "run{0}-iso-cut_SingVal.dat".format(run_num), skiprows=1)
            self.iso_singVal.append(cutted_iso_singVal)
        self.iso_singVal = np.array(self.iso_singVal)
        self.aniso_singVal = np.array(self.aniso_singVal)

    def plot_results(self):
        chunk_num = 12
        chunked_Delay = []
        for chunk_idx in range(int(len(self.time_delay)/chunk_num) + 1):
            try:
                chunked_Delay.append(self.time_delay[chunk_idx * chunk_num : (chunk_idx+1)*chunk_num])
            except:
                chunked_Delay.append(self.time_delay[chunk_idx*chunk_num:])

        num_row = 6
        num_col = 2
        subplot_width = 6
        subplot_hieght = 3

        if self.plot_non_svd:
            for chunk_idx, chunked_delay in enumerate(chunked_Delay):
                fig, axs = plt.subplots(ncols=num_col, nrows=num_row,
                                        figsize=(subplot_width * num_col, subplot_hieght * num_row))
                fig.suptitle("Difference azi intg")
                for delay_idx, delay in enumerate(chunked_delay):
                    now_axs = axs.flatten()[delay_idx]
                    real_delay_idx = delay_idx + chunk_idx*chunk_num
                    now_axs.set_title("{} ps".format(delay))
                    for run_idx, each_run_int in enumerate(self.azi_int):
                        q_val = np.load(self.azi_common_path + "q_val_run{}.npy".format(self.run_list[run_idx]))
                        now_axs.plot(q_val, each_run_int[real_delay_idx], label='{}TW/cm2'.format(str(self.fluence_list[run_idx])))
                        now_axs.grid(ls=':')
                        now_axs.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                fig.set_tight_layout(True)
                plt.close(fig)
                # fig.show()

            for chunk_idx, chunked_delay in enumerate(chunked_Delay):
                fig, axs = plt.subplots(ncols=num_col, nrows=num_row,
                                        figsize=(subplot_width * num_col, subplot_hieght * num_row))
                fig.suptitle("Isotropy")
                for delay_idx, delay in enumerate(chunked_delay):
                    now_axs = axs.flatten()[delay_idx]
                    real_delay_idx = delay_idx + chunk_idx*chunk_num
                    now_axs.set_title("{} ps".format(delay))
                    for run_idx, each_run_int in enumerate(self.all_run_iso):
                        # q_val = np.load(self.azi_common_path + "q_val_run{}.npy".format(self.run_list[run_idx]))
                        now_axs.plot(self.full_q_val, each_run_int[real_delay_idx], label='{}TW/cm2'.format(str(self.fluence_list[run_idx])))
                        now_axs.set_ylim(-30, 30)
                        now_axs.grid(ls=':')
                        now_axs.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                fig.set_tight_layout(True)
                plt.close(fig)
                fig.show()

            for chunk_idx, chunked_delay in enumerate(chunked_Delay):
                fig, axs = plt.subplots(ncols=num_col, nrows=num_row, figsize=(subplot_width * num_col, subplot_hieght * num_row))
                fig.suptitle("Anisotropy")
                for delay_idx, delay in enumerate(chunked_delay):
                    now_axs = axs.flatten()[delay_idx]
                    real_delay_idx = delay_idx + chunk_idx*chunk_num
                    now_axs.set_title("{} ps".format(delay))
                    for run_idx, each_run_int in enumerate(self.all_run_aniso):
                        # q_val = np.load(self.azi_common_path + "q_val_run{}.npy".format(self.run_list[run_idx]))
                        now_axs.plot(self.full_q_val, each_run_int[real_delay_idx], label='{}TW/cm2'.format(str(self.fluence_list[run_idx])))
                        now_axs.set_ylim(-3, 3)
                        now_axs.grid(ls=':')
                        now_axs.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                fig.set_tight_layout(True)
                # plt.close(fig)
                fig.show()

        for sing_val_idx in range(len(np.transpose(self.iso_singVal))):
            for fluence_idx, fluence_val in enumerate(self.fluence_list):
                if (self.iso_RSV_SVD_result[fluence_idx][sing_val_idx])[-1] < 0:
                    self.iso_RSV_SVD_result[fluence_idx][sing_val_idx] = -self.iso_RSV_SVD_result[fluence_idx][sing_val_idx]
                    self.iso_LSV_SVD_result[fluence_idx][sing_val_idx] = -self.iso_LSV_SVD_result[fluence_idx][sing_val_idx]
        #     plt.plot(self.time_delay, self.iso_RSV_SVD_result[run_idx], label='{}TW/cm2'.format(self.fluence_list[run_idx]))
        # plt.title("Isotropy RSV")
        # plt.legend()
        # plt.show()
            plot_rsv_linear_n_log_x_scale(self.time_delay, np.array(self.iso_RSV_SVD_result)[:, sing_val_idx], self.fluence_list, "Delay (ps)", "RSV (arb. unit)", 3, "Isotropy RSV of {}th component".format(sing_val_idx+1))

            for fluence_idx, fluence_val in enumerate(self.fluence_list):
                plt.plot(self.cut_q_val, np.array(self.iso_LSV_SVD_result)[fluence_idx][sing_val_idx], label='{}TW'.format(fluence_val))
            plt.title("Isotropy LSV of {}th component".format(sing_val_idx+1))
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.tight_layout()
            plt.show()

        for sing_val_idx in range(len(np.transpose(self.aniso_singVal))):
            for fluence_idx, fluence_val in enumerate(self.fluence_list):
                if np.max(self.aniso_RSV_SVD_result[fluence_idx][0]) > 0.2:
                    continue
                else:
                    self.aniso_RSV_SVD_result[fluence_idx][sing_val_idx] = -self.aniso_RSV_SVD_result[fluence_idx][sing_val_idx]
                    self.aniso_LSV_SVD_result[fluence_idx][sing_val_idx] = -self.aniso_LSV_SVD_result[fluence_idx][sing_val_idx]
            # plt.plot(self.time_delay, self.aniso_RSV_SVD_result[run_idx], label='{}TW/cm2'.format(self.fluence_list[run_idx]))
        # plt.title("Anisotropy RSV")
        # plt.legend()
        # plt.show()
            plot_rsv_linear_n_log_x_scale(self.time_delay, np.array(self.aniso_RSV_SVD_result)[:, sing_val_idx], self.fluence_list, "Delay (ps)", "RSV (arb. unit)", 3, "Anisotropy RSV of {}th component".format(sing_val_idx+1))

            for fluence_idx, fluence_val in enumerate(self.fluence_list):
                plt.plot(self.cut_q_val, self.aniso_LSV_SVD_result[fluence_idx][sing_val_idx], label='{}TW'.format(fluence_val))
            plt.title("Anisotropy LSV of {}th component".format(sing_val_idx+1))
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.tight_layout()
            plt.show()

        fig, axis_1 = plt.subplots()
        axis_2 = axis_1.twinx()
        axis_1.scatter(self.fluence_list, self.iso_singVal[:, 0], label="1st SingVal", color="red")
        axis_2.scatter(self.fluence_list, self.iso_singVal[:, 1], label="2nd SingVal", color="blue")
        # plt.scatter(self.fluence_list, self.iso_singVal)
        fig.suptitle("Isotropy SingVal")
        axis_1.set_xlabel("Fluence (TW/cm2)")
        axis_1.set_ylabel("SingVal")
        fig.legend()
        fig.show()

        fig, axis_1 = plt.subplots()
        axis_2 = axis_1.twinx()
        axis_1.scatter(self.fluence_list, self.aniso_singVal[:, 0], label="1st SingVal", color="red")
        axis_2.scatter(self.fluence_list, self.aniso_singVal[:, 1], label="2nd SingVal", color="blue")
        # plt.scatter(self.fluence_list, self.iso_singVal)
        fig.suptitle("Ansotropy SingVal")
        axis_1.set_xlabel("Fluence (TW/cm2)")
        axis_1.set_ylabel("SingVal")
        fig.legend()
        fig.show()

azi_common_path = "../../results/whole_run_avg/"
svd_common_path = "../../results/anisotropy/"

# TODO: change values below
# run_list = [20, 25, 28, 29, 57, 59, 45, 47, 49]
# aniso_target_sing_Num = [1, 1, 1, 1, 1, 1, 1, 1, 1]
# iso_target_sing_Num = [1, 1, 1, 1, 1, 1, 1, 1, 1]
# fluence_list = [168, 130, 100, 70, 50, 40, 30, 20, 15]
# run_list = [153, 155, 157, 159, 161, 165]
# aniso_target_sing_Num = [1, 1, 1, 1, 1, 1]
# iso_target_sing_Num = [1, 1, 1, 1, 1, 1]
# fluence_list = [168, 130, 100, 70, 50, 30]
# run_list = [165, 211, 219] #30 TW
run_list = [204, 208, 209, 213, 215, 217, 221] #7.5 TW
aniso_target_sing_Num = [1, 1]
iso_target_sing_Num = [1, 1]
# fluence_list = ['30_1', '30_2', '30_3']
fluence_list = ['7.5_1', '7.5_2', '7.5_3', '7.5_4', '7.5_5', '7.5_6', '7.5_7']
# time_delay = [-3,-0.5,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5,0.7,1,1.7,3.2,5.6,10,32,100,320,1000,3000]
time_delay = [-3,-1,-0.8,-0.6,-0.4,-0.35,-0.3,-0.28,-0.26,-0.24,-0.22,-0.2,-0.18,-0.16,-0.14,-0.12,-0.1,-0.08,-0.06,-0.04,-0.02,0,0.02,
              0.04,0.06,0.08,0.1,0.12,0.14,0.16,0.18,0.2,0.22,0.24,0.26,0.28,0.3,0.32,0.34,0.36,0.38,0.4,0.42,0.46,0.5,0.55,0.6
            ,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2,2.2,2.4,2.6,2.8,3,3.5,4,5,6,8,10,17,32,56,100,170
            ,320,560,1000,1700,2500,3000]
# time_delay = range(0, 51)

plot_non_svd = False

compare_target = data_info(azi_common_path, svd_common_path, run_list, fluence_list, aniso_target_sing_Num, iso_target_sing_Num, time_delay, plot_non_svd)
compare_target.load_azi()
compare_target.load_each_delay_results()
compare_target.load_SVD_result()
compare_target.plot_results()