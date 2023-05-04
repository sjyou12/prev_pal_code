import numpy as np
import os
from matplotlib import pyplot as plt

def plot_rsv_linear_n_log_x_scale(x_data, y_data, fluence_list, x_label, y_label, scale_cut_x_val, suptitle):
    x_data = np.array(x_data)
    rsv_arr = np.array(y_data)
    fig, axs = plt.subplots(ncols=2, figsize=(8, 4), gridspec_kw={'wspace':0})
    linear_ax = axs[0]
    log_ax = axs[1]
    for run_idx in range(len(rsv_arr)):
        y_data = rsv_arr[run_idx][:len(x_data)]
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
    svd_common_path = None
    run_list = []
    fluence_list = []
    aniso_1st_singVal = []
    aniso_2nd_singVal = []
    iso_1st_singVal = []
    iso_2nd_singVal = []

    def __init__(self, svd_common_path, run_list, fluence_list, aniso_target_sing_Num, iso_target_sing_Num, time_delay):
        self.svd_common_path = svd_common_path
        self.run_list = run_list
        self.fluence_list = fluence_list
        self.aniso_target_sing_Num = aniso_target_sing_Num
        self.iso_target_sing_Num = iso_target_sing_Num

        self.all_run_iso = []
        self.all_run_aniso = []
        self.aniso_LSV_SVD_result = []
        self.aniso_RSV_SVD_result = []
        self.iso_LSV_SVD_result = []
        self.iso_RSV_SVD_result = []

        self.time_delay = time_delay
        self.full_q_val = []
        self.cut_q_val = []

    def load_SVD_result(self):
        now_run_iso = []
        now_run_aniso = []
        # self.aniso_target_sing_Num = np.array(self.aniso_target_sing_Num) - 1
        # self.iso_target_sing_Num = np.array(self.iso_target_sing_Num) - 1
        for run_idx, run_num in enumerate(self.run_list):
            svd_dat_path = self.svd_common_path + "svd/run{0:04d}/".format(run_num)
            cutted_aniso_LSV = np.loadtxt(svd_dat_path + "run{0}-aniso-cut_LSV.dat".format(run_num), skiprows=1)
            self.cut_q_val = cutted_aniso_LSV[:, 0]
            self.aniso_LSV_SVD_result.append(cutted_aniso_LSV[:, self.aniso_target_sing_Num[run_idx]])
            cutted_aniso_RSV = np.loadtxt(svd_dat_path + "run{0}-aniso-cut_RSV.dat".format(run_num), skiprows=1)
            self.aniso_RSV_SVD_result.append(cutted_aniso_RSV[:, self.aniso_target_sing_Num[run_idx]])

            cutted_aniso_singVal = np.loadtxt(svd_dat_path + "run{0}-aniso-cut_SingVal.dat".format(run_num), skiprows=1)
            self.aniso_1st_singVal.append(cutted_aniso_singVal[0])
            self.aniso_2nd_singVal.append(cutted_aniso_singVal[1])

            cutted_iso_LSV = np.loadtxt(svd_dat_path + "run{0}-iso-cut_LSV.dat".format(run_num), skiprows=1)
            self.iso_LSV_SVD_result.append(cutted_iso_LSV[:, self.iso_target_sing_Num[run_idx]])
            cutted_iso_RSV = np.loadtxt(svd_dat_path + "run{0}-iso-cut_RSV.dat".format(run_num), skiprows=1)
            self.iso_RSV_SVD_result.append(cutted_iso_RSV[:, self.iso_target_sing_Num[run_idx]])
            cutted_iso_singVal = np.loadtxt(svd_dat_path + "run{0}-iso-cut_SingVal.dat".format(run_num), skiprows=1)
            self.iso_1st_singVal.append(cutted_iso_singVal[0])
            self.iso_2nd_singVal.append(cutted_iso_singVal[1])

    def plot_results(self):
        for run_idx, run_num in enumerate(self.run_list):
            if np.min(self.iso_RSV_SVD_result[run_idx]) < -0.1:
                self.iso_RSV_SVD_result[run_idx] = -self.iso_RSV_SVD_result[run_idx]
                self.iso_LSV_SVD_result[run_idx] = -self.iso_LSV_SVD_result[run_idx]
            plt.plot(self.time_delay, self.iso_RSV_SVD_result[run_idx], label='{}TW/cm2'.format(self.fluence_list[run_idx]))
        plt.title("Isotropy RSV")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xlim(-1, 1)
        plt.tight_layout()
        plt.show()
        plot_rsv_linear_n_log_x_scale(self.time_delay, self.iso_RSV_SVD_result, self.fluence_list, "Delay (ps)", "RSV (arb. unit)", 3, "Isotropy RSV")

        for run_idx, run_num in enumerate(self.run_list):
            # if run_idx in [0, 1, 2]:
            #     self.iso_LSV_SVD_result[run_idx] = -self.iso_LSV_SVD_result[run_idx]
            plt.plot(self.cut_q_val, self.iso_LSV_SVD_result[run_idx], label='{}TW/cm2'.format(self.fluence_list[run_idx]))
        plt.title("Isotropy LSV")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.show()

        for run_idx, run_num in enumerate(self.run_list):
            if np.min(self.aniso_RSV_SVD_result[run_idx]) < -0.3:
                self.aniso_RSV_SVD_result[run_idx] = -self.aniso_RSV_SVD_result[run_idx]
                self.aniso_LSV_SVD_result[run_idx] = -self.aniso_LSV_SVD_result[run_idx]
            # plt.plot(self.time_delay, self.aniso_RSV_SVD_result[run_idx], label='{}TW/cm2'.format(self.fluence_list[run_idx]))
        # plt.title("Anisotropy RSV")
        # plt.legend()
        # plt.show()
        plot_rsv_linear_n_log_x_scale(self.time_delay, self.aniso_RSV_SVD_result, self.fluence_list, "Delay (ps)", "RSV (arb. unit)", 3, "Anisotropy RSV")

        for run_idx, run_num in enumerate(self.run_list):
            plt.plot(self.cut_q_val, self.aniso_LSV_SVD_result[run_idx], label='{}TW/cm2'.format(self.fluence_list[run_idx]))
        plt.title("Anisotropy LSV")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.show()

        fig, axis_1 = plt.subplots()
        axis_2 = axis_1.twinx()
        axis_1.scatter(self.fluence_list, self.iso_1st_singVal, label="1st SingVal", color="red")
        axis_2.scatter(self.fluence_list, self.iso_2nd_singVal, label="2nd SingVal", color="blue")
        # plt.scatter(self.fluence_list, self.iso_singVal)
        fig.suptitle("Isotropy SingVal")
        axis_1.set_xlabel("Fluence (TW/cm2)")
        axis_1.set_ylabel("SingVal")
        fig.legend()
        fig.show()

        fig, axis_1 = plt.subplots()
        axis_2 = axis_1.twinx()
        axis_1.scatter(self.fluence_list, self.aniso_1st_singVal, label="1st SingVal", color="red")
        axis_2.scatter(self.fluence_list, self.aniso_2nd_singVal, label="2nd SingVal", color="blue")
        # plt.scatter(self.fluence_list, self.iso_singVal)
        fig.suptitle("Ansotropy SingVal")
        axis_1.set_xlabel("Fluence (TW/cm2)")
        axis_1.set_ylabel("SingVal")
        fig.legend()
        fig.show()

svd_common_path = "../../results/anisotropy/"

run_list = [66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 90, 88]
aniso_target_sing_Num = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2]
iso_target_sing_Num = [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1]
fluence_list = [168, 130, 100, 70, 50, 40, 30, 20, 15, 10, 5, 3]
# time_delay = [-3,-0.5,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5,0.7,1,1.7,3.2,5.6,10,32,100,320,1000,3000]
time_delay = [-3,-1,-0.8,-0.6,-0.4,-0.35,-0.3,-0.28,-0.26,-0.24,-0.22,-0.2,-0.18,-0.16,-0.14,-0.12,-0.1,-0.08,-0.06,-0.04,-0.02,0,0.02,
              0.04,0.06,0.08,0.1,0.12,0.14,0.16,0.18,0.2,0.22,0.24,0.26,0.28,0.3,0.32,0.34,0.36,0.38,0.4,0.42,0.46,0.5,0.55,0.6
            ,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2,2.2,2.4,2.6,2.8,3,3.5,4,5,6,8,10,17,32,56,100,170
            ,320,560,1000,1700,2500,3000]


compare_target = data_info(svd_common_path, run_list, fluence_list, aniso_target_sing_Num, iso_target_sing_Num, time_delay)
compare_target.load_SVD_result()
compare_target.plot_results()