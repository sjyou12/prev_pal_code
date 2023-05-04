from codes.SVD.SVDCalc import SVDCalc
import numpy as np
import matplotlib.pyplot as plt


class AnisoSVD:
    svd_delay_range = range(54)
    one_fig_num_of_plot = 5

    def __init__(self):
        self.first_q_val = []
        self.all_delay_iso = []
        self.all_delay_aniso = []

    def read_aniso_results(self):
        for idx_delay in self.svd_delay_range:
            file_load_root = "../results/anisotropy/anal_result/delay" + str(idx_delay + 1)
            q_val_file_name = file_load_root + "_qval.npy"
            iso_file_name = file_load_root + "_iso.npy"
            aniso_file_name = file_load_root + "_aniso.npy"
            now_q_val_arr = np.load(q_val_file_name)
            now_iso_arr = np.load(iso_file_name)
            now_aniso_arr = np.load(aniso_file_name)
            if idx_delay == 0:
                self.first_q_val = now_q_val_arr
            else:
                q_val_diff = now_q_val_arr - self.first_q_val
                diff_sum = np.sum(q_val_diff)
                if diff_sum > 0:
                    print("q_val difference :", diff_sum, "[ in " + str(idx_delay + 1) + "-th delay")
            self.all_delay_iso.append(now_iso_arr)
            self.all_delay_aniso.append(now_aniso_arr)

    def change_on_fig_num_of_plot(self, num_plot):
        self.one_fig_num_of_plot = num_plot

    def plot_iso_signal(self):
        for idx_delay in self.svd_delay_range:
            plt.plot(self.first_q_val, self.all_delay_iso[idx_delay], label=str(idx_delay + 1) + "-th delay")
            if (idx_delay % self.one_fig_num_of_plot) == 0:
                plt.title("isotropic signal of multiple delay")
                plt.xlabel("Q (A^-1)")
                plt.ylabel("dS_2")
                plt.legend()
                plt.show()

    def plot_aniso_signal(self):
        for idx_delay in self.svd_delay_range:
            plt.plot(self.first_q_val, self.all_delay_aniso[idx_delay], label=str(idx_delay + 1) + "-th delay")
            if (idx_delay % self.one_fig_num_of_plot) == 0:
                plt.title("anisotropic signal of multiple delay")
                plt.xlabel("Q (A^-1)")
                plt.ylabel("dS_2")
                plt.legend()
                plt.show()

    def aniso_svd(self):
        singular_show_num = 10
        singular_cut_num = 4
        svd_result_root = "../results/anisotropy/svd/"
        outfile_family_name = "run53-aniso"
        right_time_delay_list = [-3, -2, -1.5, -1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8, 2, 2.5, 3, 4, 5, 7, 10, 13, 18, 24, 32, 42, 56, 75, 100, 130, 180, 240, 320, 420, 560, 750, 1000, 1300, 1800, 2400]

        all_aniso_data = np.transpose(np.array(self.all_delay_aniso))
        anisoSVD = SVDCalc(all_aniso_data)
        nowSingVal = anisoSVD.calc_svd()

        print(nowSingVal[:singular_show_num])
        singular_data_y = nowSingVal[:singular_show_num]
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

        bigSingVal = nowSingVal[:singular_cut_num]
        print(bigSingVal)

        print("left", anisoSVD.leftVec.shape)
        print("right", anisoSVD.rightVecTrans.shape)
        anisoSVD.pick_meaningful_data(singular_cut_num)
        print("left", anisoSVD.meanLeftVec.shape)
        print("right", anisoSVD.meanRightVec.shape)

        '''
        right_label = []
        for idx_delay in range(54):
            now_label = str(idx_delay + 1) + "-th delay"
            right_label.append(now_label)
        '''

        lsv_title = outfile_family_name + " LSV plot"
        rsv_title = outfile_family_name + " RSV plot"
        anisoSVD.plot_left_vec_with_x_val(graph_title=lsv_title, x_val=self.first_q_val)
        anisoSVD.plot_right_vec_with_x_text(graph_title=rsv_title, x_text=right_time_delay_list)

        sVal_file_name = outfile_family_name + "_SingVal.dat"
        rsv_file_name = outfile_family_name + "_RSV.dat"
        lsv_file_name = outfile_family_name + "_LSV.dat"

        sValOutFp = open((svd_result_root + sVal_file_name), 'w')
        rsvOutFp = open((svd_result_root + rsv_file_name), 'w')
        lsvOutFp = open((svd_result_root + lsv_file_name), 'w')

        anisoSVD.file_output_singular_value(sValOutFp)
        anisoSVD.file_output_singular_vectors_with_label(lsvOutFp, rsvOutFp, leftLableName="energy", leftLabel=self.first_q_val, rightLabelName="substrate", rightLabel=right_time_delay_list)

        sValOutFp.close()
        rsvOutFp.close()
        lsvOutFp.close()

    def data_svd(self, data_for_svd, data_id, file_family_name="run53", singular_show_num=10, singular_cut_num=5):
        svd_result_root = "../results/anisotropy/svd/"
        outfile_family_name = file_family_name + "-" + data_id
        right_time_delay_list = [-3, -2, -1.5, -1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8, 2, 2.5, 3, 4, 5, 7, 10, 13, 18, 24, 32, 42, 56, 75, 100, 130, 180, 240, 320, 420, 560, 750, 1000, 1300, 1800, 2400]

        now_data = np.transpose(np.array(data_for_svd))
        dataSVD = SVDCalc(now_data)
        nowSingVal = dataSVD.calc_svd()

        print(nowSingVal[:singular_show_num])
        singular_data_y = nowSingVal[:singular_show_num]
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

        bigSingVal = nowSingVal[:singular_cut_num]
        print(bigSingVal)

        print("left", dataSVD.leftVec.shape)
        print("right", dataSVD.rightVecTrans.shape)
        dataSVD.pick_meaningful_data(singular_cut_num)
        print("left", dataSVD.meanLeftVec.shape)
        print("right", dataSVD.meanRightVec.shape)

        lsv_title = outfile_family_name + " LSV plot"
        rsv_title = outfile_family_name + " RSV plot"
        dataSVD.plot_left_vec_with_x_val(graph_title=lsv_title, x_val=self.first_q_val)
        dataSVD.plot_right_vec_with_x_text(graph_title=rsv_title, x_text=right_time_delay_list)

        sVal_file_name = outfile_family_name + "_SingVal.dat"
        rsv_file_name = outfile_family_name + "_RSV.dat"
        lsv_file_name = outfile_family_name + "_LSV.dat"

        sValOutFp = open((svd_result_root + sVal_file_name), 'w')
        rsvOutFp = open((svd_result_root + rsv_file_name), 'w')
        lsvOutFp = open((svd_result_root + lsv_file_name), 'w')

        dataSVD.file_output_singular_value(sValOutFp)
        dataSVD.file_output_singular_vectors_with_label(lsvOutFp, rsvOutFp, leftLableName="energy", leftLabel=self.first_q_val, rightLabelName="substrate", rightLabel=right_time_delay_list)

        sValOutFp.close()
        rsvOutFp.close()
        lsvOutFp.close()

