from palxfel_scatter.SVDCalc import SVDCalc
import numpy as np
import matplotlib.pyplot as plt
import os


def plot_rsv_linear_n_log_x_scale(x_data, y_data, x_label, y_label, scale_cut_x_val, suptitle):
    x_data = np.array(x_data)
    rsv_arr = np.array(y_data)
    fig, axs = plt.subplots(ncols=2, figsize=(8, 4), gridspec_kw={'wspace':0})
    linear_ax = axs[0]
    log_ax = axs[1]
    for sing_idx in range(len(np.transpose(rsv_arr))):
        # y_data = rsv_arr[sing_idx][:len(x_data)]
        y_data = rsv_arr[:, sing_idx]
        linear_range_mask = (x_data <= scale_cut_x_val)
        log_range_mask = (x_data >= scale_cut_x_val)
        linear_x_data = x_data[linear_range_mask]
        linear_y_data = y_data[linear_range_mask]
        log_x_data = x_data[log_range_mask]
        log_y_data = y_data[log_range_mask]

        linear_ax.plot(linear_x_data, linear_y_data)
        log_ax.plot(log_x_data, log_y_data, label="rightvec{}".format(sing_idx+1))

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
class AnisoSVD():
    #svd_delay_range = range()
    one_fig_num_of_plot = 5
    now_run_num = 0
    file_family_name = None

    def __init__(self):
        self.first_q_val = []
        self.all_delay_iso = []
        self.all_delay_aniso = []
        self.time_delay_list = []
        self.skipped_delay_list = []
        self.svd_delay_range = 0
        self.time_stamp = None

    def read_aniso_results(self, merge_multi_run, merge_run_list, timeStamp, time_delay_list, run_num=4):
        self.time_delay_list = time_delay_list
        self.svd_delay_range = range(len(time_delay_list))
        self.now_run_num = run_num
        self.time_stamp = timeStamp
        if merge_multi_run:
            self.file_family_name = "run{0}_{1}".format(merge_run_list[0],merge_run_list[1])
        else:
            self.file_family_name = "run" + str(run_num)
        for idx_delay in self.svd_delay_range:
            if merge_multi_run:
                file_load_root = "../../results/anisotropy/anal_result/run{0}_{1}/run{0}_{1}_{3}/run{0}_{1}_delay{2}".format(merge_run_list[0],merge_run_list[1],idx_delay + 1, timeStamp)
            else:
                # file_load_root = "../../results/anisotropy/anal_result/run_{0:05d}/run{0}_{2}/run{0}_delay{1}".format(self.now_run_num, idx_delay + 1, timeStamp)
                file_load_root = "../../results/anisotropy/anal_result/run_{0:05d}/run{0}_delay{1}".format(self.now_run_num, idx_delay + 1)
            q_val_file_name = file_load_root + "_qval.npy"
            iso_file_name = file_load_root + "_iso.npy"
            aniso_file_name = file_load_root + "_aniso.npy"
            try:
                now_q_val_arr = np.load(q_val_file_name)
            except:
                print("no q value at ", q_val_file_name)
                #now_q_val_arr = np.zeros_like(self.first_q_val)
                self.skipped_delay_list.append(idx_delay) # zero-base
                continue
            try:
                now_iso_arr = np.load(iso_file_name)
            except:
                print("no iso val at ", iso_file_name)
                #now_iso_arr = np.zeros_like(self.first_q_val)
                self.skipped_delay_list.append(idx_delay) # zero-base
                continue
            try:
                now_aniso_arr = np.load(aniso_file_name)
            except:
                print("no aniso val at ", aniso_file_name)
                #now_aniso_arr = np.zeros_like(self.first_q_val)
                self.skipped_delay_list.append(idx_delay) # zero-base
                continue

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

    def plot_aniso_signal(self):
        if len(self.skipped_delay_list) != 0:
            skipped_delay_cnt = 0
            for idx_delay in self.svd_delay_range:
                if idx_delay in self.skipped_delay_list:
                    skipped_delay_cnt = skipped_delay_cnt + 1
                    continue
                else:
                    plt.plot(self.first_q_val, self.all_delay_aniso[idx_delay - skipped_delay_cnt], label=str(idx_delay + 1) + "-th delay")
                    if (idx_delay % self.one_fig_num_of_plot) == 0:
                        plt.title("anisotropic signal of multiple delay")
                        plt.xlabel("Q (A^-1)")
                        plt.ylabel("dS_2")
                        plt.legend()
                        plt.show()
                    elif idx_delay == len(self.svd_delay_range) - len(self.skipped_delay_list):
                        plt.title("isotropic signal of multiple delay")
                        plt.xlabel("Q (A^-1)")
                        plt.ylabel("dS_2")
                        plt.legend()
                        plt.show()
        else:
            for idx_delay in self.svd_delay_range:
                plt.plot(self.first_q_val, self.all_delay_aniso[idx_delay], label=str(idx_delay + 1) + "-th delay")
                if (idx_delay % self.one_fig_num_of_plot) == 0:
                    plt.title("anisotropic signal of multiple delay")
                    plt.xlabel("Q (A^-1)")
                    plt.ylabel("dS_2")
                    plt.legend()
                    plt.show()

    def plot_iso_signal(self):
        if len(self.skipped_delay_list) != 0:
            for idx_delay in range(len(self.svd_delay_range) - len(self.skipped_delay_list)):
                if idx_delay >= self.skipped_delay_list[0]:
                    label_delay = idx_delay + len(self.skipped_delay_list)
                    plt.plot(self.first_q_val, self.all_delay_iso[idx_delay], label=str(label_delay + 1) + "-th delay")
                else:
                    plt.plot(self.first_q_val, self.all_delay_iso[idx_delay], label=str(idx_delay + 1) + "-th delay")
                if (idx_delay % self.one_fig_num_of_plot) == 0:
                    plt.title("isotropic signal of multiple delay")
                    plt.xlabel("Q (A^-1)")
                    plt.ylabel("dS_2")
                    plt.legend()
                    plt.show()
                elif idx_delay == len(self.svd_delay_range) - len(self.skipped_delay_list):
                    plt.title("isotropic signal of multiple delay")
                    plt.xlabel("Q (A^-1)")
                    plt.ylabel("dS_2")
                    plt.legend()
                    plt.show()
        else:
            for idx_delay in self.svd_delay_range:
                plt.plot(self.first_q_val, self.all_delay_iso[idx_delay], label=str(idx_delay + 1) + "-th delay")
                if (idx_delay % self.one_fig_num_of_plot) == 0:
                    plt.title("isotropic signal of multiple delay")
                    plt.xlabel("Q (A^-1)")
                    plt.ylabel("dS_2")
                    plt.legend()
                    plt.show()

    def aniso_svd(self):
        singular_show_num = 10
        singular_cut_num = 4
        svd_result_root = "../../results/anisotropy/svd/"
        outfile_family_name = self.file_family_name + "-aniso"
        right_time_delay_list = self.time_delay_list

        all_aniso_data = np.transpose(np.array(self.all_delay_aniso))
        anisoSVD = SVDCalc(all_aniso_data)
        nowSingVal = anisoSVD.calc_svd()

        if self.skipped_delay_list:
            temp_add_arr = [0]
            for idx in range(len(anisoSVD.rightVecTrans)):
                temp_sliced_arr = anisoSVD.rightVecTrans[idx][:self.skipped_delay_list[0]]
                for idx_skipped in range(len(self.skipped_delay_list)):
                    if idx_skipped == 0:
                        temp_arr = np.hstack((temp_sliced_arr, temp_add_arr))
                    else:
                        temp_arr = np.hstack((temp_arr, temp_add_arr))
                back_of_slice_point_arr = anisoSVD.rightVecTrans[idx][self.skipped_delay_list[0]:]
                if idx == 0:
                    anisoSVD.rightVec = (np.hstack((temp_arr, back_of_slice_point_arr)))
                else:
                    anisoSVD.rightVec = np.vstack((anisoSVD.rightVec, (np.hstack((temp_arr, back_of_slice_point_arr)))))
            #anisoSVD.rightVec = np.transpose(anisoSVD.rightVec)
        else:
            pass

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

    def data_svd(self, data_for_svd, data_id, singular_show_num=10, singular_cut_num=5):
        svd_result_root = "../../results/anisotropy/svd/run{0:04d}/".format(self.now_run_num)
        if os.path.isdir(svd_result_root):
            pass
        else:
            os.makedirs(("../../results/anisotropy/svd/run{0:04d}/".format(self.now_run_num)))
        outfile_family_name = self.file_family_name + "-" + data_id
        right_time_delay_list = self.time_delay_list

        now_data = np.transpose(np.array(data_for_svd))
        dataSVD = SVDCalc(now_data)
        nowSingVal = dataSVD.calc_svd()

        if self.skipped_delay_list:
            temp_add_arr = [0]
            for idx in range(len(dataSVD.rightVecTrans)):
                temp_sliced_arr = dataSVD.rightVecTrans[idx][:self.skipped_delay_list[0]]
                for idx_skipped in range(len(self.skipped_delay_list)):
                    if idx_skipped == 0:
                        temp_arr = np.hstack((temp_sliced_arr, temp_add_arr))
                    else:
                        temp_arr = np.hstack((temp_arr, temp_add_arr))
                back_of_slice_point_arr = dataSVD.rightVecTrans[idx][self.skipped_delay_list[0]:]
                if idx == 0:
                    dataSVD.rightVec = (np.hstack((temp_arr, back_of_slice_point_arr)))
                else:
                    dataSVD.rightVec = np.vstack((dataSVD.rightVec, (np.hstack((temp_arr, back_of_slice_point_arr)))))
            dataSVD.rightVec = np.transpose(dataSVD.rightVec)
        else:
            pass

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
        try:
            dataSVD.plot_right_vec_with_x_text(graph_title=rsv_title, x_text=right_time_delay_list)
        except:
            print("right vect save error")

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

    def aniso_svd_cut_q_range(self, start_q=0.5, end_q=4, singular_cut_num=4):
        singular_show_num = 10
        #singular_cut_num = 4
        # svd_result_root = "../../results/anisotropy/svd/run{0:04d}/run{0}_{1}/".format(self.now_run_num, self.time_stamp)
        svd_result_root = "../../results/anisotropy/svd/run{0:04d}/".format(self.now_run_num)
        if os.path.isdir(svd_result_root):
            pass
        else:
            # os.makedirs(("../../results/anisotropy/svd/run{0:04d}/run{0}_{1}/".format(self.now_run_num, self.time_stamp)))
            os.makedirs(("../../results/anisotropy/svd/run{0:04d}/".format(self.now_run_num)))
        outfile_family_name = self.file_family_name + "-aniso-cut"
        right_time_delay_list = self.time_delay_list

        cutted_q_val = self.first_q_val[(self.first_q_val >= start_q) & (self.first_q_val <= end_q)]
        all_aniso_data = np.transpose(np.array(self.all_delay_aniso))
        cutted_aniso_data = all_aniso_data[(self.first_q_val >= start_q) & (self.first_q_val <= end_q)]
        anisoSVD = SVDCalc(cutted_aniso_data)
        nowSingVal = anisoSVD.calc_svd()

        if self.skipped_delay_list:
            temp_add_arr = [0]
            for idx in range(len(anisoSVD.rightVecTrans)):
                temp_sliced_arr = anisoSVD.rightVecTrans[idx][:self.skipped_delay_list[0]]
                for idx_skipped in range(len(self.skipped_delay_list)):
                    if idx_skipped == 0:
                        temp_arr = np.hstack((temp_sliced_arr, temp_add_arr))
                    else:
                        temp_arr = np.hstack((temp_arr, temp_add_arr))
                back_of_slice_point_arr = anisoSVD.rightVecTrans[idx][self.skipped_delay_list[0]:]
                if idx == 0:
                    anisoSVD.rightVec = (np.hstack((temp_arr, back_of_slice_point_arr)))
                else:
                    anisoSVD.rightVec = np.vstack((anisoSVD.rightVec, (np.hstack((temp_arr, back_of_slice_point_arr)))))
            anisoSVD.rightVec = np.transpose(anisoSVD.rightVec)
        else:
            pass

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
        anisoSVD.plot_left_vec_with_x_val(graph_title=lsv_title, x_val=cutted_q_val)
        # anisoSVD.plot_right_vec_with_x_text(graph_title=rsv_title, x_text=right_time_delay_list)
        plot_rsv_linear_n_log_x_scale(right_time_delay_list, anisoSVD.meanRightVec, "time (ps)", "Intensity (a.u.)", 3, "Anisotropy RSV (q 0.8~7)")

        sVal_file_name = outfile_family_name + "_SingVal.dat"
        rsv_file_name = outfile_family_name + "_RSV.dat"
        lsv_file_name = outfile_family_name + "_LSV.dat"

        sValOutFp = open((svd_result_root + sVal_file_name), 'w')
        rsvOutFp = open((svd_result_root + rsv_file_name), 'w')
        lsvOutFp = open((svd_result_root + lsv_file_name), 'w')

        anisoSVD.file_output_singular_value(sValOutFp)
        anisoSVD.file_output_singular_vectors_with_label(lsvOutFp, rsvOutFp, leftLableName="energy", leftLabel=cutted_q_val, rightLabelName="substrate", rightLabel=right_time_delay_list)

        print("file save success in {}".format(svd_result_root))


        sValOutFp.close()
        rsvOutFp.close()
        lsvOutFp.close()

    def iso_svd_cut_q_range(self, start_q=0.5, end_q=4,singular_cut_num=4):
        singular_show_num = 10
        #singular_cut_num = 4
        # svd_result_root = "../../results/anisotropy/svd/run{0:04d}/run{0}_{1}/".format(self.now_run_num, self.time_stamp)
        svd_result_root = "../../results/anisotropy/svd/run{0:04d}/".format(self.now_run_num)
        if os.path.isdir(svd_result_root):
            pass
        else:
            # os.makedirs(("../../results/anisotropy/svd/run{0:04d}/run{0}_{1}/".format(self.now_run_num, self.time_stamp)))
            os.makedirs(("../../results/anisotropy/svd/run{0:04d}/".format(self.now_run_num)))
        outfile_family_name = self.file_family_name + "-iso-cut"
        right_time_delay_list = self.time_delay_list

        cutted_q_val = self.first_q_val[(self.first_q_val >= start_q) & (self.first_q_val <= end_q)]
        all_iso_data = np.transpose(np.array(self.all_delay_iso))
        cutted_iso_data = all_iso_data[(self.first_q_val >= start_q) & (self.first_q_val <= end_q)]
        anisoSVD = SVDCalc(cutted_iso_data)
        nowSingVal = anisoSVD.calc_svd()

        if self.skipped_delay_list:
            temp_add_arr = [0]
            for idx in range(len(anisoSVD.rightVecTrans)):
                temp_sliced_arr = anisoSVD.rightVecTrans[idx][:self.skipped_delay_list[0]]
                for idx_skipped in range(len(self.skipped_delay_list)):
                    if idx_skipped == 0:
                        temp_arr = np.hstack((temp_sliced_arr, temp_add_arr))
                    else:
                        temp_arr = np.hstack((temp_arr, temp_add_arr))
                back_of_slice_point_arr = anisoSVD.rightVecTrans[idx][self.skipped_delay_list[0]:]
                if idx == 0:
                    anisoSVD.rightVec = (np.hstack((temp_arr, back_of_slice_point_arr)))
                else:
                    anisoSVD.rightVec = np.vstack((anisoSVD.rightVec, (np.hstack((temp_arr, back_of_slice_point_arr)))))
            anisoSVD.rightVec = np.transpose(anisoSVD.rightVec)
        else:
            pass

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
        anisoSVD.plot_left_vec_with_x_val(graph_title=lsv_title, x_val=cutted_q_val)
        # anisoSVD.plot_right_vec_with_x_text(graph_title=rsv_title, x_text=right_time_delay_list)
        plot_rsv_linear_n_log_x_scale(right_time_delay_list, anisoSVD.meanRightVec, "time (ps)", "Intensity (a.u.)", 3, "Isotropy RSV (q 0.8~7)")
        plt.plot(anisoSVD.meanRightVec[:, 0] * bigSingVal[0])
        plt.title("run {}".format(self.now_run_num))
        plt.xlabel("time point")
        plt.show()
        sVal_file_name = outfile_family_name + "_SingVal.dat"
        rsv_file_name = outfile_family_name + "_RSV.dat"
        lsv_file_name = outfile_family_name + "_LSV.dat"

        sValOutFp = open((svd_result_root + sVal_file_name), 'w')
        rsvOutFp = open((svd_result_root + rsv_file_name), 'w')
        lsvOutFp = open((svd_result_root + lsv_file_name), 'w')

        anisoSVD.file_output_singular_value(sValOutFp)
        anisoSVD.file_output_singular_vectors_with_label(lsvOutFp, rsvOutFp, leftLableName="energy", leftLabel=cutted_q_val, rightLabelName="substrate", rightLabel=right_time_delay_list)

        print("file save success in {}".format(svd_result_root))

        sValOutFp.close()
        rsvOutFp.close()
        lsvOutFp.close()