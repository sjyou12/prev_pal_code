import numpy as np
import copy
import matplotlib.pyplot as plt
from palxfel_scatter.SVDCalc import SVDCalc
from datetime import datetime
import os

common_file_root = "/xfel/ffs/dat/ue_230427_FXL/analysis/results/"
azi_file_common_root = common_file_root + "whole_run_diff/"
file_out_root = common_file_root + "anisotropy/Anal_result_dat_file/"

# TODO: change values below
# avg_run_list = [20, 66, 98, 111, 153]  # 168 TW : done
# avg_run_list = [25, 68, 100, 114, 155]  # 130 TW : done
# avg_run_list = [28, 70, 116, 157]  # 100 TW avg done
# avg_run_list = [29, 72, 128, 159]  # 70 TW : done
# avg_run_list = [57, 74, 130, 161]  # 50 TW : done
# avg_run_list = [59, 76, 132, 177, 187]  # 40 TW : 163
avg_run_list = [45, 78, 134, 165, 211, 219]  # 30 TW : 229 will be added
# avg_run_list = [47, 80, 141, 167]  # 20 TW : done
# avg_run_list = [49, 82, 143, 169, 179]  # 15 TW : done
# avg_run_list = [51, 84, 146, 171, 185, 193]  # 10 TW, isoheat=2nd : 193
# avg_run_list = [204, 208, 209, 213, 215, 217, 221, 223, 225] # 7.5 TW : done
# avg_run_list = [53, 86, 90, 148, 173, 183, 191]  # 5 TW, isoheat=2nd? : done
# avg_run_list = [55, 88, 92, 151, 175, 181]#, 189]  # 3 TW, isoheat=1st? : 189
#
fluence_Val = 3
timestamp = ["2022-11-08-17hr.17min.50sec", "2022-11-08-17hr.18min.08sec"]
right_time_delay_list = [-3,-1,-0.8,-0.6,-0.4,-0.35,-0.3,-0.28,-0.26,-0.24,-0.22,-0.2,-0.18,-0.16,-0.14,-0.12,-0.1,-0.08,-0.06,-0.04,-0.02,0,0.02,0.04,0.06,0.08,0.1,0.12,0.14,0.16,0.18,0.2,0.22,0.24,0.26,0.28,0.3,0.32,0.34,0.36,0.38,0.4,0.42,0.46,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2,2.2,2.4,2.6,2.8,3,3.5,4,5,6,8,10,17,32,56,100,170,320,560,1000,1700,2500,3000]
# if len(avg_run_list) == 2:
now_file_family_name = "{}TW_avg".format(fluence_Val)
# elif len(avg_run_list) == 3:
#     now_file_family_name = "run{0:04d}_{1:04d}_{2:04d}_avg".format(56, 57, 58)

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

def calc_weight(azi_file_root, avg_run_list):
    common_delay_num = 0
    whole_run_delay_weight_list = []
    for each_run in avg_run_list:
        cutted_diff_file_path = azi_file_root + "whole_run_cutted_diff_run{}.npy".format(each_run)
        now_run_diff = np.load(cutted_diff_file_path, allow_pickle=True)[0]
        if common_delay_num == 0:
            common_delay_num = len(now_run_diff)
        now_run_delay_weight_list = []
        for idx_delay in range(common_delay_num):
            now_delay_used_shot_num = now_run_diff[idx_delay].shape[0]
            print("run{0}-delay{1} : {2} element used".format(each_run, idx_delay + 1, now_delay_used_shot_num))
            now_run_delay_weight_list.append(now_delay_used_shot_num)
        whole_run_delay_weight_list.append(now_run_delay_weight_list)
    whole_run_delay_weight_list = np.array(whole_run_delay_weight_list)
    return whole_run_delay_weight_list

def azi_avg(azi_file_root, avg_run_list):
    whole_run_diff = []
    for each_run in avg_run_list:
        cutted_diff_file_path = azi_file_root + "whole_run_cutted_diff_run{}.npy".format(each_run)
        now_run_diff = np.load(cutted_diff_file_path)
        whole_run_diff.append(now_run_diff[0])

    print(np.shape(whole_run_diff))

    common_delay_num = len(whole_run_diff[0])
    each_delay_conc_data = copy.deepcopy(whole_run_diff[0])
    print("before shape : ", np.shape(each_delay_conc_data))
    for idx_run, each_run_data in enumerate(whole_run_diff):
        if idx_run == 0:
            continue
        for idx_delay in range(common_delay_num):
            each_delay_conc_data[idx_delay] = np.concatenate((each_delay_conc_data[idx_delay],each_run_data[idx_delay]), axis=0)

    all_delay_avg = []
    for each_delay_data in each_delay_conc_data:
        delay_avg_data = np.average(each_delay_data, axis=0)
        all_delay_avg.append(delay_avg_data)

def aniso_avg(avg_run_list, each_run_delay_weight):
    common_delay_range = range(len(right_time_delay_list))
    first_q_val = []
    tp_delay_weight = np.transpose(each_run_delay_weight)
    all_delay_aniso_avg = []

    for idx_delay in common_delay_range:
        each_run_aniso_list = []
        each_run_iso_list = []
        run_idx = 0
        file_out_path = file_out_root + now_file_family_name + "/"
        dat_file_out_path = file_out_path + now_file_family_name + "_aniso.dat"
        for now_run_num in avg_run_list:
            file_load_root = "../../results/anisotropy/anal_result/run_{0:05d}/run{0}_delay{1}".format(now_run_num, idx_delay + 1)
            q_val_file_name = file_load_root + "_qval.npy"
            iso_file_name = file_load_root + "_iso.npy"
            aniso_file_name = file_load_root + "_aniso.npy"
            now_q_val_arr = np.load(q_val_file_name)
            now_iso_arr = np.load(iso_file_name)
            now_aniso_arr = np.load(aniso_file_name)
            if idx_delay == 0:
                first_q_val = now_q_val_arr
            else:
                q_val_diff = now_q_val_arr - first_q_val
                diff_sum = np.sum(q_val_diff)
                if diff_sum > 0:
                    print("q_val difference :", diff_sum, "[ in " + str(idx_delay + 1) + "-th delay")
            each_run_aniso_list.append(now_aniso_arr)
            each_run_iso_list.append(now_iso_arr)
            run_idx += 1
        now_weight = tp_delay_weight[idx_delay]
        now_delay_aniso_avg = np.average(each_run_aniso_list, weights=now_weight, axis=0)
        all_delay_aniso_avg.append(now_delay_aniso_avg)
    if os.path.isdir(file_out_path):
        pass
    else:
        os.makedirs(file_out_path)
    dat_file_out(first_q_val, all_delay_aniso_avg, dat_file_out_path)
    for idx_delay, each_delay_aniso_avg in enumerate(all_delay_aniso_avg):
        np.save(file_out_path + now_file_family_name + "_delay{0}_aniso.npy".format(idx_delay + 1) ,each_delay_aniso_avg)
        if idx_delay == 0:
            np.save(file_out_path + now_file_family_name + "_delay{0}_aniso.npy".format(idx_delay + 1), each_delay_aniso_avg)
    return all_delay_aniso_avg, first_q_val

def iso_avg(avg_run_list, each_run_delay_weight):
    common_delay_range = range(len(right_time_delay_list))
    first_q_val = []
    tp_delay_weight = np.transpose(each_run_delay_weight)
    all_delay_iso_avg = []
    for idx_delay in common_delay_range:
        each_run_aniso_list = []
        each_run_iso_list = []
        run_idx = 0
        for now_run_num in avg_run_list:
            file_load_root = "../../results/anisotropy/anal_result/run_{0:05d}/run{0}_delay{1}".format(now_run_num, idx_delay + 1)
            q_val_file_name = file_load_root + "_qval.npy"
            iso_file_name = file_load_root + "_iso.npy"
            aniso_file_name = file_load_root + "_aniso.npy"
            now_q_val_arr = np.load(q_val_file_name)
            now_iso_arr = np.load(iso_file_name)
            now_aniso_arr = np.load(aniso_file_name)
            if idx_delay == 0:
                first_q_val = now_q_val_arr
            else:
                q_val_diff = now_q_val_arr - first_q_val
                diff_sum = np.sum(q_val_diff)
                if diff_sum > 0:
                    print("q_val difference :", diff_sum, "[ in " + str(idx_delay + 1) + "-th delay")
            each_run_aniso_list.append(now_aniso_arr)
            each_run_iso_list.append(now_iso_arr)
            run_idx += 1
        now_weight = tp_delay_weight[idx_delay]
        now_delay_iso_avg = np.average(each_run_iso_list, weights=now_weight, axis=0)
        all_delay_iso_avg.append(now_delay_iso_avg)
    file_out_path = file_out_root + now_file_family_name + "/"
    dat_file_out_path = file_out_path + now_file_family_name + "_iso.dat"
    dat_file_out(first_q_val, all_delay_iso_avg, dat_file_out_path)
    for idx_delay, each_delay_aniso_avg in enumerate(all_delay_iso_avg):
        np.save(file_out_path + now_file_family_name + "_delay{0}_iso.npy".format(idx_delay + 1) ,each_delay_aniso_avg)
        if idx_delay == 0:
            np.save(file_out_path + now_file_family_name + "_delay{0}_iso.npy".format(idx_delay + 1), each_delay_aniso_avg)
    return all_delay_iso_avg, first_q_val

def dat_file_out(q_val_list, out_file_list, file_out_path):
    dat_file = open(file_out_path, 'w')
    dat_file.write("q_val" + "\t")
    for delay_idx in range(len(out_file_list)):
        dat_file.write(str(delay_idx + 1) + "-th delay" + "\t")
    dat_file.write("\n")
    for line_num in range(len(q_val_list)):
        dat_file.write(str(q_val_list[line_num]) + "\t")
        for delay_idx in range(len(out_file_list)):
            dat_file.write(str(out_file_list[delay_idx][line_num]) + "\t")
        dat_file.write("\n")

now_time = datetime.now().strftime('%Y-%m-%d-%Hhr.%Mmin.%Ssec')
# now_time = "2022-02-18-16hr.45min.44sec"
each_run_delay_weight = calc_weight(azi_file_common_root, avg_run_list)
print(each_run_delay_weight)
weighted_avg_aniso, common_q_val = aniso_avg(avg_run_list, each_run_delay_weight)
weighted_avg_iso, common_q_val = iso_avg(avg_run_list, each_run_delay_weight)


def aniso_svd_cut_q_range(aniso_data, common_q_val, time_delay_list, file_family_name, start_q=0.5, end_q=3.5, singular_cut_num=10):
    singular_show_num = 10
    #singular_cut_num = 4

    svd_result_root = common_file_root + "anisotropy/svd/{}TW_avg/".format(fluence_Val)

    if os.path.isdir(svd_result_root):
        pass
    else:
        os.makedirs(svd_result_root)
    outfile_family_name = file_family_name + "-aniso-cut"
    right_time_delay_list = time_delay_list

    cutted_q_val = common_q_val[(common_q_val >= start_q) & (common_q_val <= end_q)]
    all_aniso_data = np.transpose(np.array(aniso_data))
    cutted_aniso_data = all_aniso_data[(common_q_val>= start_q) & (common_q_val <= end_q)]
    anisoSVD = SVDCalc(cutted_aniso_data)
    nowSingVal = anisoSVD.calc_svd()

    print(nowSingVal[:singular_show_num])
    singular_data_y = nowSingVal[:singular_show_num]
    singular_data_y_log = np.log(singular_data_y)
    singular_data_x = range(1, len(singular_data_y) + 1)

    def plot_singular_value(data_x, data_y, data_y_log):
        color_r = 'tab:red'
        fig, ax1 = plt.subplots()
        fig.suptitle("singular value of anisotropy")
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
    plot_rsv_linear_n_log_x_scale(right_time_delay_list, anisoSVD.meanRightVec, "time (ps)", "Intensity (a.u.)", 3,"Anisotropy-cut RSV (q 0.8~7)")

    sVal_file_name = outfile_family_name + "_SingVal.dat"
    rsv_file_name = outfile_family_name + "_RSV.dat"
    lsv_file_name = outfile_family_name + "_LSV.dat"

    sValOutFp = open((svd_result_root + sVal_file_name), 'w')
    rsvOutFp = open((svd_result_root + rsv_file_name), 'w')
    lsvOutFp = open((svd_result_root + lsv_file_name), 'w')

    anisoSVD.file_output_singular_value(sValOutFp)
    anisoSVD.file_output_singular_vectors_with_label(lsvOutFp, rsvOutFp, leftLableName="energy",
                                                     leftLabel=cutted_q_val, rightLabelName="substrate",
                                                     rightLabel=right_time_delay_list)

    sValOutFp.close()
    rsvOutFp.close()
    lsvOutFp.close()

def iso_svd_cut_q_range(iso_data, common_q_val, time_delay_list, file_family_name, start_q=0.5, end_q=3.5, singular_cut_num=10):
    singular_show_num = 10
    #singular_cut_num = 4

    svd_result_root = common_file_root + "anisotropy/svd/{}TW_avg/".format(fluence_Val)

    if os.path.isdir(svd_result_root):
        pass
    else:
        # os.makedirs(("../results/anisotropy/svd/run{0:04d}_{1:04d}_avg/run{0}_{1}_avg_{2}/".format(avg_run_list[0], avg_run_list[1], now_time)))
        os.makedirs(svd_result_root)
    outfile_family_name = file_family_name + "-iso-cut"
    right_time_delay_list = time_delay_list

    cutted_q_val = common_q_val[(common_q_val >= start_q) & (common_q_val <= end_q)]
    all_iso_data = np.transpose(np.array(iso_data))
    cutted_iso_data = all_iso_data[(common_q_val>= start_q) & (common_q_val <= end_q)]
    anisoSVD = SVDCalc(cutted_iso_data)
    nowSingVal = anisoSVD.calc_svd()

    print(nowSingVal[:singular_show_num])
    singular_data_y = nowSingVal[:singular_show_num]
    singular_data_y_log = np.log(singular_data_y)
    singular_data_x = range(1, len(singular_data_y) + 1)

    def plot_singular_value(data_x, data_y, data_y_log):
        color_r = 'tab:red'
        fig, ax1 = plt.subplots()
        fig.suptitle("singular value of isotropy")
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
    plot_rsv_linear_n_log_x_scale(right_time_delay_list, anisoSVD.meanRightVec, "time (ps)", "Intensity (a.u.)", 3,"Isotropy-cut RSV (q 0.8~7)")

    sVal_file_name = outfile_family_name + "_SingVal.dat"
    rsv_file_name = outfile_family_name + "_RSV.dat"
    lsv_file_name = outfile_family_name + "_LSV.dat"

    sValOutFp = open((svd_result_root + sVal_file_name), 'w')
    rsvOutFp = open((svd_result_root + rsv_file_name), 'w')
    lsvOutFp = open((svd_result_root + lsv_file_name), 'w')

    anisoSVD.file_output_singular_value(sValOutFp)
    anisoSVD.file_output_singular_vectors_with_label(lsvOutFp, rsvOutFp, leftLableName="energy",leftLabel=cutted_q_val, rightLabelName="substrate",rightLabel=right_time_delay_list)

    sValOutFp.close()
    rsvOutFp.close()
    lsvOutFp.close()

def data_svd(data_for_svd, data_id, singular_show_num=10, singular_cut_num=5):

    svd_result_root = common_file_root + "anisotropy/svd/{}TW_avg/".format(fluence_Val)
    if os.path.isdir(svd_result_root):
        pass
    else:
        os.makedirs(svd_result_root)
    # outfile_family_name = file_family_name + "-iso-cut"
    outfile_family_name = now_file_family_name + "-" + data_id

    now_data = np.transpose(np.array(data_for_svd))
    dataSVD = SVDCalc(now_data)
    nowSingVal = dataSVD.calc_svd()

    # if self.skipped_delay_list:
    #     temp_add_arr = [0]
    #     for idx in range(len(dataSVD.rightVecTrans)):
    #         temp_sliced_arr = dataSVD.rightVecTrans[idx][:self.skipped_delay_list[0]]
    #         for idx_skipped in range(len(self.skipped_delay_list)):
    #             if idx_skipped == 0:
    #                 temp_arr = np.hstack((temp_sliced_arr, temp_add_arr))
    #             else:
    #                 temp_arr = np.hstack((temp_arr, temp_add_arr))
    #         back_of_slice_point_arr = dataSVD.rightVecTrans[idx][self.skipped_delay_list[0]:]
    #         if idx == 0:
    #             dataSVD.rightVec = (np.hstack((temp_arr, back_of_slice_point_arr)))
    #         else:
    #             dataSVD.rightVec = np.vstack((dataSVD.rightVec, (np.hstack((temp_arr, back_of_slice_point_arr)))))
    #     dataSVD.rightVec = np.transpose(dataSVD.rightVec)
    # else:
    #     pass

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
    dataSVD.plot_left_vec_with_x_val(graph_title=lsv_title, x_val=common_q_val)
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
    dataSVD.file_output_singular_vectors_with_label(lsvOutFp, rsvOutFp, leftLableName="energy", leftLabel=common_q_val, rightLabelName="substrate", rightLabel=right_time_delay_list)

    sValOutFp.close()
    rsvOutFp.close()
    lsvOutFp.close()

data_svd(weighted_avg_aniso, "aniso", singular_cut_num=10)
data_svd(weighted_avg_aniso, "iso", singular_cut_num=10)
aniso_svd_cut_q_range(weighted_avg_aniso, common_q_val, right_time_delay_list, now_file_family_name, 0.8, 7, singular_cut_num=3)
iso_svd_cut_q_range(weighted_avg_iso, common_q_val, right_time_delay_list, now_file_family_name, 0.8, 7, singular_cut_num=3)
# print("This file's time stamp is " + now_time)
print("Avaerage for {0} TW/cm2 is finished \n Total {1} runs are used".format(fluence_Val, len(avg_run_list)))