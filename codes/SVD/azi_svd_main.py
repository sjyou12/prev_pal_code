import matplotlib
import matplotlib.pyplot as plt
import numpy as np

now_run_num = 4

common_root = "../results/"
now_run_cutted_diff_path = common_root + "whole_run_diff/whole_run_cutted_diff_run{}.npy".format(now_run_num)
now_run_cutted_diff = np.load(now_run_cutted_diff_path)

first_q_val_path = common_root + "q_val_run3.npy"
first_q_val = np.load(first_q_val_path)

print(now_run_cutted_diff.shape)

plt.title("run{} whole delay diff (cutted)".format(now_run_num))
whole_delay_avg = []
for idx_delay, each_delay in enumerate(now_run_cutted_diff[0]):
    now_avg = np.average(each_delay, axis=0)
    whole_delay_avg.append(now_avg)


from SVDCalc import SVDCalc

singular_show_num = 10
singular_cut_num = 4
outfile_family_name = "run{}-azi".format(now_run_num)
right_time_delay_list = [-2, -1.9, -1.8, -1.7, -1.6, -1.5, -1.4, -1.3, -1.2, -1.1, -1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2]

svd_result_root = "../results/svd/"


all_azi_diff_signal = np.transpose(np.array(whole_delay_avg))

aziSVD = SVDCalc(all_azi_diff_signal)
nowSingVal = aziSVD.calc_svd()

print(nowSingVal[:singular_show_num])

aziSVD.pick_meaningful_data(singular_cut_num)
print("left", aziSVD.meanLeftVec.shape)
print("right", aziSVD.meanRightVec.shape)

lsv_title = outfile_family_name + " LSV plot"
rsv_title = outfile_family_name + " RSV plot"
aziSVD.plot_left_vec_with_x_val(graph_title=lsv_title, x_val=first_q_val)
plt.figure(figsize=(12,9))
aziSVD.plot_right_vec_with_x_text(graph_title=rsv_title, x_text=right_time_delay_list)

sVal_file_name = outfile_family_name + "_SingVal.dat"
rsv_file_name = outfile_family_name + "_RSV.dat"
lsv_file_name = outfile_family_name + "_LSV.dat"

sValOutFp = open((svd_result_root + sVal_file_name), 'w')
rsvOutFp = open((svd_result_root + rsv_file_name), 'w')
lsvOutFp = open((svd_result_root + lsv_file_name), 'w')

aziSVD.file_output_singular_value(sValOutFp)
aziSVD.file_output_singular_vectors_with_label(lsvOutFp, rsvOutFp, leftLableName="q_val", leftLabel=first_q_val, rightLabelName="time", rightLabel=right_time_delay_list)

sValOutFp.close()
rsvOutFp.close()
lsvOutFp.close()