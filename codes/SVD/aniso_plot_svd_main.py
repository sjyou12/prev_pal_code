from palxfel_scatter.anisotropy.AnisoSVD import AnisoSVD
import os
import numpy as np

merge_multi_run = False
merge_run_list = [19, 20]
# TODO: change the run number
now_run_num = 163
right_time_delay_list = [-3,-1,-0.8,-0.6,-0.4,-0.35,-0.3,-0.28,-0.26,-0.24,-0.22,-0.2,-0.18,-0.16,-0.14,-0.12,-0.1,-0.08,-0.06,-0.04,-0.02,0,0.02,0.04,0.06,0.08,0.1,0.12,0.14,0.16,0.18,0.2,0.22,0.24,0.26,0.28,0.3,0.32,0.34,0.36,0.38,0.4,0.42,0.46,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2,2.2,2.4,2.6,2.8,3,3.5,4,5,6,8,10,17,32,56,100,170,320,560,1000,1700,2500,3000]
# right_time_delay_list = np.ones(51)*100
# right_time_delay_list = [-3,-0.5,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5,0.7,1,1.7,3.2,5.6,10,32,100,320,1000,3000]
timeStamp = "2023-04-27-22hr.26min.10sec"

num_delay = len(right_time_delay_list)
anisoSVDanalyzer = AnisoSVD()
anisoSVDanalyzer.read_aniso_results(merge_multi_run, merge_run_list, timeStamp, time_delay_list=right_time_delay_list,run_num=now_run_num)
# anisoSVDanalyzer.plot_iso_signal()
# anisoSVDanalyzer.plot_aniso_signal()

#anisoSVDanalyzer.aniso_svd()
# anisoSVDanalyzer.data_svd(anisoSVDanalyzer.all_delay_aniso, "aniso", singular_cut_num=3)
# anisoSVDanalyzer.data_svd(anisoSVDanalyzer.all_delay_iso, "iso", singular_cut_num=3)
anisoSVDanalyzer.aniso_svd_cut_q_range(0.8, 7, singular_cut_num=3)
anisoSVDanalyzer.iso_svd_cut_q_range(0.8, 7, singular_cut_num=3)

# anisoSVDanalyzer.data_svd(anisoSVDanalyzer.all_delay_aniso, "aniso", singular_cut_num=3)
# anisoSVDanalyzer.data_svd(anisoSVDanalyzer.all_delay_iso, "iso", singular_cut_num=3)
# anisoSVDanalyzer.aniso_svd_cut_q_range(0.5, 3.5, singular_cut_num = 3)
# anisoSVDanalyzer.iso_svd_cut_q_range(0.5, 3.5, singular_cut_num = 3)