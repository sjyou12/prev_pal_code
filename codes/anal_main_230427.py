from palxfel_scatter.diff_pair_1dcurve.MultiRunProc import MultiRunProc
from palxfel_scatter.anisotropy.AnisoAnal import AnisoAnal
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
import os
import multiprocessing
import h5py as h5

# TODO: change the run number
now_run_num = 231
bkg_run_num = now_run_num + 1
# bkg_run_num = 209
run_list_to_analyze_together = [50, 54]
whole_run_int_val_list = []
num_negative_delay = 5
DataDB_list = []

want_log = True
isotropic_only = False
anisotropic_only = False
#dat_file_out_whole_file = True
diff_img_2d_only = True
merge_multi_run = False
expand_negative_pool = False
rm_vapor = False
paused_run = False
shorten_img_num = False
on_on_off_test = False
timestamp = "2023-03-28-17hr.09min.38sec"

UpperBoundOfNotWater = 5000
LowerBoundOfWater = 70000
WaterOutlierLowerBound = 700000

def write_log_info(now_time, run_num, contents, delay_idx, merge_run_list, merge_multi_run = False):
    global want_log
    if merge_multi_run:
        out_file_name = "/data/exp_data/myeong0609/PAL-XFEL_20230427/analysis/codes/run_{0}_{1}_log.txt".format(merge_run_list[0], merge_run_list[1])
    else:
        out_file_name = "/data/exp_data/myeong0609/PAL-XFEL_20230427/analysis/codes/run_" + str(run_num) + "_log.txt"
    if want_log:
        try:
            if delay_idx == 0:
                log = open(out_file_name, 'r')
                if merge_multi_run:
                    os.remove("/data/exp_data/myeong0609/PAL-XFEL_20230427/analysis/codes/run_{0}_{1}_log.txt".format(merge_run_list[0], merge_run_list[1]))
                else:
                    os.remove("/data/exp_data/myeong0609/PAL-XFEL_20230427/analysis/codes/run_" + str(run_num) + "_log.txt")
        except:
            pass
        log = open(out_file_name, 'a')
        #log = open(out_file_name, 'w')
        time = datetime.now()
        if delay_idx == 0:
            if merge_multi_run:
                run_num_info = "Status of anisotropic analysis for run {0}&{1}".format(merge_run_list[0], merge_run_list[1])+ "\n"
                timeStamp = "Time stamp of this run is " + now_time

            else:
                run_num_info = "Status of anisotropic analysis for run " + str(run_num) + "\n"
                timeStamp = "Time stamp of this run is " + now_time + "\n"

            log.write(run_num_info)
            log.write(timeStamp)
            contents = str(time) + " " + contents + "\n"
            log.write(contents)
        else:
            contents = str(time) + " " + contents + "\n"
            log.write(contents)
    else:
        print(contents, "\n")

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

def single_run_DB_make(run_number):
    singleDB = [(run_number, UpperBoundOfNotWater, LowerBoundOfWater, WaterOutlierLowerBound)]
    return singleDB

# common constant for calculating q values
XrayEnergy = 20  # keV unit
NormFactor = 100000  # Normalization factor (sum of all integration)
# FileCommonRoot = "/data/exp_data/myeong0609/PAL-XFEL_20210514/pyFAI_calib_1D/"
FileCommonRoot = "/xfel/ffs/dat/scan/"
# FileCommonRoot = "/xfel/ffs/dat/ue_230427_FXL/"


def make_data():
    global XrayEnergy, FileCommonRoot, now_run_num
    make_pair_info = True
    global whole_run_int_val_list
    global merge_multi_run
    global expand_negative_pool
    global num_negative_delay
    if merge_multi_run:
        for rpt_num in range(len(run_list_to_analyze_together)):
            now_run_num = run_list_to_analyze_together[rpt_num]
            DataDB_list.append(single_run_DB_make(now_run_num))
        #for num_run_to_merge in range(len(run_list_to_analyze_together)):
            dataPreProcessor = MultiRunProc(DataDB_list[rpt_num])
            dataPreProcessor.common_variables(x_ray_energy=XrayEnergy, file_common_root=FileCommonRoot)
            dataPreProcessor.set_file_name_and_read_tth()
            dataPreProcessor.find_peak_pattern_multirun(incr_dist_plot=True, plot_outlier=False, sum_file_out=True)
            dataPreProcessor.read_intensity_only_water(np_file_out=False, plot_watersum_pass=False)  # if need q_val -> make np_file_out True
            if rpt_num == (len(run_list_to_analyze_together) - 1):
                whole_run_int_val_list.append(MultiRunProc.each_run_int_val_list)
            else:
                continue
        dataPreProcessor.small_ice_test(num_negative_delay, run_list_to_analyze_together, expand_negative_pool, plot_outlier=False, sum_file_out=False)
        dataPreProcessor.pairwise_diff_calc(num_negative_delay, run_list_to_analyze_together, merge_multi_run, expand_negative_pool, fileout_pair_info=make_pair_info)
        dataPreProcessor.additional_process_diff(run_list_to_analyze_together, merge_multi_run, fileout_pair_info=make_pair_info, plot_difference=False, plot_azimuthal=True)
    elif rm_vapor:
        nowDataDB = single_run_DB_make(now_run_num)
        dataPreProcessor = MultiRunProc(nowDataDB)
        dataPreProcessor.common_variables(x_ray_energy=XrayEnergy, file_common_root=FileCommonRoot)
        dataPreProcessor.set_file_name_and_read_tth()
        # get graph for decide criteria
        dataPreProcessor.vapor_anal(incr_dist_plot=True, plot_outlier=False, sum_file_out=False, plot_vapor_avg=True)
        # save intensity files
        dataPreProcessor.read_intensity_only_water(np_file_out=False, plot_watersum_pass=False, rm_vapor=True)  # if need q_val -> make np_file_out True
        dataPreProcessor.pairwise_diff_calc(fileout_pair_info=make_pair_info)
        dataPreProcessor.additional_process_diff(fileout_pair_info=make_pair_info, plot_difference=False, plot_azimuthal=True)
    else:
        nowDataDB = single_run_DB_make(now_run_num)
        dataPreProcessor = MultiRunProc(nowDataDB)
        if shorten_img_num:
            dataPreProcessor.shorten_img = shorten_img_num
            dataPreProcessor.num_shorten_img = 50
        if on_on_off_test:
            dataPreProcessor.on_on_off_test = on_on_off_test
        dataPreProcessor.common_variables(x_ray_energy=XrayEnergy, file_common_root=FileCommonRoot)
        dataPreProcessor.set_file_name_and_read_tth()
        # get graph for decide criteria
        #dataPreProcessor.plot_water_sum_dist(each_run_plot=False, sum_file_out=True)
        dataPreProcessor.find_peak_pattern(incr_dist_plot=True, plot_outlier=False, sum_file_out=True)
        # save intensity files
        dataPreProcessor.read_intensity_only_water(np_file_out=False, plot_watersum_pass=False, I0_normalize=False)  # if need q_val -> make np_file_out True
        dataPreProcessor.pairwise_diff_calc(num_negative_delay, run_list_to_analyze_together, merge_multi_run, expand_negative_pool, fileout_pair_info=make_pair_info)
        dataPreProcessor.additional_process_diff(run_list_to_analyze_together, merge_multi_run, fileout_pair_info=make_pair_info, plot_difference=False, plot_azimuthal=False)

make_data()
print("make data end")

def anal_anistorpy(merge_multi_run = False, expand_negative_pool = False, run_num = 10, merge_run_list = []):
    global want_log
    global timestamp
    global paused_run
    water_sum = []
    if merge_multi_run:
        for idx_run in merge_run_list:
            temp_water_sum = []
            load_water_sum_file_root = "../results/anisotropy/run" + str(idx_run) + "_watersum.npy"
            temp_water_sum = np.load(load_water_sum_file_root)
            if idx_run == merge_run_list[0]:
                water_sum = temp_water_sum
            else:
                water_sum = np.vstack((water_sum, temp_water_sum))
        load_pair_info_file_root = "../results/anisotropy/run{0}_{1}_pairinfo.npy".format(merge_run_list[0], merge_run_list[1])
        pair_info = np.load(load_pair_info_file_root, allow_pickle=True)
    else:
        load_water_sum_file_root = "../results/anisotropy/run" + str(run_num) + "_watersum.npy"
        water_sum = np.load(load_water_sum_file_root)
        load_pair_info_file_root = "../results/anisotropy/run" + str(run_num) + "_pairinfo.npy"
        pair_info = np.load(load_pair_info_file_root, allow_pickle=True)


    print(water_sum.shape, pair_info.shape)
    def anal_anisotropy_each_delay(now_time, puased_run, water_sum_data, run_num=53, pair_info_data=None, test_delay=7, merge_multi_run = False):
        #global want_log
        global run_list_to_analyze_together
        merge_run_list = run_list_to_analyze_together
        if test_delay == 0:
            if merge_multi_run:
                if not os.path.isdir('../results/anisotropy/anal_result/run{0}_{1}/run{0}_{1}_{2}'.format(merge_run_list[0], merge_run_list[1],now_time)):
                    os.makekdirs('../results/anisotropy/anal_result/run{0}_{1}/run{0}_{1}_{2}'.format(merge_run_list[0], merge_run_list[1],now_time))
            else:
                # if not os.path.isdir('../results/anisotropy/anal_result/run_{0:05d}/run{0}_{1}'.format(run_num,now_time)):
                #     os.makedirs('../results/anisotropy/anal_result/run_{0:05d}/run{0}_{1}'.format(run_num,now_time))
                if not os.path.isdir(
                        '../results/anisotropy/anal_result/run_{0:05d}/'.format(run_num)):
                    os.makedirs(
                        '../results/anisotropy/anal_result/run_{0:05d}/'.format(run_num))
        anisoAnalyzer = AnisoAnal()
        bkg_mask_common_path = "/xfel/ffs/dat/ue_230427_FXL/scratch/"
        now_common_path = "/xfel/ffs/dat/scan/"
        # now_common_path = "/xfel/ffs/dat/ue_230427_FXL/"

        now_bkg_file_path = now_common_path + "run_{0:05d}_DIR/eh1rayMX_img/".format(bkg_run_num)#00000001_00000300.h5".format(bkg_run_num)
        # now_bkg_file_path = now_common_path + "230427_bkg_1_00005_DIR/eh1rayMX_img/"#00000001_00000300.h5".format(bkg_run_num)
        # now_bkg_file_path = "/xfel/ffs/dat/scan/" + "230427_bkg_1_00005_DIR/eh1rayMX_img/"#00000001_00000300.h5".format(bkg_run_num)
        for file_name in os.listdir(now_bkg_file_path):
            temp_bkg = []
            if file_name.endswith(".h5"):
                now_bkg_file = now_bkg_file_path + file_name
                now_h5_file = h5.File(now_bkg_file, 'r')
                now_file_keys = now_h5_file.keys()
                for now_key in now_file_keys:
                    now_file_int = list(now_h5_file[now_key])
                    temp_bkg.append(now_file_int)
                now_h5_file.close()
        now_bkg_file = np.average(temp_bkg, axis=0)

        # mask_file_common_path = "/data/exp_data/PAL-XFEL_20210514/scratch/"
        mask_file_common_path = bkg_mask_common_path
        now_mask_file = mask_file_common_path + "230428_mask.h5"

        # now_mask_file = "/data/exp_data/PAL-XFEL_20210514/scratch/210517_mask_dis1.h5"
        # now_bkg_file = "/data/exp_data/PAL-XFEL_20210514/rawdata/210517_bg_00002_DIR/eh1rayMX_img/00000001_00000300.h5"

        poni_file_path = "/xfel/ffs/dat/ue_230427_FXL/calibration/230427_bkg_3.poni"

        # anisoAnalyzer.set_print_log_file(run_num, want_log)
        anisoAnalyzer.set_common_env(common_path=now_common_path)
        anisoAnalyzer.set_mask(now_mask_file, show_mask=False)
        anisoAnalyzer.set_background(now_bkg_file)
        anisoAnalyzer.set_img_info(poni_file_path)
        anisoAnalyzer.UpperBoundOfNotWater = UpperBoundOfNotWater
        anisoAnalyzer.LowerBoundOfWater = LowerBoundOfWater
        anisoAnalyzer.WaterOutlierLowerBound = WaterOutlierLowerBound
        anisoAnalyzer.read_img_file_names(run_list_to_analyze_together, merge_multi_run, run_num=run_num)
        anisoAnalyzer.aniso_anal_each_delay(now_time, paused_run, run_list_to_analyze_together, merge_multi_run, expand_negative_pool, rm_vapor, norm_data=water_sum_data, pair_info=pair_info_data, idx_delay=test_delay, run_num=run_num)

    # test_delays = range(22, 87)
    test_delays = range(0, 87)
    # test_delays = range(0, 51)
    now_time = datetime.now().strftime('%Y-%m-%d-%Hhr.%Mmin.%Ssec')
    # now_time = "2023-04-27-21hr.22min.55sec"
    iso_whole_dat_list = []
    aniso_whole_dat_list = []
    iso_whole_stderr_list = []
    aniso_whole_stderr_list = []

    if want_log:
        print("This file's time stamp is " + now_time)
        chunk_len = 10
        chunked_delay_arr = []
        num_chunk = int(np.ceil(len(test_delays) / chunk_len))

        for chunk_idx in range(num_chunk):
            try:
                chunked_delay_arr.append(test_delays[chunk_idx * chunk_len:(chunk_idx + 1) * chunk_len])
            except:
                chunked_delay_arr.append(test_delays[chunk_idx * chunk_len:])
        for chunk_idx, chunked_arr in enumerate(chunked_delay_arr):
            processes = []
            for each_delay in chunked_arr:
                p = multiprocessing.Process(target=anal_anisotropy_each_delay, args=(now_time, paused_run, water_sum, run_num, pair_info, each_delay, merge_multi_run))
                p.start()
                processes.append(p)
            for process in processes:
                process.join()
        # for each_delay in test_delays:
        #     anal_anisotropy_each_delay(now_time, paused_run, water_sum, run_num, pair_info, each_delay, merge_multi_run)
    else:
        for each_delay in test_delays:
            if merge_multi_run:
                if want_log:
                    anal_anisotropy_each_delay(now_time, paused_run, water_sum_data=water_sum, run_num=run_num, pair_info_data=pair_info, test_delay=each_delay, merge_multi_run = True)
                    contents = "Finished anisotropy analysis of " + str(each_delay+1) + "-th delay"
                    # write_log_info(now_time, now_run_num, contents, each_delay, merge_run_list ,merge_multi_run=True)

                else:
                    file_open_path = "../results/anisotropy/anal_result/run{0}_{1}_{2}".format(merge_run_list[0],merge_run_list[1], timestamp)
                    if isotropic_only:
                        isotropy_file = np.load("../results/anisotropy/anal_result/run{0}_{1}_delay{2}_iso.npy".format(merge_run_list[0],merge_run_list[1],each_delay + 1))
                        q_val_file = np.load("../results/anisotropy/anal_result/run{0}_{1}_delay{2}_qval.npy".format(merge_run_list[0],merge_run_list[1],each_delay + 1))
                        plt.title("isotropic signal of " + str(each_delay + 1) + "-th delay")
                        plt.xlim(0.5, 4.5)
                        plt.ylim(-7.5, 7.5)
                        plt.scatter(q_val_file, isotropy_file)
                        plt.xlabel("Q (A^-1)")
                        plt.ylabel("dS_2")
                        plt.show()

                    else:
                        anisotropy_file = np.load("../results/anisotropy/anal_result/run{0}_{1}_delay{2}_aniso.npy".format(merge_run_list[0],merge_run_list[1],each_delay + 1))
                        q_val_file = np.load("../results/anisotropy/anal_result/run{0}_{1}_delay{2}_qval.npy".format(merge_run_list[0],merge_run_list[1], each_delay + 1))
                        plt.title("anisotropic signal of " + str(each_delay + 1) + "-th delay")
                        plt.xlim(0.5, 3.5)
                        plt.ylim(-3.0, 3.0)
                        plt.scatter(q_val_file, anisotropy_file)
                        plt.xlabel("Q (A^-1)")
                        plt.ylabel("dS_2")
                        plt.show()
            else:
                # if want_log:
                #     anal_anisotropy_each_delay(now_time, paused_run, water_sum_data=water_sum, run_num=run_num, pair_info_data=pair_info, test_delay=each_delay, merge_multi_run=False)
                #     contents = "Finished anisotropy analysis of " + str(each_delay + 1) + "-th delay"
                #     write_log_info(now_time, now_run_num, contents, each_delay, merge_run_list, merge_multi_run=False)

                # else:
                    file_open_path = "../results/anisotropy/anal_result/run{0:05d}/run{0}_{1}".format(run_num, timestamp)
                    file_out_path = "../results/anisotropy/Anal_result_dat_file/run{0:05d}/run{0}_{1}".format(run_num, timestamp)
                    if os.path.isdir(file_out_path):
                        pass
                    else:
                        os.makedirs(("../results/anisotropy/Anal_result_dat_file/run{0:05d}/run{0}_{1}".format(run_num, timestamp)))
                    # if each_delay == 0:
                    #     if os.path.isdir(file_out_path):
                    #         shutil.rmtree("../results/anisotropy/Anal_result_dat_file/run{0:04d}/run{0}_{1}".format(run_num, timestamp))
                    #     os.makedirs('../results/anisotropy/Anal_result_dat_file/run{0:04d}/run{0}_{1}'.format(run_num,timestamp))
                    #     dat_file_common_out_path = '../results/anisotropy/Anal_result_dat_file/run{0:04d}/run{0}_{1}'.format(run_num,timestamp)

                    if isotropic_only:
                        iso_dat_file_out_path = '../results/anisotropy/Anal_result_dat_file/run{0:05d}/run{0}_{1}/'.format(run_num,timestamp)
                        isotropy_file = np.load(file_open_path + "/run{0}_delay{1}_iso.npy".format(now_run_num, each_delay + 1))
                        #isotropy_file = np.load("../results/anisotropy/anal_result/run{0}_delay{1}_iso.npy".format(now_run_num, each_delay + 1))
                        q_val_file = np.load(file_open_path + "/run{0}_delay{1}_qval.npy".format(now_run_num, each_delay + 1))
                        iso_stderr = np.load(file_open_path + "/run{0}_delay{1}_stderr_iso.npy".format(now_run_num, each_delay + 1))
                        #np.save(iso_dat_file_out_path + "run{0}_delay{1}_iso.npy".format(now_run_num, each_delay + 1), isotropy_file)
                        average_line = np.average(isotropy_file)
                        iso_whole_dat_list.append(isotropy_file)
                        iso_whole_stderr_list.append(iso_stderr)
                        #q_val_file = np.load("../results/anisotropy/anal_result/run{0}_delay{1}_qval.npy".format(now_run_num, each_delay + 1))
                        # if each_delay == 0:
                        #     np.savetxt(dat_file_common_out_path + '{0}_delay{1}_q_val.dat'.format(now_run_num,each_delay + 1),q_val_file, delimiter='\n', fmt='%1.9f')
                        # np.savetxt(iso_dat_file_out_path + '/run{0}_delay{1}_iso.dat'.format(now_run_num, each_delay + 1), isotropy_file, delimiter= '\n', fmt='%1.9f')
                        plt.title("isotropic signal of " + str(each_delay + 1) + "-th delay")
                        plt.xlim(0.5, 4.5)
                        plt.ylim(-1, 1)
                        plt.axhline(y=average_line)
                        plt.scatter(q_val_file, isotropy_file)
                        plt.xlabel("Q (A^-1)")
                        plt.ylabel("dS_2")
                        plt.show()

                    elif anisotropic_only:
                        anisotropy_dat_out_path = '../results/anisotropy/Anal_result_dat_file/run{0:05d}/run{0}_{1}/'.format(run_num,timestamp)
                        anisotropy_file = np.load(file_open_path + "/run{0}_delay{1}_aniso.npy".format(now_run_num, each_delay + 1))
                        q_val_file = np.load(file_open_path + "/run{0}_delay{1}_qval.npy".format(now_run_num, each_delay + 1))
                        aniso_stderr = np.load(file_open_path + "/run{0}_delay{1}_stderr_aniso.npy".format(now_run_num, each_delay + 1))
                        #np.save(anisotropy_dat_out_path + "run{0}_delay{1}_aniso.npy".format(now_run_num, each_delay + 1), anisotropy_file)
                        aniso_whole_dat_list.append(anisotropy_file)
                        aniso_whole_stderr_list.append(aniso_stderr)
                        #np.savetxt(anisotropy_dat_out_path + '/run{0}_delay{1}_aniso.dat'.format(now_run_num, each_delay + 1), anisotropy_file, delimiter= '\n', fmt='%1.9f')
                        plt.title("anisotropic signal of " + str(each_delay + 1) + "-th delay")
                        plt.xlim(0.5, 3.5)
                        plt.ylim(-2, 2)
                        plt.scatter(q_val_file, anisotropy_file)
                        plt.xlabel("Q (A^-1)")
                        plt.ylabel("dS_2")
                        plt.show()

                    elif diff_img_2d_only:
                        img_load_root = "../results/anisotropy/anal_result/run{0}_delay{1}_2d_diff_img.npy".format(run_num, each_delay + 1)
                        masked_val = np.load(img_load_root)
                        plt.title("2d image of " + (str(each_delay +1)) + "-th delay")
                        plt.pcolor(masked_val)
                        plt.colorbar()
                        plt.clim(-2, 2)
                        plt.xlabel("x pixel")
                        plt.ylabel("y pixel")
                        # plt.axhline(y=724.525)  # [730.112, 727.711]    right : [713.351, 724.525]
                        # plt.axvline(x=713.351)
                        plt.show()
    if isotropic_only:
        file_out_path = iso_dat_file_out_path + '/run{0}_iso.dat'.format(now_run_num)
        dat_file_out(q_val_file, iso_whole_dat_list, file_out_path)
        stderr_out_path = iso_dat_file_out_path + '/run{0}_stderr_iso.dat'.format(now_run_num)
        dat_file_out(q_val_file, iso_whole_stderr_list, stderr_out_path)
    elif anisotropic_only:
        file_out_path = anisotropy_dat_out_path + '/run{0}_aniso.dat'.format(now_run_num)
        dat_file_out(q_val_file, aniso_whole_dat_list, file_out_path)
        stderr_out_path = anisotropy_dat_out_path + '/run{0}_stderr_aniso.dat'.format(now_run_num)
        dat_file_out(q_val_file, aniso_whole_stderr_list, stderr_out_path)
anal_anistorpy(merge_multi_run, expand_negative_pool, run_num=now_run_num, merge_run_list=run_list_to_analyze_together)
