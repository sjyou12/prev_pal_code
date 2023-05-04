import numpy as np
from ScattAnalyzer import ScattAnalyzer
from datetime import datetime

def linear_combination_with_minuit(random_try_num=1000):
    now_custom_fit = ScattAnalyzer()
    timestr = datetime.now().strftime('%Y-%m-%d-%Hhr.%Mmin.%Ssec')
    now_custom_fit.now_time = timestr
    now_custom_fit.dat_file_time_stamp = dat_file_time_stamp
    now_custom_fit.runNum = run_num
    now_custom_fit.runList = run_list
    now_custom_fit.file_out_path = file_out_dir_path + file_common_name

    log_print_per_try = 1
    if random_try_num >= 10000:
        log_print_per_try = 1000
    elif random_try_num >= 1000:
        log_print_per_try = 100
    elif random_try_num > 20:
        log_print_per_try = 20
    check_fit_ongoing = False
    # now_custom_fit.data_from_given_arr(q_val=avg_q_val_list, anal_data=noise_simulate_data)
    if weighted_average:
        now_custom_fit.read_input_file_for_LC(dat_input_common_root, material_file_time_stamp, material_run_list, material_sole_run, iso_LC, material_is_droplet, weighted_average=True)
    else:
        now_custom_fit.read_input_file_for_LC(dat_input_common_root, material_file_time_stamp, material_run_list, material_sole_run, iso_LC, material_is_droplet)

    # now_custom_fit.set_fit_param(param_name="c1", left_limit=0, right_limit=None, max_oom=1, min_oom=-3, is_fixed=False)
    # now_custom_fit.set_fit_param(param_name="c2", left_limit=1, right_limit=1, max_oom=1, min_oom=-1, is_fixed=False)

    now_custom_fit.run_minuit_with_LC(plot_each_delay=True)
    best_fit_result = now_custom_fit.plot_for_save
    best_params = now_custom_fit.best_params_save
    best_chi_square = now_custom_fit.best_chi_square_save
    # now_custom_fit.save_fit_result_as_dat(best_params, best_fit_result, now_custom_fit.input_raw_data)

num_of_random_initialization = 1000
run_num = 10
material_run_list = [45, 46]
material_sole_run = 9
run_list = [48, 49]
dat_file_time_stamp = "2022-05-01-20hr.36min.29sec"
material_file_time_stamp = ["2022-11-09-10hr.50min.14sec", "2022-05-01-20hr.36min.09sec"] # order : run 4&5, run9
previous_time_stamp = "2022-1-19-10hr.19min.47sec"

material_is_droplet = False
weighted_average = False
iso_LC = False

if weighted_average:
    if len(run_list) == 2:
        file_common_name = '/run{0:04d}_{1:04d}_avg_'.format(run_list[0], run_list[1])
        dat_infile_name_averaged = "run{0:04d}_{1:04d}_avg-aniso-cut_RSV.dat".format(run_list[0], run_list[1])
        file_out_dir_path = './anisotropy_fit_result/run{0:04d}_{1:04d}_avg/anisotropy'.format(run_list[0], run_list[1])
        dat_input_common_root = "../results/anisotropy/Anal_result_dat_file/run{0:04d}_{1:04d}_avg/{2}/".format(run_list[0], run_list[1], dat_file_time_stamp)

    elif len(run_list) == 3:
        file_common_name = '/run{0:04d}_{1:04d}_{2:04d}_avg_'.format(run_list[0], run_list[1], run_list[2])
        dat_infile_name_averaged = "run{0:04d}_{1:04d}_{2:04d}_avg-aniso-cut_RSV.dat".format(run_list[0], run_list[1], run_list[2])
        file_out_dir_path = './anisotropy_fit_result/run{0:04d}_{1:04d}_{2:04d}_avg/anisotropy'.format(run_list[0], run_list[1], run_list[2])
        dat_input_common_root = "../results/anisotropy/Anal_result_dat_file/run{0:04d}_{1:04d}_{2:04d}_avg/{3}/".format(run_list[0], run_list[1], run_list[2], dat_file_time_stamp)

else:
    file_common_name = '/run{0:04d}_'.format(run_num)
    file_out_dir_path = './anisotropy_fit_result/run{0:04d}/anisotropy'.format(run_num)
    dat_input_common_root = "../results/anisotropy/anal_result/run{0:04d}/run{0}_{1}/".format(run_num, dat_file_time_stamp)

if weighted_average:
    if len(run_list) == 2:
        print("Now run number is weighted average of {0} and {1}".format(run_list[0], run_list[1]))

    if len(run_list) == 3:
        print("Now run number is weighted average of {0}, {1} and {2}".format(run_list[0], run_list[1], run_list[2]))

else:
    print("Now run number is " + str(run_num))
linear_combination_with_minuit(random_try_num=num_of_random_initialization)
