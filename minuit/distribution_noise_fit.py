import numpy as np
from ScattAnalyzer import ScattAnalyzer
# from fit_fxn_n_const import saxs_intensity_near10_avg_no_baseline
# from fit_fxn_n_const import test_radius_list, avg_q_val_list, test_radius_std_list
# import argparse
from datetime import datetime
from matplotlib import pyplot as plt
# parser = argparse.ArgumentParser(description='give miunit fit setting value')
# parser.add_argument('-t', '--try', nargs=1, dest='try_num', type=int, default=100,
#                     help='enter random intialize try number to do')
# parser.add_argument('-r', '--radiusidx', nargs=1, dest='radi_idx', type=int, default=1,
#                     help='enter radius idx (1-based) to fit now')
# args = parser.parse_args()

def saxs_analyze_with_minuit(num_rsv_to_fit, random_try_num=1000):
    now_custom_fit = ScattAnalyzer()
    # now_custom_fit.set_scattering_function(saxs_intensity_near10_avg_no_baseline)
    timestr = datetime.now().strftime('%Y-%m-%d-%Hhr.%Mmin.%Ssec')
    now_custom_fit.now_time = timestr
    now_custom_fit.log_out_file_common_name = log_out_file_common_name
    now_custom_fit.runNum = run_num
    now_custom_fit.runList = run_list
    now_custom_fit.file_out_path = file_out_dir_path
    now_custom_fit.file_common_name = file_common_name
    now_custom_fit.previous_dat_load_path = previous_result_load_path

    log_print_per_try = 1
    if random_try_num >= 10000:
        log_print_per_try = 1000
    elif random_try_num >= 1000:
        log_print_per_try = 100
    elif random_try_num > 20:
        log_print_per_try = 20
    check_fit_ongoing = False
    # now_custom_fit.data_from_given_arr(q_val=avg_q_val_list, anal_data=noise_simulate_data)

    if anisotropy:
        now_custom_fit.anisotropy_process = True
        if aniso_single_exponential:
            if num_rsv_to_fit == 2:
                now_custom_fit.set_fit_param(param_name="a1", left_limit=-15, right_limit=-0.1, max_oom=1, min_oom=-3, is_fixed=False)
                now_custom_fit.set_fit_param(param_name="t1", left_limit=0.01, right_limit=1, max_oom=1, min_oom=-1, is_fixed=False)
                now_custom_fit.set_fit_param(param_name="x0", left_limit=-1, right_limit=1, is_fixed=False, fixed_value=0, no_random_initial=True)
                now_custom_fit.set_fit_param(param_name="y0", left_limit=None, right_limit=None, max_oom=1, min_oom=-10, is_fixed=True, fixed_value=0.0)
                now_custom_fit.set_fit_param(param_name="FWHM", left_limit=0, right_limit=10, is_fixed=True, fixed_value=2.210915532782842)
            elif num_rsv_to_fit == 1:
                now_custom_fit.set_fit_param(param_name="a1", left_limit=0.1, right_limit=15, is_fixed=False,fixed_value=10, no_random_initial=False)
                now_custom_fit.set_fit_param(param_name="t1", left_limit=0.5, right_limit=100, is_fixed=False,fixed_value=10, no_random_initial=True)
                now_custom_fit.set_fit_param(param_name="x0", left_limit=-0.1, right_limit=0.1, is_fixed=False,fixed_value=0, no_random_initial=True)
                now_custom_fit.set_fit_param(param_name="y0", left_limit=-0.5, right_limit=0.5, is_fixed=True,fixed_value=0.0)
                now_custom_fit.set_fit_param(param_name="FWHM", left_limit=0, right_limit=10, is_fixed=True,fixed_value=2.210915532782842)
            now_custom_fit.aniso_single_exp = True

        elif aniso_double_exponential:
            if num_rsv_to_fit == 2:
                now_custom_fit.set_fit_param(param_name="a1", left_limit=-15, right_limit=-0.1, is_fixed=False)
                now_custom_fit.set_fit_param(param_name="t1", left_limit=0.01, right_limit=10, is_fixed=False)
                now_custom_fit.set_fit_param(param_name="x0", left_limit=-1, right_limit=1, is_fixed=True, fixed_value=0.0399381081598766)
                now_custom_fit.set_fit_param(param_name="y0", left_limit=-1, right_limit=5, is_fixed=False)
                # now_custom_fit.set_fit_param(param_name="a2", left_limit=0, right_limit=None, max_oom=1, min_oom=-2, is_fixed=False)
                now_custom_fit.set_fit_param(param_name="ratio_a", left_limit=0.001, right_limit=2, is_fixed=False) # ratio_a = a2/a1
                # now_custom_fit.set_fit_param(param_name="t2", left_limit=1, right_limit=100, max_oom=3, min_oom=-3, is_fixed=False)
                now_custom_fit.set_fit_param(param_name="delta_t", left_limit=0, right_limit=40, is_fixed=False) # delta t = t2-t1
                now_custom_fit.set_fit_param(param_name="FWHM", left_limit=0.75, right_limit=2.5, is_fixed=True, fixed_value=2.210915532782842)
            else:
                now_custom_fit.set_fit_param(param_name="a1", left_limit=0, right_limit=3, is_fixed=False, fixed_value=1.2, no_random_initial=True)
                now_custom_fit.set_fit_param(param_name="t1", left_limit=0.1, right_limit=0.3, is_fixed=True, fixed_value=0.16, no_random_initial=True)
                # now_custom_fit.set_fit_param(param_name="t1", left_limit=0.1, right_limit=0.5, is_fixed=True, fixed_value=0.16)
                now_custom_fit.set_fit_param(param_name="x0", left_limit=-2, right_limit=1, is_fixed=False, fixed_value=0, no_random_initial=True)
                now_custom_fit.set_fit_param(param_name="y0", left_limit=-1, right_limit=1, is_fixed=False, fixed_value=0.0)
                # now_custom_fit.set_fit_param(param_name="a2", left_limit=0, right_limit=None, max_oom=1, min_oom=-2, is_fixed=False)
                now_custom_fit.set_fit_param(param_name="ratio_a", left_limit=0.03, right_limit=0.3, is_fixed=False) # ratio_a = a2/a1
                # now_custom_fit.set_fit_param(param_name="t2", left_limit=1, right_limit=100, max_oom=3, min_oom=-3, is_fixed=False)
                now_custom_fit.set_fit_param(param_name="delta_t", left_limit=0.5, right_limit=100, fixed_value=0.8, is_fixed=True) # delta t = t2-t1
                now_custom_fit.set_fit_param(param_name="FWHM", left_limit=1.0, right_limit=2.5, is_fixed=False, fixed_value=2.210915532782842)
                # now_custom_fit.set_fit_param(param_name="FWHM", left_limit=1.0, right_limit=2.5, is_fixed=False, fixed_value=1.8214210226034)
            now_custom_fit.aniso_double_exp = True

        elif aniso_stretched_exponential:
            now_custom_fit.set_fit_param(param_name="a", left_limit=-15, right_limit=0, max_oom=1, min_oom=-3, is_fixed=False)
            now_custom_fit.set_fit_param(param_name="t", left_limit=0.1, right_limit=None, max_oom=2, min_oom=-1, is_fixed=False)
            now_custom_fit.set_fit_param(param_name="beta", left_limit=0.5, right_limit=0.7, is_fixed=False)
            now_custom_fit.set_fit_param(param_name="x0", left_limit=-2, right_limit=1, max_oom=1, min_oom=-10, is_fixed=True, fixed_value=0.0000514130048231349)
            now_custom_fit.set_fit_param(param_name="y0", left_limit=None, right_limit=None, max_oom=1, min_oom=-10, is_fixed=False)
            now_custom_fit.set_fit_param(param_name="FWHM", left_limit=0, right_limit=10, is_fixed=True, fixed_value=2.210915532782842)
            now_custom_fit.aniso_stretched_exp = True

        elif aniso_p_gallo_function:
            now_custom_fit.set_fit_param(param_name="a", left_limit=0, right_limit=1, is_fixed=False, fixed_value=0.3, no_random_initial=True)
            now_custom_fit.set_fit_param(param_name="t1", left_limit=0.1, right_limit=0.3, is_fixed=False, fixed_value=0.1, no_random_initial=True)
            now_custom_fit.set_fit_param(param_name="x0", left_limit=-2, right_limit=1, is_fixed=False, fixed_value=0, no_random_initial=True)
            now_custom_fit.set_fit_param(param_name="y0", left_limit=-1, right_limit=1, is_fixed=True, fixed_value=0.0)
            # now_custom_fit.set_fit_param(param_name="ratio_a", left_limit=0.03, right_limit=0.3, is_fixed=False) # ratio_a = a2/a1
            now_custom_fit.set_fit_param(param_name="t2", left_limit=0.7, right_limit=100, max_oom=3, min_oom=-3, is_fixed=False)
            # now_custom_fit.set_fit_param(param_name="delta_t", left_limit=0.5, right_limit=40, is_fixed=False) # delta t = t2-t1
            now_custom_fit.set_fit_param(param_name="beta", left_limit=0.5, right_limit=1.0, is_fixed=True, fixed_value=0.74328) # beta 0.80907 for RT, 0.81136 for 270 K, 0.7853 for 250 K, 0.74328 for 235 K
            now_custom_fit.set_fit_param(param_name="FWHM", left_limit=1.6, right_limit=2.5, is_fixed=True, fixed_value=2.210915532782842)
            now_custom_fit.aniso_p_gallo = True

        elif aniso_cos_square_fit:
            now_custom_fit.set_fit_param(param_name="a1", left_limit=0.001, right_limit=0.01, is_fixed=False, fixed_value=0.007, no_random_initial=False)
            now_custom_fit.set_fit_param(param_name="t1", left_limit=0.03, right_limit=0.3, is_fixed=False, fixed_value=0.16, no_random_initial=True)
            # now_custom_fit.set_fit_param(param_name="t1", left_limit=0.1, right_limit=0.5, is_fixed=True, fixed_value=0.16)
            now_custom_fit.set_fit_param(param_name="x0", left_limit=-0.1, right_limit=0.1, is_fixed=True, fixed_value=0, no_random_initial=True)
            now_custom_fit.set_fit_param(param_name="y0", left_limit=-0.5, right_limit=0.5, is_fixed=True, fixed_value=0.0)
            # now_custom_fit.set_fit_param(param_name="a2", left_limit=0, right_limit=None, max_oom=1, min_oom=-2, is_fixed=False)
            now_custom_fit.set_fit_param(param_name="ratio_a", left_limit=0.2, right_limit=0.7, is_fixed=False, fixed_value=0.45) # ratio_a = a2/a1
            # now_custom_fit.set_fit_param(param_name="t2", left_limit=1, right_limit=100, max_oom=3, min_oom=-3, is_fixed=False)
            now_custom_fit.set_fit_param(param_name="delta_t", left_limit=0.5, right_limit=100, is_fixed=False, fixed_value=1.3) # delta t = t2-t1
            now_custom_fit.aniso_cos_square_fit = True

        elif aniso_cut_gaussian:
            now_custom_fit.set_fit_param(param_name="a1", left_limit=0.01, right_limit=3, is_fixed=False, fixed_value=0.1, no_random_initial=False)
            now_custom_fit.set_fit_param(param_name="t1", left_limit=0.5, right_limit=100, is_fixed=False, fixed_value=1, no_random_initial=True)
            now_custom_fit.set_fit_param(param_name="x0", left_limit=-0.1, right_limit=0.1, is_fixed=False, fixed_value=0, no_random_initial=True)
            now_custom_fit.set_fit_param(param_name="y0", left_limit=-0.5, right_limit=0.5, is_fixed=True, fixed_value=0.0)
            now_custom_fit.set_fit_param(param_name="FWHM", left_limit=0, right_limit=10, is_fixed=True, fixed_value=2.210915532782842)
            if (run_num in [9, 10, 11] and not weighted_average) or (weighted_average and (run_list in [[21, 23], [19, 20], [18, 24], [4, 5]])):
                now_custom_fit.mask_list = [0.1, 0.6]
            elif (run_num in [30, 34, 35] and not weighted_average) or (weighted_average and run_list == [26, 28]):
                now_custom_fit.mask_list = [0.2, 1]
            elif (weighted_average and (run_list in [[41, 43], [45, 46], [48, 49]])):
                now_custom_fit.mask_list = [0.05, 0.5]
            elif (run_num in [68, 69, 70, 71]):
                now_custom_fit.mask_list = [0.1, 0.6]
            now_custom_fit.temporary_mask = True

        elif aniso_single_exp_without_conv:
            now_custom_fit.set_fit_param(param_name="a1", left_limit=0.01, right_limit=3, is_fixed=False, fixed_value=0.1, no_random_initial=False)
            now_custom_fit.set_fit_param(param_name="t1", left_limit=0.5, right_limit=100, is_fixed=False, fixed_value=1, no_random_initial=True)
            now_custom_fit.set_fit_param(param_name="x0", left_limit=-2, right_limit=0.3, is_fixed=True, fixed_value=0.0207571190617934, no_random_initial=True)
            now_custom_fit.set_fit_param(param_name="y0", left_limit=-0.5, right_limit=0.5, is_fixed=True, fixed_value=0.0)
            if (run_num in [9, 10, 11] and not weighted_average) or (weighted_average and (run_list in [[21, 23], [19, 20], [18, 24]])):
                now_custom_fit.mask_list = [0.1, 0.6]
            elif weighted_average and (run_list == [4, 5]):
                now_custom_fit.mask_list = [-1.5, -1]
            elif (run_num in [30, 34, 35] and not weighted_average) or (weighted_average and run_list == [26, 28]):
                now_custom_fit.mask_list = [0.2, 1]
            elif (weighted_average and (run_list in [[41, 43], [45, 46], [48, 49]])):
                now_custom_fit.mask_list = [0.05, 0.5]
            elif (run_num in [68, 69, 70, 71]):
                now_custom_fit.mask_list = [0.1, 0.6]
            now_custom_fit.aniso_single_exp_without_conv = True

    elif not anisotropy:
        now_custom_fit.isotropy_process = True
        if iso_heating:
            now_custom_fit.iso_heating = True
            now_custom_fit.set_fit_param(param_name="a", left_limit=None, right_limit=0, max_oom=0, min_oom=-3, is_fixed=False)
            now_custom_fit.set_fit_param(param_name="t", left_limit=0, right_limit=100, is_fixed=False)
            now_custom_fit.set_fit_param(param_name="x0", left_limit=None, right_limit=None, max_oom=1, min_oom=-10, is_fixed=False)
        elif iso_second:
            now_custom_fit.iso_second = True
            now_custom_fit.set_fit_param(param_name="a1", left_limit=0, right_limit=3, is_fixed=False, fixed_value=1.2, no_random_initial=True)
            now_custom_fit.set_fit_param(param_name="t1", left_limit=0.1, right_limit=0.3, is_fixed=False, fixed_value=0.1, no_random_initial=True)
            # now_custom_fit.set_fit_param(param_name="t1", left_limit=0.1, right_limit=0.5, is_fixed=True, fixed_value=0.16)
            now_custom_fit.set_fit_param(param_name="x0", left_limit=-2, right_limit=1, is_fixed=True, fixed_value=-1.46222063943803)
            now_custom_fit.set_fit_param(param_name="y0", left_limit=-1, right_limit=1, is_fixed=False)
            # now_custom_fit.set_fit_param(param_name="a2", left_limit=0, right_limit=None, max_oom=1, min_oom=-2, is_fixed=False)
            now_custom_fit.set_fit_param(param_name="ratio_a", left_limit=0.03, right_limit=0.3, is_fixed=False)  # ratio_a = a2/a1
            # now_custom_fit.set_fit_param(param_name="t2", left_limit=1, right_limit=100, max_oom=3, min_oom=-3, is_fixed=False)
            now_custom_fit.set_fit_param(param_name="delta_t", left_limit=0.1, right_limit=10, is_fixed=False)  # delta t = t2-t1
            now_custom_fit.set_fit_param(param_name="FWHM", left_limit=1.6, right_limit=2.5, is_fixed=True, fixed_value=2.210915532782842)

    if weighted_average:
        now_custom_fit.set_anal_data_as_file(dat_input_common_root, dat_infile_name_averaged, num_rsv_to_fit, weighted_average=True)
    else:
        now_custom_fit.set_anal_data_as_file(dat_input_common_root, dat_infile_name, num_rsv_to_fit, sing_val)

    now_custom_fit.calculate_time_list = np.round(np.arange(start_time, start_time + time_step * num_of_calculation, 10E-3), 3)
    now_custom_fit.time_step = time_step
    if aniso_cut_gaussian:
        now_custom_fit.original_calc_time_list = now_custom_fit.calculate_time_list
        mask_start_idx = np.where(now_custom_fit.original_calc_time_list == np.float64(now_custom_fit.mask_list[0]))[0]
        mask_end_idx = np.where(now_custom_fit.original_calc_time_list == np.float64(now_custom_fit.mask_list[1]))[0]
        temp_time_arr_1 = now_custom_fit.original_calc_time_list[0:int(mask_start_idx)]
        temp_time_arr_2 = now_custom_fit.original_calc_time_list[int(mask_end_idx) + 1:]
        temp_time = np.hstack((temp_time_arr_1, temp_time_arr_2))
        now_custom_fit.calculate_time_list = temp_time

    elif aniso_single_exp_without_conv:
        now_custom_fit.original_calc_time_list = now_custom_fit.calculate_time_list
        fit_start_idx = np.where(now_custom_fit.original_calc_time_list == now_custom_fit.mask_list[1])[0][0]
        now_custom_fit.calculate_time_list = now_custom_fit.original_calc_time_list[fit_start_idx:]

    # now_custom_fit.finish_minuit = True
    # # convoluted_function = now_custom_fit.p_gallo_function_convolution(now_custom_fit.calculate_time_list, a=1.66213165878085, t1=1.00013954921794, x0=-0.731983295685843, y0=0, t2=8.71389673533409, beta=0.75, FWHM=2.21091553278284)
    # # gaussian_function = now_custom_fit.gaussian(now_custom_fit.calculate_time_list, FWHM=2.21091553278284)
    # exp_function = now_custom_fit.exp_decay(now_custom_fit.calculate_time_list, a1=0.007, t1=0.16, x0=0, y0=0, a2=0.0025, t2=1.5)
    # common_time_delay = now_custom_fit.input_time_delay
    # plt.plot(now_custom_fit.calculate_time_list, exp_function, label="exp_decay")
    # plt.plot(now_custom_fit.input_time_delay, now_custom_fit.input_raw_data, label="experiment")
    # # plt.plot(common_time_delay, now_custom_fit.input_raw_data, label="run 69")
    # # plt.plot(now_custom_fit.calculate_time_list, convoluted_function, label="test")
    # plt.legend()
    # plt.show()
    now_custom_fit.random_initial_fit(random_try_num, plot_min_chi_square=check_fit_ongoing, log_print_per_try=log_print_per_try, plot_skip=True)
    best_fit_result = now_custom_fit.plot_for_save
    best_params = now_custom_fit.best_params_save
    now_custom_fit.save_fit_result_as_dat(best_params, best_fit_result, now_custom_fit.input_raw_data)

# num_of_random_initialization = 5
# num_of_calculation = 1099 #TODO change if tim scale is approaching ns scale
# time_step = 10E-3
# start_time = -0.98
# temperature = '250K'

num_of_random_initialization = 5
num_of_calculation = 621 #TODO change if tim scale is approaching ns scale
time_step = 10E-3
start_time = -3

run_num = 1
run_list = [48, 49]

dir_time_stamp = "2023-02-16-16hr.00min.45sec" #only for MD results
dat_file_time_stamp = "2023-03-31-11hr.57min.57sec"
previous_time_stamp = "2022-03-18-16hr.0min.54sec"

num_of_RSV_to_fit = 1
need_negative = False
weighted_average = False
fit_with_LC = False
fit_with_droplet_LC = False # if want to fit with LC result with run 45&46
sing_val = False

anisotropy = True
aniso_single_exponential = False
aniso_double_exponential = True
aniso_stretched_exponential = False
aniso_p_gallo_function = False
aniso_cos_square_fit = False
aniso_cut_gaussian = False #앞에 자르고, 뒤에만 fitting하는 것. 두번째 exponential만 피팅
aniso_single_exp_without_conv = False
p_width_10 = False
p_width_20 = True
p_width_40 = False

iso_heating = False
iso_second = False


if anisotropy:
    if weighted_average:
        if len(run_list) == 2:
            dat_input_common_root = "../results/anisotropy/svd/run{0:04d}_{1:04d}_avg/run{0}_{1}_avg_{2}".format(run_list[0], run_list[1], dat_file_time_stamp)
            file_common_name = 'run{0:04d}_{1:04d}_avg_'.format(run_list[0], run_list[1])
            if fit_with_LC:
                dat_infile_name_averaged = "/run{0:04d}_{1:04d}_contributions_from_LSV_1_2.dat".format(run_list[0], run_list[1])
                if fit_with_droplet_LC:
                    dat_infile_name_averaged = "/run{0:04d}_{1:04d}_contributions_from_LSV_1_2_of_droplet.dat".format(run_list[0], run_list[1])
            else:
                dat_infile_name_averaged = "/run{0:04d}_{1:04d}_avg-aniso-cut_RSV.dat".format(run_list[0], run_list[1])
            if num_of_RSV_to_fit == 2:
                file_out_dir_path = './anisotropy_fit_result/run{0:04d}_{1:04d}_avg/unexpected/'.format(run_list[0], run_list[1])
            else:
                file_out_dir_path = './anisotropy_fit_result/run{0:04d}_{1:04d}_avg/anisotropy/'.format(run_list[0], run_list[1])
            previous_result_load_path = file_out_dir_path + file_common_name + "{}".format(previous_time_stamp)
            log_out_file_common_name = "/data/exp_data/myeong0609/PAL-XFEL_20210514/analysis/minuit/run_{0}_{1}_".format(run_list[0], run_list[1])

        elif len(run_list) == 3:
            dat_input_common_root = "../results/anisotropy/svd/run{0:04d}_{1:04d}_{2:04d}_avg/run{0}_{1}_{2}_avg_{3}/".format(run_list[0], run_list[1], run_list[2], dat_file_time_stamp)
            file_common_name = 'run{0:04d}_{1:04d}_{2:04d}_avg_'.format(run_list[0], run_list[1], run_list[2])
            if fit_with_LC:
                dat_infile_name_averaged = "run{0:04d}_{1:04d}_{2:04d}_contributions_from_LSV_1_2.dat".format(run_list[0], run_list[1], run_list[2])
                if fit_with_droplet_LC:
                    dat_infile_name_averaged = "run{0:04d}_{1:04d}_{2:04d}_contributions_from_LSV_1_2_of_droplet.dat".format(run_list[0], run_list[1], run_list[2])
            else:
                dat_infile_name_averaged = "run{0:04d}_{1:04d}_{2:04d}_avg-aniso-cut_RSV.dat".format(run_list[0], run_list[1], run_list[2])
            if num_of_RSV_to_fit == 2:
                file_out_dir_path = './anisotropy_fit_result/run{0:04d}_{1:04d}_{2:04d}_avg/unexpected/'.format(run_list[0], run_list[1], run_list[2])
            else:
                file_out_dir_path = './anisotropy_fit_result/run{0:04d}_{1:04d}_{2:04d}_avg/anisotropy/'.format(run_list[0], run_list[1], run_list[2])
            previous_result_load_path = file_out_dir_path + file_common_name + "{}".format(previous_time_stamp)
            log_out_file_common_name = "/data/exp_data/myeong0609/PAL-XFEL_20210514/analysis/minuit/run_{0}_{1}_{2}_".format(run_list[0], run_list[1], run_list[2])

    else:
        if aniso_cos_square_fit:
            dat_input_common_root = "/data/exp_data/myeong0609/gromacs/cos_square_calc/"
            # dat_infile_name = "cos_square_calc_1_box_" + dat_file_time_stamp + ".dat"
            dat_infile_name = "cos_square_calc_8_box_" + dat_file_time_stamp + ".dat"
            if p_width_10:
                dir_name = "pulse_10_fs/amp_50/"
                file_common_name = 'MD_sim_result_8_box_10fs_'
            elif p_width_20:
                # dir_name = "test_pcoupl/pulse_20_fs/amp_50/"
                dir_name = "pulse_20_fs/" + temperature + "/"+ temperature + "_amp_50_10ps/"
                # dir_name = "pulse_20_fs/" + temperature + "/"+ "amp_50_800nm/" + dir_time_stamp + "/"
                # file_common_name = temperature + '_MD_sim_result_8_box_20fs_'
                file_common_name = 'cos_square_calc_8_box_{}.dat'.format(dat_file_time_stamp)
            elif p_width_40:
                dir_name = "pulse_40_fs/amp_50_emtol_700/"
                file_common_name = 'MD_sim_result_8_box_40fs_'
            # dat_input_common_root = dat_input_common_root + dir_name
            dat_input_common_root = dat_input_common_root + dir_name #+ file_common_name
            file_out_dir_path = './anisotropy_fit_result/MD_result/anisotropy/'
            previous_result_load_path = file_out_dir_path + dir_name + file_common_name + "{}".format(previous_time_stamp)
            log_out_file_common_name = "/data/exp_data/myeong0609/PAL-XFEL_20210514/analysis/minuit/sim_result"
        else:
            dat_input_common_root = "../results/anisotropy/svd/run{0:04d}/run{0}".format(run_num)
            # dat_input_common_root = "../results/anisotropy/svd/run_{0:04d}/run{0}_{1}".format(run_num, dat_file_time_stamp)
            if fit_with_LC:
                dat_infile_name = "/run{0}_contributions_from_LSV_1_2.dat".format(run_num)
                if fit_with_droplet_LC:
                    dat_infile_name = "/run{0}_contributions_from_LSV_1_2_of_droplet.dat".format(run_num)
            else:
                dat_infile_name = "/run{0}-aniso-cut_RSV.dat".format(run_num)
            file_common_name = 'run{0:04d}_'.format(run_num)
            if num_of_RSV_to_fit ==2 :
                file_out_dir_path = './anisotropy_fit_result/run{0:04d}/unexpected/'.format(run_num)
            else:
                file_out_dir_path = './anisotropy_fit_result/run{0:04d}/anisotropy/'.format(run_num)
            previous_result_load_path = file_out_dir_path + file_common_name + "{}".format(previous_time_stamp)
            log_out_file_common_name = "/data/exp_data/myeong0609/PAL-XFEL_20210514/analysis/minuit/run_{0}_".format(run_num)

elif not anisotropy:
    if weighted_average:
        if len(run_list) == 2:
            dat_input_common_root = "../results/anisotropy/svd/run{0:04d}_{1:04d}_avg/run{0}_{1}_avg_{2}".format(run_list[0], run_list[1], dat_file_time_stamp)
            file_common_name = 'run{0:04d}_{1:04d}_avg_'.format(run_list[0], run_list[1])
            if fit_with_LC:
                dat_infile_name_averaged = "/run{0:04d}_{1:04d}_iso_contributions_from_LSV_1_2.dat".format(run_list[0], run_list[1])
            else:
                dat_infile_name_averaged = "/run{0:04d}_{1:04d}_avg-iso-cut_RSV.dat".format(run_list[0], run_list[1])
            if iso_second:
                file_out_dir_path = './anisotropy_fit_result/run{0:04d}_{1:04d}_avg/iso_second/'.format(run_list[0], run_list[1])
            elif iso_heating:
                file_out_dir_path = './anisotropy_fit_result/run{0:04d}_{1:04d}_avg/heating/'.format(run_list[0], run_list[1])
            previous_result_load_path = file_out_dir_path + file_common_name + "{}".format(previous_time_stamp)
            log_out_file_common_name = "/data/exp_data/myeong0609/PAL-XFEL_20210514/analysis/minuit/run_{0}_{1}_".format(run_list[0], run_list[1])

        elif len(run_list) == 3:
            dat_input_common_root = "../results/anisotropy/svd/run{0:04d}_{1:04d}_{2:04d}_avg/run{0}_{1}_{2}_avg_{3}/".format(run_list[0], run_list[1], run_list[2], dat_file_time_stamp)
            file_common_name = 'run{0:04d}_{1:04d}_{2:04d}_avg_'.format(run_list[0], run_list[1], run_list[2])
            if fit_with_LC:
                dat_infile_name_averaged = "run{0:04d}_{1:04d}_{2:04d}_iso_contributions_from_LSV_1_2.dat".format(run_list[0], run_list[1], run_list[2])
            else:
                dat_infile_name_averaged = "run{0:04d}_{1:04d}_{2:04d}_avg-iso-cut_RSV.dat".format(run_list[0], run_list[1], run_list[2])
            if iso_second:
                file_out_dir_path = './anisotropy_fit_result/run{0:04d}_{1:04d}_{2:04d}_avg/iso_second/'.format(run_list[0], run_list[1], run_list[2])
            elif iso_heating:
                file_out_dir_path = './anisotropy_fit_result/run{0:04d}_{1:04d}_{2:04d}_avg/heating/'.format(run_list[0], run_list[1], run_list[2])
            previous_result_load_path = file_out_dir_path + file_common_name + "{}".format(previous_time_stamp)
            log_out_file_common_name = "/data/exp_data/myeong0609/PAL-XFEL_20210514/analysis/minuit/run_{0}_{1}_{2}_".format(run_list[0], run_list[1], run_list[2])

    else:
        dat_input_common_root = "../results/anisotropy/svd/run{0:04d}/run{0}_{1}".format(run_num, dat_file_time_stamp)
        if fit_with_LC:
            dat_infile_name = "/run{0}_iso_contributions_from_LSV_1_2.dat".format(run_num)
        else:
            dat_infile_name = "/run{0}-iso-cut_RSV.dat".format(run_num)
        file_common_name = 'run{0:04d}_'.format(run_num)
        if iso_second :
            file_out_dir_path = './anisotropy_fit_result/run{0:04d}/iso_second/'.format(run_num)
        elif iso_heating:
            file_out_dir_path = './anisotropy_fit_result/run{0:04d}/heating/'.format(run_num)
        previous_result_load_path = file_out_dir_path + file_common_name + "{}".format(previous_time_stamp)
        log_out_file_common_name = "/data/exp_data/myeong0609/PAL-XFEL_20210514/analysis/minuit/run_{0}_".format(run_num)

saxs_analyze_with_minuit(num_of_RSV_to_fit, random_try_num=num_of_random_initialization)