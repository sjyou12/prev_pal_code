import numpy as np
# from ScattAnalyzer import ScattAnalyzer
from iminuit.cost import LeastSquares
from iminuit import Minuit
import matplotlib.pyplot as plt

class MinuitParam:
    def __init__(self, p_name, p_limit, p_is_fixed, p_max_oom, p_min_oom, p_fixed_value, no_random_initial):
        self.name = p_name  # string
        self.is_fixed = p_is_fixed
        self.fixed_value = p_fixed_value
        self.limit = p_limit  # range a ~ param ~ b -> (a, b)
        self.fit_results = []
        self.max_order_of_magnitude = p_max_oom
        self.min_order_of_magnitude = p_min_oom
        self.no_random_initial = no_random_initial

def heat_diff_sum_square(num_rsv_to_fit, random_try_num = 5):
    # now_custom_fit.file_common_name = file_common_name
    data_yerr = 1
    fit_delay_range = (-3, 3) # TODO: fitting range for time delay & chi2
    fit_data_mask = (now_time_delay > fit_delay_range[0]) & (now_time_delay < fit_delay_range[1])
    now_time_delay_fit = now_time_delay[fit_data_mask]
    now_target_rsv_fit = now_target_rsv[fit_data_mask]
    LC_least_square = LeastSquares(now_time_delay_fit, now_target_rsv_fit, data_yerr, exp_heating)
    a2_init = -0.2
    t2_init = 1
    x0_init = 0
    m = Minuit(LC_least_square, a2=a2_init, t2=t2_init, x0=x0_init)  # , c3=c3_init)
    m.migrad()
    m.hesse()
    plot_rsv_linear_n_log_x_scale(now_time_delay, [now_target_rsv, exp_heating(now_time_delay, *m.values)], ['data', 'fit'], "time delay (ps)", "RSV", 3, "chi2 = {}".format(m.fval))
    # plt.plot(now_time_delay, exp_heating(now_time_delay, *m.values), label='fit')
    # plt.plot(now_time_delay, now_target_rsv, label='data')
    # plt.show()
    print("chi2 of this fit is " + str(m.fval))
    print("a2 = " + str(m.values['a2']))
    print("t2 = " + str(m.values['t2']) + "ps")
    print("x0 = " + str(m.values['x0']) + "ps")

def exp_heating(q_val, a2, t2, x0):
    exponential_temp = []
    for q_idx in range(len(q_val)):
        if (q_val[q_idx] - x0) > 0:
            exponential_temp.append(a2 * np.exp(-((q_val[q_idx] - x0) / t2)) - a2)
            # if temp_val >= 0:
            #     exponential_temp.append(temp_val)
            # else :
            #     exponential_temp.append(0.0)
        else:
            exponential_temp.append(0.0)
    return exponential_temp

def plot_rsv_linear_n_log_x_scale(x_data, y_data, label_list, x_label, y_label, scale_cut_x_val, suptitle):
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
        log_ax.plot(log_x_data, log_y_data, label=label_list[run_idx])

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


dat_common_path = "/xfel/ffs/dat/ue_230427_FXL/analysis/results/anisotropy/svd/"
# TODO: change run_num & num_of_RSV_to_fit
run_num = 221
dat_input_path = dat_common_path + "run{0:04d}/run{0}-iso-cut_RSV.dat".format(run_num)
file_out_dir_path = dat_common_path + "run{0:04d}/".format(run_num)

num_of_random_initialization = 5
num_of_calculation = 621 #TODO change if tim scale is approaching ns scale
time_step = 10E-3
start_time = -3

anisotropy = False
iso_heating = True

num_of_RSV_to_fit = 2

now_rsv_file = np.loadtxt(dat_input_path, skiprows=1)
now_time_delay = now_rsv_file[:, 0]
now_target_rsv = now_rsv_file[:, num_of_RSV_to_fit]

if now_target_rsv[now_time_delay == 1] < 0:
    now_target_rsv = -now_target_rsv

heat_diff_sum_square(num_of_RSV_to_fit, num_of_random_initialization)




