#from minuit.CustomModelFit import CustomModelFit
from CustomModelFit import CustomModelFit


run_num = 11
run_list = [19, 20]
previous_time_stamp = "2022-02-10-20hr.7min.45sec"

dat_input_root = "../results/anisotropy/svd/"
num_of_RSV_to_fit = 11
need_negative = False
weighted_average = False
abs_sum_result_fit = False

heating = False
anisotropy = True
aniso_single_exponential = False
aniso_double_exponential = True
aniso_stretched_exponential = False

if abs_sum_result_fit:
    if anisotropy:
        dat_infile_name = "run{0}_aniso_abs_sum.dat".format(run_num)
        if len(run_list) == 2:
            dat_infile_name_averaged = "run{0:04d}_{1:04d}_avg-aniso-cut_RSV.dat".format(run_list[0], run_list[1])
        elif len(run_list) == 3:
            dat_infile_name_averaged = "run{0}_{1}_{2}_avg-aniso-cut_RSV.dat".format(run_list[0], run_list[1], run_list[2])
    elif heating:
        dat_infile_name = "run{0}-iso-cut_RSV.dat".format(run_num)
        if len(run_list) == 2:
            dat_infile_name_averaged = "run{0:04d}_{1:04d}_avg-iso-cut_RSV.dat".format(run_list[0], run_list[1])
        elif len(run_list) == 3:
            dat_infile_name_averaged = "run{0}_{1}_{2}_avg-iso-cut_RSV.dat".format(run_list[0], run_list[1], run_list[2])
else:
    if anisotropy:
        dat_infile_name = "run{0}-aniso-cut_RSV.dat".format(run_num)
        if len(run_list) == 2:
            dat_infile_name_averaged = "run{0:04d}_{1:04d}_avg-aniso-cut_RSV.dat".format(run_list[0], run_list[1])
        elif len(run_list) == 3:
            dat_infile_name_averaged = "run{0:04d}_{1:04d}_{2:04d}_avg-aniso-cut_RSV.dat".format(run_list[0], run_list[1], run_list[2])
    elif heating:
        dat_infile_name = "run{0}-iso-cut_RSV.dat".format(run_num)
        if len(run_list) == 2:
            dat_infile_name_averaged = "run{0:04d}_{1:04d}_avg-iso-cut_RSV.dat".format(run_list[0], run_list[1])
        elif len(run_list) == 3:
            dat_infile_name_averaged = "run{0}_{1}_{2}_avg-iso-cut_RSV.dat".format(run_list[0], run_list[1], run_list[2])

if weighted_average:
    now_biexponential_anal = CustomModelFit(dat_input_root, dat_infile_name_averaged, num_of_RSV_to_fit, run_num, need_negative, heating, anisotropy, weighted_average, run_list, previous_time_stamp, aniso_single_exponential, aniso_double_exponential, aniso_stretched_exponential)
else:
    now_biexponential_anal = CustomModelFit(dat_input_root, dat_infile_name, num_of_RSV_to_fit, run_num, need_negative, heating, anisotropy, weighted_average, run_list, previous_time_stamp, aniso_single_exponential, aniso_double_exponential, aniso_stretched_exponential)
now_biexponential_anal.custom_function_fit()
