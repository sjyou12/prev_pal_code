from ScattAnaylzer import ScattAnaylzer
import numpy as np

test_input_root = "data/"
# test_infile_name = "diff_Ih_170K_250uJ.dat"
test_infile_name = "Ih_difference_SAXS_example.dat"
reduced_input_data_num = 200
test_radi_set_num = 1000

now_saxs_anal = ScattAnaylzer(test_input_root, test_infile_name, reduced_input_data_num, test_radi_set_num)


def exp_decay(q_val, a1, t1, x0, y0, a2, t2):
    exponential_temp = []
    if len(q_val) != 1:
        for q_idx in range(len(q_val)):
            if (q_val[q_idx] - x0 * self.para_unit['x0']) > 0:
                exponential_temp.append(a1 * self.para_unit['a1'] * np.exp(
                    -((q_val[q_idx] - x0 * self.para_unit['x0']) / (t1 * self.para_unit['t1']))) + a2 * self.para_unit[
                                            'a2'] * np.exp(
                    -((q_val[q_idx] - x0 * self.para_unit['x0']) / (t2 * self.para_unit['t2']))) + y0)
            else:
                exponential_temp.append(float(0))

        return exponential_temp

    elif len(q_val) == 1:
        if (q_val[0] - x0 * self.para_unit['x0']) > 0:
            calc_val = (a1 * self.para_unit['a1'] * np.exp(
                -((q_val[0] - x0 * self.para_unit['x0']) / (t1 * self.para_unit['t1']))) + a2 * self.para_unit[
                            'a2'] * np.exp(
                -((q_val[0] - x0 * self.para_unit['x0']) / (t2 * self.para_unit['t2']))) + y0)
        else:
            calc_val = float(0)

        return calc_val


def gaussian(q_val, FWHM):
    gaussian_temp = (1 / (FWHM * self.para_unit['FWHM'] * (np.sqrt(2 * np.pi))) * np.exp(
        -pow(q_val / (FWHM * self.para_unit['FWHM']), 2.0) / 2.0))
    return gaussian_temp

def function_convolution(time_list, a1, t1, x0, y0, a2, t2, FWHM):
    # exponential_temp = exp_decay(q_val, a1, t1, x0, y0, a2, t2)
    time_list = self.calculate_time_list
    gaussian_temp = gaussian(time_list, FWHM)
    # convoluted_function = np.convolve(gaussian_temp, exponential_temp, 'same')
    # for value in (convoluted_function):
    #     text = str(value)
    #     print(text)
    temp_convoluted_function = []
    convoluted_function = []
    for time_point_idx in range(len(time_list)):
        if time_point_idx == 0:
            temp_calc_time = []
            for calc_idx in range(num_of_calculation):
                temp_calc_time.append(time_list[time_point_idx] - time_list[calc_idx])
            exponential_temp = exp_decay(temp_calc_time, a1, t1, x0, y0, a2, t2)
        else:
            temp_time_list = []
            # temp_calc_time = np.roll(temp_calc_time, 1)
            # temp_calc_time[0] = (time_list[time_point_idx] - time_list[0])
            temp_time_list.append(time_list[time_point_idx] - time_list[0])
            exponential_temp = np.roll(exponential_temp, 1)
            exponential_temp[0] = exp_decay(temp_time_list, a1, t1, x0, y0, a2, t2)
        temp_calc_val = np.multiply(exponential_temp, gaussian_temp) * time_step
        temp_convolution_val = np.sum(temp_calc_val)
        temp_convoluted_function.append([time_list[time_point_idx], temp_convolution_val])
    gaussian_temp = []
    # temp_convoluted_function.append([q_val[num_time_point], gaussian_temp[num_time_point]*exponential_temp[len(q_val)-num_time_point-1]*time_step])

    for idx in range(len(time_list)):
        if self.temporary_mask:
            if time_list[idx] <= 0.5E-12:
                if temp_convoluted_function[idx][0] in self.input_time_delay:
                    convoluted_function.append(temp_convoluted_function[idx][1])
                else:
                    continue
            else:
                continue
        else:
            if temp_convoluted_function[idx][0] in self.input_time_delay:
                convoluted_function.append(temp_convoluted_function[idx][1])
            else:
                continue
    # temp_convoluted_function = np.array(temp_convoluted_function)
    # return temp_convoluted_function
    return convoluted_function

now_saxs_anal.set_scattering_function(saxs_intensity)

# now_saxs_anal.theo_int_calc()
# now_saxs_anal.calc_data_file_out()
# now_saxs_anal.fitting_first_delay()

random_try_num = 100
now_saxs_anal.set_fit_param(param_name="num_particle", left_limit=0, max_oom=8)
now_saxs_anal.set_fit_param(param_name="radius", left_limit=0, max_oom=2)
now_saxs_anal.set_fit_param(param_name="baseline", max_oom=2)
now_saxs_anal.random_initial_fit(random_try_num)
