import numpy as np
import math
import scipy.fft as fft
import matplotlib.pyplot as plt
import os
from codes.PDF.function_PDF import pdfConverter

def set_initial_condition(dat_path, dat_file_name):
    dat_file = np.loadtxt(dat_path + dat_file_name, skiprows=1, delimiter=',')
    number_density = 33.3679 * 1E-3 # number density per A^3

    return dat_file, number_density

def FT(q_val, data, number_density, Q_max_arr, Q_min):
    a = 2.8
    b = 0.5
    total_FT_result = []
    total_r_arr = []
    if Q_min == 0:
        q_min_idx = 1
    else:
        q_min_idx = np.where(q_val == Q_min)[0][0]
    for max_q in Q_max_arr:
        q_max = max_q
        points_space = np.round(q_val[1] - q_val[0], 10) # gap of original space
        q_max_idx = np.where(q_val == q_max)[0][0]
        sliced_q_val = q_val[q_min_idx:q_max_idx+1]
        r_arr = fft.fftfreq(len(sliced_q_val), points_space)[:len(sliced_q_val)//2]
        total_r_arr.append(r_arr)
        modified_fxn = []
        delta_r_arr = (math.pi/q_max) * (1-np.exp(-(np.abs(r_arr-a)/b)))
        for delta_r in delta_r_arr:
            modified_fxn.append(np.sin(sliced_q_val * delta_r)/(sliced_q_val * delta_r))
        FT_result = []
        for r_idx, r in enumerate(r_arr):
            temp_sum_arr = []
            for q_idx, q in enumerate(sliced_q_val):
                temp_sum_arr.append(1 + 1/(2*math.pi*number_density*r) * (modified_fxn[r_idx][q_idx] * q * (data[q_idx]-1)) * (np.sin(q*r)))
            FT_result.append(np.sum(temp_sum_arr))
        total_FT_result.append(FT_result)

    return total_r_arr, total_FT_result

dat_common_path = "/xfel/ffs/dat/scan/"
run_num = 9

Q_min = 0.5
Q_max_arr = [11.5, 26]

pdf_generator = pdfConverter(dat_common_path, run_num)
pdf_generator.load_1d()
# pdf_generator.load_ref_data(Q_max_arr, Q_min)
# pdf_generator.FT_science_paper()
pdf_generator.calc_FT()
pdf_generator.plt_each_PDF()
# pdf_generator.plot_contour()







