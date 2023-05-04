import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
from scipy import interpolate


def q_intrp(q_from, q_to, q_step_size):
    custom_q_val = np.arange(q_from, q_to + q_step_size, q_step_size)

    return custom_q_val


def interpolation(before_interp_q_val, before_interp_data, int_after_q_val):
    int_func = interpolate.interp1d(before_interp_q_val, before_interp_data)
    int_after = int_func(int_after_q_val[:-1])

    return int_after


test_file = np.loadtxt("/data/exp_data/myeong0609/PAL-XFEL_20210514/analysis/results/anisotropy/svd/run0009/run9_2023-02-19-00hr.57min.13sec/run9-aniso-cut_LSV.dat", skiprows=1)
now_q_val = test_file[:,0]
now_aniso_1 = test_file[:, 1]
now_aniso_2 = test_file[:, 2]

data_save_arr = []

interp_q_val = q_intrp(0.518437367361475, 3.49657259740, 0.054339)
after_interp_aniso_1 = interpolation(now_q_val, now_aniso_1, interp_q_val)
after_interp_aniso_2 = interpolation(now_q_val, now_aniso_2, interp_q_val)

data_save_arr.append(interp_q_val)
data_save_arr.append(after_interp_aniso_1)
data_save_arr.append(after_interp_aniso_2)

np.save("/data/exp_data/myeong0609/PAL-XFEL_20210514/analysis/results/anisotropy/svd/run0009/run9_2023-02-19-00hr.57min.13sec/q_interp_data.npy", data_save_arr)

