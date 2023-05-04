import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

input_file_path = "/home/myeong0609/"
input_file_name = "PAL-XFEL_221118_powder_1d"
# input_file_name = "run00056_AgBh_1d_jung"
xrd_file_name = "AgBh_XRD.csv"

xrd_input_dat = np.loadtxt(input_file_path + xrd_file_name, delimiter=',')
xrd_q = xrd_input_dat[:, 0]
xrd_int = xrd_input_dat[:, 1]

input_dat = np.loadtxt(input_file_path + input_file_name, skiprows=1, delimiter=';')
q_val = input_dat[:, 0]
int_val = input_dat[:, 1]
PAL_peak_q_range = [0.2, 0.24]#[0.2, 0.24]
XRD_peak_q_range = [0.2, 0.24]#[1.3, 1.45

class peak_info:

    def __init__(self, q_val, int_val, peak_range, peak_num):
        peak_range_info = peak_range
        peak_num = peak_num
        self.int_val = int_val
        self.q_val = q_val

        self.peak_q_mask = (q_val > peak_range_info[0]) & (q_val < peak_range_info[1])
        self.peak_q_arr = q_val[self.peak_q_mask]
        self.peak_min_q_idx = np.where(q_val == self.peak_q_arr[0])[0][0]
        self.peak_max_q_idx = np.where(q_val == self.peak_q_arr[-1])[0][0]
        if peak_num == 1:
            self.regress_start_idx = self.peak_min_q_idx - 1
            self.regress_end_idx = self.peak_max_q_idx + 2
        elif peak_num == 2:
            self.regress_start_idx = self.peak_min_q_idx - 20
            self.regress_end_idx = self.peak_max_q_idx + 20

        self.slope = None
        self.intercept = None
        self.regressed_int_val = []

    def peak_process(self):
        peak_integration_val = np.sum(self.int_val[self.peak_q_mask])
        x_axis = np.concatenate((self.q_val[self.regress_start_idx:self.peak_min_q_idx], self.q_val[self.peak_max_q_idx:self.regress_end_idx]))
        y_axis = np.concatenate((self.int_val[self.regress_start_idx:self.peak_min_q_idx],self.int_val[self.peak_max_q_idx:self.regress_end_idx]))
        linregress_result = stats.linregress(x_axis, y_axis)
        self.slope, self.intercept = linregress_result.slope, linregress_result.intercept
        bkg_arr = np.array(self.intercept + self.slope * self.q_val[self.peak_q_mask])
        bkg_val = np.sum(bkg_arr)
        peak_integration_val = peak_integration_val - bkg_val

        return peak_integration_val

    def linear_regression(self):
        regressed_arr = np.array(self.intercept + self.slope * self.q_val[self.peak_q_mask])
        self.regressed_int_val = self.int_val[self.peak_q_mask] - regressed_arr

PAL_peak_info = peak_info(q_val, int_val, PAL_peak_q_range, 2)
# PAL_peak_info = peak_info(xrd_q, xrd_int, PAL_peak_q_range, 1)
XRD_peak_info = peak_info(xrd_q, xrd_int, XRD_peak_q_range, 2)

first_peak_proc_int = PAL_peak_info.peak_process()
second_peak_proc_int = XRD_peak_info.peak_process()

PAL_peak_info.linear_regression()
XRD_peak_info.linear_regression()


print("integration value of each peak \n 1st : {0}, 2nd : {1}".format(second_peak_proc_int, first_peak_proc_int))
print("ratio of each value : {}".format(first_peak_proc_int/second_peak_proc_int))
print("q values of first peak")
print("Integration range : {0}~{1}, Linear regression range : {2}~{0} and {1}~{3}".format(PAL_peak_info.peak_q_arr[0], PAL_peak_info.peak_q_arr[-1], PAL_peak_info.q_val[PAL_peak_info.regress_start_idx], PAL_peak_info.q_val[PAL_peak_info.regress_end_idx]))
print("q values of second peak")
print("Integration range : {0}~{1}, Linear regression range : {2}~{0} and {1}~{3}".format(XRD_peak_info.peak_q_arr[0], XRD_peak_info.peak_q_arr[-1], XRD_peak_info.q_val[XRD_peak_info.regress_start_idx], XRD_peak_info.q_val[XRD_peak_info.regress_end_idx]))

plt.plot(PAL_peak_info.q_val[PAL_peak_info.peak_q_mask], PAL_peak_info.regressed_int_val, label='PAL')
# plt.plot(PAL_peak_info.q_val, PAL_peak_info.int_val, label='PAL')
plt.plot(XRD_peak_info.q_val[XRD_peak_info.peak_q_mask], (XRD_peak_info.regressed_int_val/1E3)*0.8, label='XRD')
plt.xlabel('q (A^-1)')
plt.ylabel('Intensity (a.u.)')
plt.title('AgBh powder curve from 202211 PAL and XRD')
plt.legend()
plt.xlim(0.2, 0.24)#  [1.322, 1.424]
# plt.ylim(0, 10)
plt.show()

# plt.plot(PAL_peak_info.q_val, PAL_peak_info.int_val/1E5, label='PAL')
# plt.xlabel('q (A^-1)')
# plt.ylabel('Intensity (a.u.)')
# plt.title('AgBh powder curve from 202304 PAL')
# plt.xlim(0.2, 0.24)#  [1.322, 1.424]
# plt.ylim(0, 0.1)
# plt.show()



