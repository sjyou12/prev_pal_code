import numpy as np
import matplotlib.pyplot as plt
import os
import h5py as h5
from palxfel_scatter.diff_pair_1dcurve.Tth2qConvert import Tth2qConvert


def norm_range_q_idx_calc(dat_common_path, run_num):
    # set idx range for normalization
    def read_twotheta_value(tth_path):
        print("read tth value from file")

        # now_tth_path = self.FileCommonRoot + "run" + str(self.runList[0]) + "/"
        now_file_name = os.listdir(tth_path)[0]
        now_tth_path = tth_path + now_file_name

        twotheta_file = h5.File(now_tth_path, 'r')
        twotheta_keys = list(twotheta_file.keys())
        # print(len(twotheta_keys), "key values, head : ", twotheta_keys[0], "tail : ", twotheta_keys[-1])

        now_tth_obj_name = twotheta_keys[0]
        twotheta_val = np.array(twotheta_file[now_tth_obj_name])
        print("read fixed 2theta value end. shape of value : ", twotheta_val.shape)
        q_val = np.array(tth_to_q_cvt.tth_to_q(twotheta_val))
        print("now q values : from ", q_val[0], "to", q_val[-1])
        return q_val

    now_tth_path = dat_common_path + "run_{0:05d}_DIR/eh1rayMXAI_tth/".format(run_num)
    tth_to_q_cvt = Tth2qConvert(20)
    q_val = read_twotheta_value(now_tth_path)
    NormStartQ = 0.7
    NormEndQ = 1
    # TODO : make new method for set norm / pairing range
    if len(q_val) == 0:
        print("no q value now!")
    norm_q_range_start_idx = int(np.where(q_val >= NormStartQ)[0][0])
    # this index is not included in water q range!!!
    norm_q_range_after_idx = int(np.where(q_val > NormEndQ)[0][0])

    print("( normalization] {0} is in {1}th index ~ {2} is in {3}th index )".format(
        q_val[norm_q_range_start_idx],
        norm_q_range_start_idx,
        q_val[norm_q_range_after_idx],
        norm_q_range_after_idx))

    return norm_q_range_start_idx, norm_q_range_after_idx, q_val

run_num = 64
dat_common_path = "/xfel/ffs/dat/scan/"
now_dat_path = dat_common_path + "run_{0:05d}_DIR/".format(run_num)
curve_path = now_dat_path + "eh1rayMXAI_int/"
I0_path = now_dat_path + "eh1qbpm1_totalsum/"
dat_save_path = "/xfel/ffs/dat/ue_230427_FXL/analysis/results/each_run_watersum_int/"

now_file_names = os.listdir(curve_path)
norm_int_arr = []
each_file_avg_int = []

norm_start_q_idx, norm_end_q_idx, q_val = norm_range_q_idx_calc(dat_common_path, run_num)

for file_idx, file_name in enumerate(now_file_names):
    now_curve_file = h5.File(curve_path + file_name, "r")
    now_I0_file = h5.File(I0_path + file_name, "r")
    now_file_keys = list(now_curve_file.keys())
    for each_key in now_file_keys:
        now_int_val = np.array(now_curve_file[each_key])
        now_I0_val = np.array(now_I0_file[each_key])
        I0_norm_int_val = now_int_val / now_I0_val
        # norm_int_val = I0_norm_int_val/sum(I0_norm_int_val[norm_start_q_idx:norm_end_q_idx])
        norm_int_arr.append(I0_norm_int_val)
    each_file_avg_int.append(np.average(norm_int_arr, axis=0))
    if (file_idx+1)%10 == 0:
        print("read {}/{}".format(file_idx+1, len(now_file_names)))

run_avg_int = np.average(each_file_avg_int, axis=0)
plt.plot(q_val, run_avg_int)
# plt.title('norm range q 0.7~1.0')
plt.show()
np.savetxt(dat_save_path + "run_{}_static_avg_int.txt".format(run_num), run_avg_int)