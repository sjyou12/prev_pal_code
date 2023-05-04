from palxfel_scatter.diff_pair_1dcurve.MultiRunProc import MultiRunProc
import numpy as np
from matplotlib import pyplot as plt
import os
import h5py as h5
from palxfel_scatter.diff_pair_1dcurve.Tth2qConvert import Tth2qConvert


def norm_range_q_idx_calc(dat_common_path, run_num):
    # set idx range for normalization
    def read_twotheta_value(tth_path):
        print("read tth value from file")

        # now_tth_path = self.FileCommonRoot + "run" + str(self.runList[0]) + "/"
        now_tth_path = tth_path + "001_001_001.h5"
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
    NormStartQ = 1.5
    NormEndQ = 3.5
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

    return norm_q_range_start_idx, norm_q_range_after_idx

def norm_given_range(norm_range_start_idx, norm_range_after_idx, intensity_val):
    norm_range_sum = sum(intensity_val[norm_range_start_idx : norm_range_after_idx])
    try:
        intensity_val = list((np.array(intensity_val) / norm_range_sum) * 1E7)
        return intensity_val
    except:
        print("normalization error ")
        print(intensity_val)
        print(norm_range_sum)

dat_common_path = "/xfel/ffs/dat/scan/"
run_num = 29
now_dat_path = dat_common_path + "run_{0:05d}_DIR/eh1rayMXAI_int/".format(run_num)

norm_q_range_start_idx, norm_q_range_after_idx = norm_range_q_idx_calc(dat_common_path, run_num)

now_file_names = os.listdir(now_dat_path)
norm_int_arr = []
for file_idx, file_name in enumerate(now_file_names):
    now_h5_File = h5.File(now_dat_path + file_name, "r")
    now_h5_keys = list(now_h5_File.keys())
    for now_key in now_h5_keys:
        now_int = list(now_h5_File[now_key])
        now_int = np.array(now_int)
        norm_int = norm_given_range(norm_q_range_start_idx, norm_q_range_after_idx, now_int)
        norm_int_arr.append(sum(norm_int))
    now_h5_File.close()
    if (file_idx+1) % 10 == 0:
        print("read {0} / {1} file".format(file_idx+1, len(now_file_names)))

plt.hist(norm_int_arr, bins=200, log=True)
plt.title("Normalized intensity histogram of run {}".format(run_num))
plt.xlabel("Integration value")
plt.ylabel("Frequency")
plt.show()



