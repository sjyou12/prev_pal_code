import numpy as np
import h5py as h5
import os
import matplotlib.pyplot as plt
from palxfel_scatter.diff_pair_1dcurve.Tth2qConvert import Tth2qConvert

def load_h5(dat_common_path, run_num, norm_start_q_idx, norm_end_q_idx):
    dat_input_path = dat_common_path + "run_{0:05d}_DIR/".format(run_num)
    now_1d_path = dat_input_path + "eh1rayMXAI_int/"
    temp_file_names = os.listdir(now_1d_path)
    common_file_names = temp_file_names
    print(now_1d_path)
    if run_num in [105, 106, 107, 108, 109]:
        common_file_names = common_file_names[:10]
    now_I0_path = dat_input_path + "eh1qbpm1_totalsum/"
    normalized_int_arr = []
    I0_norm_int_arr = []
    temp_int = []
    for file_idx, file_name in enumerate(common_file_names):
        now_1d_h5 = h5.File(now_1d_path + file_name, "r")
        now_I0_h5 = h5.File(now_I0_path + file_name, "r")
        now_delay_keys = list(now_1d_h5.keys())

        for key_idx, now_key in enumerate(now_delay_keys):
            each_int = np.array(now_1d_h5[now_key])
            each_I0 = np.array(now_I0_h5[now_key])
            I0_norm_int = each_int/each_I0
            range_norm_int = I0_norm_int/sum(I0_norm_int[norm_start_q_idx:norm_end_q_idx])
            # print(sum(I0_norm_int[norm_start_q_idx:norm_end_q_idx]))
            normalized_int_arr.append(range_norm_int)
            I0_norm_int_arr.append(I0_norm_int)
            temp_int.append(each_int)
        now_1d_h5.close()
        now_I0_h5.close()
        if (file_idx+1)%10 == 0:
            print("read {0}/{1} files".format(file_idx+1, len(common_file_names)))
    return normalized_int_arr, I0_norm_int_arr

def norm_range_q_idx_calc(dat_common_path, run_num):
    # set idx range for normalization
    def read_twotheta_value(tth_path):
        print("read tth value from file")
        now_tth_file_names = os.listdir(tth_path)
        twotheta_file = h5.File(tth_path + now_tth_file_names[0], 'r')
        twotheta_keys = list(twotheta_file.keys())

        now_tth_obj_name = twotheta_keys[0]
        twotheta_val = np.array(twotheta_file[now_tth_obj_name])
        print("read fixed 2theta value end. shape of value : ", twotheta_val.shape)
        q_val = np.array(tth_to_q_cvt.tth_to_q(twotheta_val))
        print("now q values : from ", q_val[0], "to", q_val[-1])
        return q_val

    now_tth_path = dat_common_path + "run_{0:05d}_DIR/eh1rayMXAI_tth/".format(run_num)
    tth_to_q_cvt = Tth2qConvert(20)
    q_val = read_twotheta_value(now_tth_path)
    NormStartQ = 6
    NormEndQ = 8
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

def plot_water_normalized_int(q_val, water_normalized_int, water_I0_norm_int, mag_water_normalized_int, mag_water_I0_norm_int, air_I0_norm_int):
    # avg_water_int = np.average(water_I0_norm_int, axis=0)
    # avg_mag_water_int = np.average(mag_water_I0_norm_int, axis=0)
    # avg_air_int = np.average(air_I0_norm_int, axis=0)
    avg_water_int = np.average(water_normalized_int, axis=0)
    avg_mag_water_int = np.average(mag_water_normalized_int, axis=0)
    avg_air_int = np.average(air_I0_norm_int, axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(q_val, avg_water_int, label="pure water")
    plt.plot(q_val, avg_mag_water_int, label="magnetized water")
    # plt.plot(q_val, avg_air_int, label="air")
    plt.axvline(x=1.5, color = 'r', linestyle='--')
    plt.axvline(x=3.5, color = 'r', linestyle='--')
    plt.xlabel("q (1/A)")
    plt.ylabel("Intensity (a.u.)")
    plt.title("I0 normalized Scattering pattern")
    plt.legend()
    plt.show()

    # plt.plot(q_val, avg_water_int-avg_air_int, label="pure water")
    # plt.plot(q_val, avg_mag_water_int-avg_air_int, label="magnetized water")
    # plt.title("Normalized Scattering pattern")
    # plt.legend()
    # plt.show()

    diff_int = avg_mag_water_int - avg_water_int
    plt.figure(figsize=(10, 6))
    plt.plot(q_val, diff_int)
    plt.title("Difference pattern of Magnetized water")
    plt.axvline(x=1.5, color = 'r', linestyle='--')
    plt.axvline(x=3.5, color = 'r', linestyle='--')
    plt.axhline(y=0, color = 'gray')
    plt.xlabel("q (1/A)")
    plt.ylabel("Intensity (a.u.)")
    plt.legend()
    plt.show()

mag_run_num = 197
mag_air_run_num = 201
mag_dark_run_num = 200
water_run_num = 105
water_air_run_num = 64
water_dark_run_num = 112

dat_common_path = "/xfel/ffs/dat/scan/"
norm_start_q_idx, norm_end_q_idx, q_val = norm_range_q_idx_calc(dat_common_path, mag_run_num)
mag_water_normalized_int, non_normalized_mag_int = load_h5(dat_common_path, mag_run_num, norm_start_q_idx, norm_end_q_idx)
mag_air_normalized_int, non_normalized_mag_air = load_h5(dat_common_path, mag_air_run_num, norm_start_q_idx, norm_end_q_idx)
mag_dark_normalized_int, non_normalized_mag_dark = load_h5(dat_common_path, mag_dark_run_num, norm_start_q_idx, norm_end_q_idx)
# water_normalized_int, water_I0_norm_int = load_h5(dat_common_path, pure_water_run, norm_start_q_idx, norm_end_q_idx)
water_normalized_int, non_normalized_water_int = load_h5(dat_common_path, water_run_num, norm_start_q_idx, norm_end_q_idx)
# air_normalized_int, air_I0_norm_int = load_h5(dat_common_path, air_run_num, norm_start_q_idx, norm_end_q_idx)
water_air_normalized_int, non_normalized_water_air = load_h5(dat_common_path, water_air_run_num, norm_start_q_idx, norm_end_q_idx)
water_dark_normalized_int, non_nomralized_water_dark = load_h5(dat_common_path, water_dark_run_num, norm_start_q_idx, norm_end_q_idx)

# plt.plot(q_val, np.average(non_normalized_water_int, axis=0), label='not normalized water')
# plt.plot(q_val, np.average(non_normalized_water_air, axis=0), label='not normalized air')
# plt.plot(q_val, np.average(non_nomralized_water_dark, axis=0), label='not normalized dark')
# plt.legend()
# plt.axvline(x=1.0, color='r', linestyle='--')
# plt.show()
#
# plt.plot(q_val, np.average(non_normalized_mag_int, axis=0), label='not normalized mag')
# plt.plot(q_val, np.average(non_normalized_mag_air, axis=0), label='not normalized air')
# plt.plot(q_val, np.average(non_normalized_mag_dark, axis=0), label='not normalized dark')
# plt.legend()
# plt.axvline(x=1.0, color='r', linestyle='--')
# plt.show()

temp_mag = np.average(non_normalized_mag_int, axis=0) - np.average(non_normalized_mag_dark, axis=0)
temp_mag_air = np.average(non_normalized_mag_air, axis=0) - np.average(non_normalized_mag_dark, axis=0)
mag = temp_mag - temp_mag_air

temp_water = np.average(non_normalized_water_int, axis=0) - np.average(non_nomralized_water_dark, axis=0)
temp_water_air = np.average(non_normalized_water_air, axis=0) - np.average(non_nomralized_water_dark, axis=0)
water = temp_water - temp_water_air

# plt.plot(q_val, mag, label='bg sub mag')
# plt.plot(q_val, water, label='bg sub water')
# plt.legend()
# plt.axvline(x=1.0, color='r', linestyle='--')
# plt.show()

q_start = 8
q_end = 10

q_start_idx = np.where(q_val > q_start)[0][0]
q_end_idx = np.where(q_val > q_end)[0][0]

mag_norm_int = mag / np.average(mag[q_start_idx:q_end_idx])
water_norm_int = water / np.average(water[q_start_idx:q_end_idx])

plt.plot(q_val, mag_norm_int, label='mag norm')
plt.plot(q_val, water_norm_int, label='water norm')
plt.legend()
plt.axvline(x=1.0, color='r', linestyle='--')
plt.show()

plt.plot(q_val, mag_norm_int- water_norm_int, label='mag norm - water norm')
plt.axhline(0, color='gray')
plt.legend()
plt.show()

# plot_water_normalized_int(q_val, water_normalized_int, water_I0_norm_int, mag_water_normalized_int, mag_water_I0_norm_int ,air_I0_norm_int)


