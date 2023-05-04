import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
import os

compare_run_list = [6, 9]
data_common_path = "/data/exp_data/myeong0609/azi_intg_test/"
target_data=[]
avg_data = []
for run_idx, run_num in enumerate(compare_run_list):
    temp_target_data = np.load(data_common_path + "watersum_test_run{0}.npy".format(run_num), allow_pickle=True)
    avg_data.append(np.average(temp_target_data[0], axis=0))
    if run_idx == 0:
        common_q_val = np.load(data_common_path + "run{0:04d}_00001_DIR/common_q_val.npy".format(run_num))
I3m_max = np.max(avg_data[0])
water_max = np.max(avg_data[1])
ratio = I3m_max/water_max
print(ratio)
plt.plot(common_q_val, avg_data[0]/ratio, label='I3- data')
plt.plot(common_q_val, avg_data[1], label='water data')
plt.legend()
plt.show()
plt.plot(common_q_val, avg_data[0], label='I3- data')
plt.plot(common_q_val, avg_data[1], label='water data')
plt.legend()
plt.show()
