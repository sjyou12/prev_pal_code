import h5py as h5
import numpy as np
import re
import matplotlib.pyplot as plt
import os
import pyFAI

bkg_int_arr =[]

dat_common_path = "/data/exp_data/myeong0609/SACLA_20230120/background/"
bkg_name_list = os.listdir(dat_common_path)
AzimuthalIntegrator = pyFAI.load(dat_common_path + "CeO2_1218653_14keV.poni")
mask_file_name = '1218653_mask.h5'
now_mask_file = h5.File(dat_common_path + mask_file_name, 'r')
mask_img = now_mask_file['mask']

for file_idx, file_name in enumerate(bkg_name_list):
    try:
        now_h5_file = h5.File(dat_common_path + file_name, 'r')
        now_bkg = np.array(now_h5_file['run_0']['detector_2d_assembled_1']['tag_0']['detector_data'])
    except:
        continue
    q_val, intensity = AzimuthalIntegrator.integrate1d(now_bkg, npt=1024, polarization_factor=0.996, mask=mask_img, method='splitpixel', unit="q_A^-1")
    now_bkg_sum = np.sum(intensity[324:799])
    bkg_int_arr.append(now_bkg_sum)

plt.hist(bkg_int_arr, bins=200, log=True)
plt.title("bkg's integration value between 1.5~3.5")
plt.xlabel("integration value")
plt.ylabel("frequency")
plt.show()


