import numpy as np
import h5py as h5
import os
import sys

mask_version = '210519_mask_dis1'

mask_fn = '/xfel/ffs/dat/ue_210514_FXL/scratch/' + mask_version + '.h5'


f1 = h5.File(mask_fn, 'a')
f1['/mask_version'] = np.string_(mask_version)

f1.close()

