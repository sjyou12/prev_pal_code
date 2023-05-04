import numpy as np
import scipy.io
import matplotlib.pyplot as plt

mat_file_path = "/home/myeong0609/"
mat_file_name = "det_darknoise.mat"

mat_file_data = scipy.io.loadmat(mat_file_path + mat_file_name)
det_img = mat_file_data['bg_mean']
intg_curve = np.transpose(mat_file_data['psd'])[0]
x_axis = range(len(mat_file_data['psd']))
plt.plot(x_axis, intg_curve)
plt.xlabel('q idx')
plt.ylabel('intensity (a.u.)')
plt.title("Jungfrau dark image")
plt.show()

np.save('/home/myeong0609/det_2d_img.npy', det_img)