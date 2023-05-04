import h5py as h5
import numpy as np
import matplotlib.pyplot as plt

img_file_path = "/xfel/ffs/dat/scan/210516_powder_1_00006_DIR/eh1rayMX_img/"
img_name = "00000001_00000030.h5"

file_full_name = img_file_path + img_name
now_file = h5.File(file_full_name, 'r')
keys_now_file = list(now_file.keys())

img_datas = []
for each_key in keys_now_file:
    img_datas.append(np.array(now_file[each_key]))

avg_img = np.average(np.array(img_datas), axis=0)
avg_img = np.flip(avg_img, 0)

print("average done")

# plt.title("1st img")
# plt.pcolor(img_datas[0])
# plt.colorbar()
# plt.show()

out_bin_name = "210516_powder_00006_avg.bin"
out_img_path = "../results/img/"
out_img_file = out_img_path + out_bin_name
# imwrite(out_img_file, avg_img, photometric='MINISBLACK')  # try 'MINISWHITE"
avg_img.astype('uint16').tofile(out_img_file)

