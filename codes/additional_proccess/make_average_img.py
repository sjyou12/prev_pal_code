import h5py as h5
import numpy as np

# num_img_to_avg = 30
data_input_root = "/data/exp_data/PAL-XFEL_20210514/rawdata/run0037_00001_DIR/eh1rayMX_img/"
file_name = "00000001_00000300.h5"

temp_img = []

now_file = h5.File(data_input_root + file_name, 'r')
now_file_keys = list(now_file.keys())
for key in now_file_keys:
    temp_img.append(np.array(now_file[key]))

temp_img = np.average(temp_img, axis=0)
out_file_name = "avg_img_for_mask"
output_file_name = data_input_root + out_file_name + ".h5"
h5Fp = h5.File(output_file_name, 'w')
# for key, intensity in delay_int_dict.items():
#     h5Fp.create_dataset(key, data=intensity)
h5Fp.create_dataset(out_file_name, data=temp_img)
h5Fp.close()
print("\n" + output_file_name + " save completed\n")


# import numpy as np
# import h5py as h5
# import sys
# from PIL import Image
#
#
# # aborted_run = [26, 39, 68, 83]
# # TODO
# # range(5, 10)
# run_to_anal = [1]
# img_name = "copy_avg_img.h5"
# out_file_name = img_name[:-3] + ".tiff"
# for each_run_idx in run_to_anal:
#     # if each_run_idx in aborted_run:
#     #     continue
#     img_file_path = "/data/exp_data/PAL-XFEL_20210514/rawdata/210515_powder_00009_DIR/eh1rayMX_img/"
#     # out_file_name = "211106_{0:05d}_shot2.tiff".format(each_run_idx)
#     img_name = "00000001_00000300.h5"
#
#     filename = img_file_path + img_name
#     output_filename = img_file_path + out_file_name
#     raymx_data = np.zeros((1440, 1440))
#     shot_number = 2
#
#     print(filename)
#     with h5.File(filename, 'r') as f:
#         keys = list(f.keys())
#         for key in keys:
#             if key == '1621048067.5957222_97632':
#                 raymx_data_avr = np.array(f[key])
#
#         print(raymx_data_avr.shape)
#         im = Image.fromarray(raymx_data_avr)
#
#         im.save(output_filename)
#         print("make img done : ", out_file_name[:12])
#
#         # sys.exit(1)

# import numpy as np
# import h5py as h5
# import sys
# from PIL import Image
#
#
# # aborted_run = [26, 39, 68, 83]
# # TODO
# # range(5, 10)
# run_to_anal = [1]
# for each_run_idx in run_to_anal:
#     # if each_run_idx in aborted_run:
#     #     continue
#     img_file_path = "/data/exp_data/PAL-XFEL_20210514/rawdata/210515_powder_00009_DIR/eh1rayMX_img/"
#     out_file_name = "211106_{0:05d}_shot2.tiff".format(each_run_idx)
#     img_name = "00000001_00000300.h5"
#
#     filename = img_file_path + img_name
#     output_filename = img_file_path + out_file_name
#     raymx_data = np.zeros((1440, 1440))
#     shot_number = 300
#
#     with h5.File(filename, 'r') as f:
#         try:
#             foo = list(f.keys())
#
#             # for i in range(0, shot_number):
#             #     raymx_data = raymx_data + f[foo[i]]
#
#             # raymx_data_avr = raymx_data / shot_number
#             raymx_data_avr = np.array(f[foo[shot_number - 1]])
#             print(raymx_data_avr.shape)
#             im = Image.fromarray(raymx_data_avr)
#
#             im.save(output_filename)
#             f.close()
#             print("make img done : ", out_file_name[:12])
#         except SystemExit:
#             f.close()
#             sys.exit(1)


