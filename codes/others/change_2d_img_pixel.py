import numpy as np
import os
import h5py as h5
import matplotlib.pyplot as plt
import multiprocessing

class reducing_pixel:

    def __init__(self, run_num, img_load_path, img_save_path, target_horiz_num, target_verti_num):
        self.run_num = run_num
        self.img_load_path = img_load_path
        self.img_save_path = img_save_path
        self.target_horizontal_num = target_horiz_num
        self.target_vertical_num = target_verti_num
        self.now_run_files = []

        if not os.path.isdir(self.img_save_path):
            os.makedirs(self.img_save_path)
        else:
            pass

    def make_reduced_img(self):
        self.now_run_files = os.listdir(self.img_load_path)
        now_delay_num = len(self.now_run_files)
        now_delays = range(0, now_delay_num)

        chunk_len = 5
        chunked_delay_arr = []
        num_chunk = int(np.ceil(now_delay_num / chunk_len))

        for chunk_idx in range(num_chunk):
            try:
                chunked_delay_arr.append(now_delays[chunk_idx * chunk_len:(chunk_idx + 1) * chunk_len])
            except:
                chunked_delay_arr.append(now_delays[chunk_idx * chunk_len:])
        for chunk_idx, chunked_arr in enumerate(chunked_delay_arr):
            processes = []
            for delay_idx in chunked_arr:
                p = multiprocessing.Process(target=self.process_each_delay, args=(delay_idx,))
                p.start()
                processes.append(p)
            for process in processes:
                process.join()

    def process_each_delay(self, delay_idx):
        save_dict = {}

        now_file = self.now_run_files[delay_idx]
        now_h5_file = h5.File(self.img_load_path + now_file, "r")
        now_file_keys = list(now_h5_file.keys())
        for key_idx, now_key in enumerate(now_file_keys):
            now_img = np.array(now_h5_file[now_key])
            new_pixel_img = now_img[0:720][:]
            save_dict[now_key] = new_pixel_img
            if key_idx == 0:
                plt.title("2d img with reduced pixel number")
                plt.pcolor(new_pixel_img)
                plt.colorbar()
                plt.xlabel("x pixel")
                plt.ylabel("y pixel")
                plt.show()
        self.h5_file_out(delay_idx, save_dict)

    def h5_file_out(self, delay_idx, save_dict):
        out_file_name = self.img_save_path + "001_001_{:0>3}.h5".format(delay_idx+1)
        h5Fp = h5.File(out_file_name, "w")
        for key, intensity in save_dict.items():
            h5Fp.create_dataset(key, data=intensity)
        h5Fp.close()
        print("\n" + out_file_name + " save completed\n")



run_num = 1
img_common_path = "/data/exp_data/PAL-XFEL_20210514/rawdata/"
img_load_path = img_common_path + "run{0:04d}_00001_DIR/eh1rayMX_img/".format(run_num)
# img_load_path = img_common_path + "210516_powder_1_00006_DIR/eh1rayMX_img/"
img_save_path = "/data/exp_data/myeong0609/PAL-XFEL_20210514/new_pixel_img/run{0:04d}_00001_DIR/eh1rayMX_img/".format(run_num)
# img_save_path = "/data/exp_data/myeong0609/PAL-XFEL_20210514/new_pixel_img/210516_powder_1_00006_DIR/eh1rayMX_img/".format(run_num)

target_horizontal_num = 1440
target_vertical_num = 720

target_img_info = reducing_pixel(run_num, img_load_path, img_save_path, target_horizontal_num, target_vertical_num)
target_img_info.make_reduced_img()
