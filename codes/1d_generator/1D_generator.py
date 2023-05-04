from matplotlib import pyplot as plt
from scipy import interpolate
import pyFAI
import h5py as h5
import numpy as np
import os
import multiprocessing


def set_q(info):
    q_from, q_to, q_step_size = info
    custom_q_val = np.arange(q_from, q_to + q_step_size, q_step_size)

    num_of_points = int((q_to - q_from) / q_step_size + 1)

    print("setting q values for interpolation...")
    print("{0} < q < {1}, {2} points".format(q_from, q_to, num_of_points))

    return custom_q_val


class CurveGenerator:
    run_num = None

    will_interp = False
    interp_q_val = None
    raw_q_val = None
    save_q_val = None

    delay_num = None
    img_file_load_path = None
    curve_save_path = None
    bkg_img = None
    mask_img = None
    q_save_path = None

    def __init__(self, run_num, calib_file_path, custom_q_val):
        self.run_num = run_num
        self.AzimuthalIntegrator = pyFAI.load(calib_file_path)
        if len(custom_q_val) != 0:
            self.interp_q_val = custom_q_val
            self.will_interp = True

    def set_IO_path(self, in_path, out_path):
        if I3m_data:
            self.img_file_load_path = in_path + "run6/".format(self.run_num) + "eh2rayMX_img"
        else:
            self.img_file_load_path = in_path + "run_{0:05d}_DIR/".format(self.run_num) + "eh1rayMX_img"
        out_path = out_path + 'run_{0:05d}_DIR'.format(self.run_num)
        if os.path.isdir(out_path):
            pass
        else:
            os.mkdir(out_path)
        self.curve_save_path = out_path
        self.q_save_path = out_path

    def set_bkg_and_mask_path(self):

        bkg_common_path = "/xfel/ffs/dat/scan/"
        # bkg_file_name = "run_{0:05d}_DIR/eh1rayMX_img/00000001_00000300.h5".format(bkg_run_num)
        bkg_file_name = "230427_bkg_1_00005_DIR/eh1rayMX_img/00000001_00000300.h5".format(bkg_run_num)

        mask_path = "/xfel/ffs/dat/ue_230427_FXL/scratch/"
        mask_file_name = "230427_mask4.h5"

        bkg_file = h5.File(bkg_common_path + bkg_file_name, 'r')
        # bkg_file = h5.File(bkg_mask_common_path + bkg_file_name, 'r')
        keys_now_file = list(bkg_file.keys())
        bkg_img_datas = []
        for each_key in keys_now_file:
            bkg_img_datas.append(np.array(bkg_file[each_key]))
        self.bkg_img = np.average(np.array(bkg_img_datas), axis=0)
        # self.bkg_img = bkg_img_datas[0]
        mask_file = h5.File(mask_path + mask_file_name, 'r')
        # mask_file = h5.File(bkg_mask_common_path + mask_file_name, 'r')
        self.mask_img = np.array(mask_file['mask'], dtype='int32')

    def plot_bkg_and_mask(self):
        plt.pcolor(self.bkg_img)
        plt.title("bkg img")
        plt.colorbar()
        plt.clim(5, 15)
        plt.show()

        plt.pcolor(self.mask_img)
        plt.title("mask img")
        plt.colorbar()
        plt.clim(0, 1)
        plt.show()

    def single_run_process(self, test_plot=False):
        print("now running on {:0>5}...\n".format(self.run_num))
        now_run_intensity_files = os.listdir(self.img_file_load_path)
        delay_num = len(now_run_intensity_files)
        delay_arr = range(0, delay_num)
        chunk_len = 1
        chunked_delay_arr = []
        num_chunk = int(np.ceil(delay_num/chunk_len))

        for chunk_idx in range(num_chunk):
            try:
                chunked_delay_arr.append(delay_arr[chunk_idx * chunk_len:(chunk_idx+1)*chunk_len])
            except:
                chunked_delay_arr.append(delay_arr[chunk_idx * chunk_len:])

        # for chunk_idx, chunked_arr in enumerate(chunked_delay_arr):
        #     processes = []
        #     for delay_idx in chunked_arr:
        #         if delay_idx == 0 and chunk_idx == 0:
        #             p = multiprocessing.Process(target=self.single_delay_process, args=(delay_idx, test_plot))
        #             p.start()
        #             processes.append(p)
        #         else:
        #             break
        #         # delay_int_dict = self.single_delay_process(delay_idx, test_plot)
        #         # self.h5_file_out(delay_idx, delay_int_dict)
        #         if delay_idx == 0:
        #             # np.save(self.q_save_path + "/common_q_val.npy", self.save_q_val)
        #             np.savetxt(self.q_save_path + "/common_q_val.dat", self.save_q_val)
        #     for process in processes:
        #         process.join()
        self.single_delay_process(10, test_plot)

    def single_delay_process(self, delay_idx, test_plot=False):
        now_delay_h5_file_name = self.img_file_load_path + "/001_001_{:0>3}".format(delay_idx + 1) + ".h5"
        now_delay_h5_file = h5.File(now_delay_h5_file_name, 'r')
        now_delay_keys = list(now_delay_h5_file.keys())
        total_shot_num = len(now_delay_keys)
        print("now processing on {}...\n".format(now_delay_h5_file_name))
        now_delay_1D_curve_dict = dict()
        for idx, key in enumerate(now_delay_keys):
            img = np.array(now_delay_h5_file[key])
            if not idx:
                self.raw_q_val, raw_intensity = self.AzimuthalIntegrator.integrate1d(img, npt=1024,
                                                                                     polarization_factor=0.996,
                                                                                     mask=self.mask_img,
                                                                                     dark=self.bkg_img,
                                                                                     method="splitpixel", unit="q_A^-1")
                print("q values after masking : {0} < q < {1}".format(self.raw_q_val[0], self.raw_q_val[-1]))
                if self.will_interp:
                    self.save_q_val = self.interp_q_val
                    np.save(self.q_save_path + "/common_q_val.npy", self.save_q_val)
                else:
                    self.save_q_val = self.raw_q_val
                    np.savetxt(self.q_save_path + "/common_q_val.dat", self.save_q_val)

            else:
                _, raw_intensity = self.AzimuthalIntegrator.integrate1d(img, npt=1024, polarization_factor=0.996,
                                                                        mask=self.mask_img, dark=self.bkg_img,
                                                                        method="splitpixel", unit="q_A^-1")

            if self.will_interp:
                save_intensity = self.interpolation(raw_intensity)
            else:
                save_intensity = raw_intensity

            # plt.title("raw vs. interp comparison")
            # plt.plot(self.raw_q_val, raw_intensity, label="raw dat")
            # plt.plot(self.interp_q_val, save_intensity, label="interp dat")
            # plt.legend()
            # plt.show()

            now_delay_1D_curve_dict[key] = save_intensity   # 1D intensity with keys to make h5 file in the same format
            if test_plot:
                if not (idx+1) % (total_shot_num/4):        # test plot
                    print("{0} / {1} processing in {2}-th delay".format(idx+1, total_shot_num, delay_idx+1))

                    plt.pcolor(img)
                    plt.title("raw 2D img")
                    plt.clim(0, 1500)
                    plt.colorbar()
                    plt.show()

                    img = img - self.bkg_img
                    mask_remained_area = np.ones((1440, 1440)) - self.mask_img
                    img = img * mask_remained_area

                    plt.pcolor(img)
                    plt.title("bkg and mask corrected 2D img")
                    plt.clim(0, 1500)
                    plt.colorbar()
                    plt.show()

                    plt.plot(self.save_q_val, save_intensity)
                    plt.title("1D curve from corrected 2D img, " + key)
                    plt.show()
        self.h5_file_out(delay_idx, now_delay_1D_curve_dict)
        # return now_delay_1D_curve_dict

    def h5_file_out(self, delay_idx, delay_int_dict):
        output_file_name = self.curve_save_path + "/" + "001_001_{:0>3}.h5".format(delay_idx + 1)
        h5Fp = h5.File(output_file_name, 'w')
        for key, intensity in delay_int_dict.items():
            h5Fp.create_dataset(key, data=intensity)
        h5Fp.close()
        print("\n" + output_file_name + " save completed\n")

    def interpolation(self, int_before):
        int_func = interpolate.interp1d(self.raw_q_val, int_before)
        int_after = int_func(self.interp_q_val)

        return int_after


run_list = [5]
bkg_run_num = 31
img_loading_path = "/xfel/ffs/dat/scan/"
# img_loading_path = "/data/exp_data/PAL-XFEL_20201217-front/rawData/"
# curve_save_path = "../../pyFAI_calib_1D/"     # should make directories "run_num" in "run_list" in the "curve_save_path"
curve_save_path = "/xfel/ffs/dat/ue_230427_FXL/analysis/"     # should make directories "run_num" in "run_list" in the "curve_save_path"
will_interp = False      # perform interpolation with the given q values. If "will_interp = False", q values will be
                        # saved as "raw_q_val" from the AzimuthalIntegrator.interp1d function with "npt" points.
interp_info = [0.13391795, 4.44822642, 0.005307]   # input list : [q_from, q_to, q_step_size], work in range of [ q_fwrom <= q <= q_to ]
                                    # should put the range values (1st and 2nd arguments in interp_info) in "raw_q_val".

interp_q_val = []
I3m_data = False
if will_interp:
    interp_q_val = set_q(interp_info)
    test_q_val_file = np.load("/data/exp_data/myeong0609/PAL-XFEL_20210514/common_q_val.npy")

for now_run_num in run_list:
    if I3m_data:
        poni_file_path = "/data/exp_data/namkyeongmin/PAL-XFEL_20211030/poni_files/20220505_jet.poni"
    else:
        poni_file_path = "/xfel/ffs/dat/ue_230427_FXL/calibration/230427_bkg_3.poni"
    now_run_process = CurveGenerator(now_run_num, poni_file_path, interp_q_val)
    now_run_process.set_IO_path(img_loading_path, curve_save_path)
    now_run_process.set_bkg_and_mask_path()
    # now_run_process.plot_bkg_and_mask()
    now_run_process.single_run_process(test_plot=False)
