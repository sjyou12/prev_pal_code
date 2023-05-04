import os
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import tifffile as tf
import pyFAI.azimuthalIntegrator
import pyFAI.detectors
import math



class MaskedImgPlot:
    def __init__(self):
        self.data_file_dir = None
        self.mask_file_dir = None
        self.bkg_file_dir = None
        self.now_run_avg_img = []
        self.now_run_avg_beamline_int = []
        self.beamline_q = []
        self.mask_data = []
        self.bkg_data = []
        self.mask_name = []
        self.bkg_name = []
        self.now_run_name = None

        self.xray_energy = 0

    def set_file_path(self, data_path, mask_path, bkg_path):
        self.data_file_dir = data_path
        self.mask_file_dir = mask_path
        self.bkg_file_dir = bkg_path

    def load_h5_img_file(self, run_name, mask_name, bkg_name):
        self.now_run_name = run_name
        now_run_img_dir = f"{self.data_file_dir}{self.now_run_name}_DIR/eh2rayMX_img/"
        now_run_int_dir = f"{self.data_file_dir}{self.now_run_name}_DIR/eh2rayMXAI_int/"
        now_run_tth_file = f"{self.data_file_dir}{self.now_run_name}_DIR/eh2rayMXAI_tth/00000001_00000300.h5"

        # jet_data_beam_center_change_check.py mask file
        now_mask_file_name = self.mask_file_dir + mask_name
        self.mask_name = mask_name

        if mask_name[-3:] == ".h5":
            mask_file_obj = h5.File(now_mask_file_name, 'r')
            self.mask_data = np.array(mask_file_obj['mask'], dtype='int32')
        elif mask_name[-4:] == ".tif":
            self.mask_data = np.array(tf.imread(now_mask_file_name), dtype='int32')
        else:
            print(f"wrong mask file name : {now_mask_file_name}")

        now_bkg_file_name = self.bkg_file_dir + bkg_name
        self.bkg_name = bkg_name

        bkg_file_obj = h5.File(now_bkg_file_name, 'r')
        self.bkg_data = np.array(bkg_file_obj['bkg'], dtype='int32')

        # now_bkg_file = h5.File(now_bkg_file_name, 'r')
        # keys_bkg_file = list(now_bkg_file.keys())
        #
        # bkg_img_data = []
        # for each_key in keys_bkg_file:
        #     bkg_img_data.append(np.array(now_bkg_file[each_key]))
        #
        # self.bkg_data = np.average(np.array(bkg_img_data), axis=0)

        self.now_run_avg_img = self.avg_dir_file_data(now_run_img_dir)
        self.now_run_avg_beamline_int = self.avg_dir_file_data(now_run_int_dir)

        self.test_plot_2d_img(self.now_run_avg_img, self.mask_data, self.bkg_data)

    def avg_dir_file_data(self, run_data_file_dir):
        file_name_list = os.listdir(run_data_file_dir)
        print(run_data_file_dir)

        file_name_list.sort()

        avg_data_list = []
        file_data_num_list = []

        for each_file_name in file_name_list:
            now_data_file_name = run_data_file_dir + each_file_name
            now_data_file_obj = h5.File(now_data_file_name, 'r')
            now_data_file_keys = list(now_data_file_obj.keys())
            file_data_list = []
            for each_key in now_data_file_keys:
                now_data = np.array(now_data_file_obj[each_key], dtype='int32')
                file_data_list.append(now_data)
            file_data_list = np.array(file_data_list)
            now_file_data_avg = np.average(file_data_list, axis=0)
            now_weight = len(now_data_file_keys)
            if now_weight != 300:
                print(f"not 300 data file] {each_file_name} weight : {now_weight}")
            avg_data_list.append(now_file_data_avg)
            file_data_num_list.append(now_weight)
            # print(f"average shape : {now_file_data_avg.shape}")

            now_data_file_obj.close()
            now_data_file_keys = []
            file_data_list = []

        avg_data_list = np.array(avg_data_list)
        file_data_num_list = np.array(file_data_num_list)
        whole_run_avg = np.average(avg_data_list, weights=file_data_num_list, axis=0)
        print(f"each file avg weight : {file_data_num_list}")
        print(f"all file average shape : {whole_run_avg.shape}")

        file_data_num_list = []
        avg_data_list = []

        return whole_run_avg

    def test_plot_2d_img(self, raw_img, mask_img, bkg_img):
        ma_mask = np.ma.masked_equal(mask_img, 0)
        plt.title("background removed with mask (red)")
        bkg_subtract = raw_img - bkg_img
        plt.pcolormesh(bkg_subtract, cmap='viridis')
        cMap = colors.ListedColormap(['r'])
        plt.pcolormesh(ma_mask, cmap=cMap)
        # plt.scatter(707.498, 720.106, c='r', s=1)
        plt.colorbar()
        plt.show()

    def set_azi_integrator(self):
        # TODO : set integrator parameter
        pixel_size = 234e-6  # unit of m
        detector_num_pixel = 960

        # beam_center_x = 707.498  # unit : pixels
        # beam_center_y = 720.106  # unit : pixels
        # sample_detector_dist = 98.358  # unit : mm

        # beam center change jet_data_beam_center_change_check.py
        beam_center_x = 469.430  # unit : pixels
        beam_center_y = 483.346  # unit : pixels
        # beam_center_x = detector_num_pixel - beam_center_x
        sample_detector_dist = 99.069  # unit : mm
        xray_energy = 13.9975  # keV unit
        self.xray_energy = xray_energy

        sd_dist_in_m = sample_detector_dist * 1e-3  # unit of m
        beam_center_x_in_m = beam_center_x * pixel_size  # unit of m
        beam_center_y_in_m = beam_center_y * pixel_size  # unit of m
        xray_wavelength = 12.3984 / xray_energy  # Angstrom unit (10^-10 m)
        xray_wavelength_in_m = xray_wavelength * 1e-10  # unit of m

        # now_detector = pyFAI.detectors.Detector(pixel1=pixel_size, pixel2=pixel_size, max_shape=(detector_num_pixel, detector_num_pixel))
        now_detector = pyFAI.detectors.RayonixMx225hs(pixel1=pixel_size, pixel2=pixel_size)

        self.ai = pyFAI.azimuthalIntegrator.AzimuthalIntegrator(dist=sd_dist_in_m, poni1=beam_center_y_in_m,
                                                                poni2=beam_center_x_in_m,
                                                                wavelength=xray_wavelength_in_m, detector=now_detector)

    def cmp_intg_result(self):
        img_intg_q, img_intg_Iq = self.ai.integrate1d(data=self.now_run_avg_img, npt=1024, mask=self.mask_data,
                                                      unit="q_A^-1", polarization_factor=0.996, dark=self.bkg_data)

        self.test_plot_1d_azi_avg(img_intg_q, img_intg_Iq)

    def test_plot_1d_azi_avg(self, my_q, my_Iq):
        cmp_int_data = self.now_run_avg_beamline_int
        cmp_tth_file_name = f"{self.data_file_dir}{self.now_run_name}_DIR/eh2rayMXAI_tth/001_001_001.h5"
        tth_file_obj = h5.File(cmp_tth_file_name, 'r')
        tth_key = list(tth_file_obj.keys())[0]
        tth_to_q_cvt = Tth2qConvert(self.xray_energy)
        beamline_tth = np.array(tth_file_obj[tth_key])
        self.beamline_q = tth_to_q_cvt.tth_to_q(beamline_tth)

        plt.plot(my_q, my_Iq, label="my avg result")
        plt.plot(self.beamline_q, cmp_int_data, label="beamline int data")
        plt.xlabel(r'q ($A^{-1})$')
        plt.title(f"mask={self.mask_name}\nbkg={self.bkg_name}")
        plt.legend()
        plt.show()

        diff_tth = self.beamline_q - my_q
        plt.plot(self.beamline_q, diff_tth)
        plt.title("q difference")
        plt.show()

        diff_int = cmp_int_data - my_Iq
        plt.plot(self.beamline_q, diff_int)
        plt.title("int difference (beamline - my)")
        plt.show()


class Tth2qConvert:
    def __init__(self, xray_energy):
        self.XrayEnergy = xray_energy  # keV unit
        XrayWavelength = 12.3984 / self.XrayEnergy  # Angstrom unit (10^-10 m)
        self.QCoefficient = 4 * math.pi / XrayWavelength

    def tth_to_q(self, tth_arr):
        """
        convert 2theta value to q
        :param tth_arr: 2theta(degree unit) array
        :return: q value array
        """
        output_q = []
        for each_tth in tth_arr:
            now_q = self.QCoefficient * math.sin(math.radians(each_tth / 2))
            output_q.append(now_q)
        return output_q
