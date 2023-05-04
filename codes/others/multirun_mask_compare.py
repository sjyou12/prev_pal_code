from MaskedImgPlot import MaskedImgPlot

# data_dir_path = "/data/exp_data/PAL-XFEL_20211030/scan/"
# run_name = "211105_00067"
# # mask_dir_path = "/data/exp_data/PAL-XFEL_20211030/analysis/mask/"
# # mask_name = "211105_00067_mask.h5.tif"
# mask_dir_path = "/data/exp_data/PAL-XFEL_20211030/scratch/"
# mask_name = "211105_mask_00007.h5"
# bkg_dir_path = "/data/exp_data/PAL-XFEL_20211030/scratch/"
# bkg_name = "211104_bkg_00096.tiff"

common_path = "/data/exp_data"

data_dir_path = common_path + "/i3_raw_data/"
run_name = "211108_00013"
mask_dir_path = common_path + "/PAL-XFEL_20211030/scratch/"
mask_name = "211108_mask_00002.h5"
bkg_dir_path = common_path + "/PAL-XFEL_20211030/scratch/"
bkg_name = "211108_bkg_00001.tiff"


nowImgPlot = MaskedImgPlot()
nowImgPlot.set_file_path(data_dir_path, mask_dir_path, bkg_dir_path)
nowImgPlot.load_h5_img_file(run_name, mask_name, bkg_name)
nowImgPlot.set_azi_integrator()
nowImgPlot.cmp_intg_result()
