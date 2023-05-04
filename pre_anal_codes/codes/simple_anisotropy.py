from codes.AnisoAnal import ImgData, AnisoAnal

testAnisoAnalyzer = AnisoAnal()
now_common_path = "/home/common/exp_data/PAL-XFEL_20201217-back/rawData/"
now_mask_file = "/home/common/exp_data/PAL-XFEL_20201217-back/scratch/mask_droplet_1222.h5"
now_beam_center = [723.145, 723.498]  # unit : pixels
now_sample_detector_distance = 90.958  # unit : mm

testAnisoAnalyzer.set_common_env(common_path=now_common_path)
testAnisoAnalyzer.set_mask(now_mask_file, show_mask=False)
testAnisoAnalyzer.set_img_info(now_beam_center[0], now_beam_center[1], now_sample_detector_distance)
testAnisoAnalyzer.read_single_delay_h5_files(run_num=53)
testAnisoAnalyzer.get_diff_img()
testAnisoAnalyzer.aniso_anal_diff_img(result_plot=True)