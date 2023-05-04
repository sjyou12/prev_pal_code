from palxfel_scatter.anisotropy.AnisoSVD import AnisoSVD

anisoSVDanalyzer = AnisoSVD()
anisoSVDanalyzer.read_aniso_results()
# anisoSVDanalyzer.plot_iso_signal()
anisoSVDanalyzer.plot_aniso_signal()

# anisoSVDanalyzer.aniso_svd()
# anisoSVDanalyzer.data_svd(anisoSVDanalyzer.all_delay_aniso, "aniso", singular_cut_num=4)
# anisoSVDanalyzer.data_svd(anisoSVDanalyzer.all_delay_iso, "iso", singular_cut_num=4)


