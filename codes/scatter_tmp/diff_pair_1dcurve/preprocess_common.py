from palxfel_scatter.diff_pair_1dcurve.MultiRunProc import MultiRunProc
import math

## process similar condition data simultaneously
## use sqlite3 database about each run

nowDataFromDB = [(53, 200000, 300000, 1000000), (54, 200000, 300000, 1000000)]
# common constant for calculating q values
XrayEnergy = 14  # keV unit
XrayWavelength = 12.3984 / XrayEnergy  # Angstrom unit (10^-10 m)
QCoefficient = 4 * math.pi / XrayWavelength
NormFactor = 100000  # Normalization factor (sum of all integration)
FileCommonRoot = "/home/common/exp_data/PAL-XFEL_20201217-back/rawData/"

dataPreProcessor = MultiRunProc(nowDataFromDB)
dataPreProcessor.common_variables(file_common_root=FileCommonRoot)
dataPreProcessor.set_file_name_and_read_tth()
## get graph for decide criteria
# dataPreProcessor.plot_water_sum_dist(each_run_plot=True)
# save intensity files
dataPreProcessor.read_intensity_only_water()
dataPreProcessor.pairwise_diff_calc()
dataPreProcessor.additional_process_diff()

print("here")
