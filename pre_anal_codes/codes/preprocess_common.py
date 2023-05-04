import sqlite3
from codes.DataClasses import ReadOneDataSet
from codes.MultiRunProc import MultiRunProc
import math

## process similar condition data simultaneously
## use sqlite3 database about each run

runvardb = sqlite3.connect('each-run-var.db')
cur = runvardb.cursor()

needSetDB = False
readAllDB = True
nowDataFromDB = None

# common constant for calculating q values
XrayEnergy = 14  # keV unit
XrayWavelength = 12.3984 / XrayEnergy  # Angstrom unit (10^-10 m)
QCoefficient = 4 * math.pi / XrayWavelength
NormFactor = 100000  # Normalization factor (sum of all integration)
FileCommonRoot = "/home/common/exp_data/PAL-XFEL_20201217-back/rawData/"

# codes for make table
if needSetDB:
    cur.execute('CREATE TABLE IF NOT EXISTS process_var(runid INTEGER, watermin REAL, watermax REAL, outlier REAL)')
    # cur.execute('ALTER TABLE process_var ADD COLUMN outlier REAL')
    cur.executemany('INSERT INTO process_var VALUES (?, ?, ?, ?)',
                    [(53, 200000, 300000, 1000000), (54, 200000, 300000, 1000000)])

if readAllDB:
    cur.execute("SELECT * FROM process_var")
    dataRows = cur.fetchall()
    for eachDataRow in dataRows:
        print(eachDataRow)
    nowDataFromDB = dataRows

runvardb.commit()
runvardb.close()

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
