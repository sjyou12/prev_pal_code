import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.interpolate import interp1d

dat_common_path = "/xfel/ffs/dat/ue_230427_FXL/analysis/results/each_run_watersum_int/"
ref_data = np.loadtxt(dat_common_path + "input_Iq.csv", delimiter=',', skiprows=1)[:, 1]
ref_q_val = np.loadtxt(dat_common_path + "input_Iq.csv", delimiter=',', skiprows=1)[:, 0]
bkg = np.loadtxt(dat_common_path + "run_64_static_avg_int.txt")
data = np.loadtxt(dat_common_path + "run_105_static_avg_int.txt")
exp_q_val = np.loadtxt(dat_common_path + "q_val_run90.dat")
#############################################################
x_common = exp_q_val

f1 = interp1d(ref_q_val, ref_data, kind = 'cubic')
interp_ref_data = f1(x_common)
ref_data = interp_ref_data
#############################################################
anal_q_val = (1.5, 11.5)
anal_range = (x_common >= anal_q_val[0]) & (x_common <= anal_q_val[1])
x_common = x_common[anal_range]
ref_data = ref_data[anal_range]
bkg = bkg[anal_range]
data = data[anal_range]
#############################################################
bkg_q_val = (10, 11.5)
bkg_range = (x_common >= bkg_q_val[0]) & (x_common <= bkg_q_val[1])
bkg_cut = bkg[bkg_range].reshape(-1, 1)
data_cut = data[bkg_range]
#############################################################
model1 = LinearRegression().fit(bkg_cut, data_cut)
ratio1 = model1.coef_

bkg_signal = bkg * ratio1
refined_signal = data - bkg_signal
#############################################################
ref_q_val = (1.5, 3)
ref_range = (x_common >= ref_q_val[0]) & (x_common <= ref_q_val[1])
ref_cut = ref_data[ref_range].reshape(-1, 1)
refined_signal_cut = refined_signal[ref_range]
#############################################################
model2 = LinearRegression().fit(ref_cut, refined_signal_cut)
ratio2 = model2.coef_

refined_ref_data = ref_data * ratio2
noise = refined_signal - refined_ref_data
#############################################################
degree = 3
poly_features = PolynomialFeatures(degree=degree)
X_poly = poly_features.fit_transform(np.arange(len(noise)).reshape(-1, 1))
poly_model = LinearRegression().fit(X_poly, noise)
noise_fit = poly_model.predict(X_poly)
#############################################################
fit, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(10, 18), constrained_layout=True)
ax1.plot(x_common, ratio1 * bkg, color = 'blue', label = 'background')
ax1.plot(x_common, data, color = "black", label = 'now_data')
ax1.axvline(x=bkg_q_val[0], color = 'gray', linestyle = '--')
ax1.axvline(x=bkg_q_val[1], color = 'gray', linestyle = '--')
ax1.legend()

ax2.plot(x_common, refined_signal, color = 'blue', label = 'now_data-bkg')
ax2.plot(x_common, ratio2*ref_data, color = 'red', label = 'ref')
ax2.axvline(x=ref_q_val[0], color = 'gray', linestyle = '--')
ax2.axvline(x=ref_q_val[1], color = 'gray', linestyle = '--')
ax2.axhline(y=15e10, color = 'gray')
ax2.legend()

ax3.plot(x_common, np.subtract(data,noise), color = 'black', label = 'now_data-bkg-ref')
ax3.plot(x_common, noise, color = 'blue', label = 'noise')
ax3.plot(x_common, noise_fit, color = 'red', label = 'noise fit')
ax3.axhline(y=37e10, color = 'gray')
ax3.legend()
plt.show()
############################################################


