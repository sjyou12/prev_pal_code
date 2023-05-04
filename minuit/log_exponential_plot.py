import numpy as np
import os

def exp_decay(tau_arr, random_noise=False):
    exponential_temp = []
    x= np.round(np.linspace(-1, 2000, 20010, endpoint=True), 1)
    for tau in tau_arr:
        temp_exp_arr = []
        for x_val in x:
            if x_val < 0:
                temp_exp_arr.append(0.0)
            else:
                temp_exp_arr.append(1* np.exp(-(x_val/tau)))
        if random_noise:
            temp_exp_arr = temp_exp_arr + np.random.normal(0, 0.05, size=len(temp_exp_arr))
        exponential_temp.append(temp_exp_arr)
    return x, exponential_temp

def save_as_dat(save_arr, save_path, tau_arr):
    try:
        # dat_file = open(save_path + 'exponential_decay_time_point_est' + '.dat', 'w')
        dat_file = open(save_path + 'random_noise_test' + '.dat', 'w')
    except:
        os.makedirs(save_path)
        dat_file = open(save_path + 'exponential_decay_time_point_est' + '.dat', 'w')
    selected_time_arr = [0, 0.1, 0.3, 1, 3.2, 10, 32, 100, 320, 1000]
    temp_selc_time = []
    selected_time_idx = []
    for time_idx, time in enumerate(x):
        if time in selected_time_arr:
            temp_selc_time.append(time)
            selected_time_idx.append(time_idx)
        else:
            continue
    save_arr.append(np.array(selected_time_arr))
    for arr_idx in range(len(tau_arr)):
        temp_selc_data= []
        for time_idx in selected_time_idx:
            temp_selc_data.append(save_arr[arr_idx+1][time_idx])
        save_arr.append(np.array(temp_selc_data))

    first_row_data = ['Time', tau_arr[0], tau_arr[1], tau_arr[2], 'selected delay', tau_arr[0], tau_arr[1], tau_arr[2]]
    for idx in range(len(first_row_data)):
        if idx == len(first_row_data)-1:
            dat_file.write(str(first_row_data[idx]) + "\n")
        else:
            dat_file.write(str(first_row_data[idx]) + "\t")

    for idx in range(len(save_arr[0])):
        for data_idx in range(len(save_arr)):
            if idx < len(selected_time_arr):
                if data_idx == 7:
                    dat_file.write(str(save_arr[data_idx][idx]) + "\n")
                else:
                    dat_file.write(str(save_arr[data_idx][idx]) + "\t")
            else:
                if data_idx == 3:
                    dat_file.write(str(save_arr[data_idx][idx]) + "\n")
                    break
                else:
                    dat_file.write(str(save_arr[data_idx][idx]) + "\t")

tau_arr = [10, 50, 100]
save_path = "./log_exp_test/"

x, exp_data = exp_decay(tau_arr, random_noise=True)
save_arr = []
save_arr.append(x)
for idx in range(len(exp_data)):
    save_arr.append(np.array(exp_data[idx]))
save_as_dat(save_arr, save_path, tau_arr)






