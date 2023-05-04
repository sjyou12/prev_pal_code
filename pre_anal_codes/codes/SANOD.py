from codes.old_SVDCalc import SVDCalc
import numpy as np
import matplotlib.pyplot as plt
import copy

time_delay = [-3, -2, -1.5, -1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8, 2, 2.5, 3, 4, 5, 7, 10, 13, 18, 24, 32, 42, 56, 75, 100, 130, 180, 240, 320, 420, 560, 750, 1e3, 1.3e3, 1.8e3, 2.4e3]


def mgs(X, verbose=False):
    n, p = np.shape(X)
    if p > n:
        print("need to transpose input matrix")
        return
    else:
        q, r = np.linalg.qr(X)
        if verbose:
            print("q is ", q)
            print("r is ", r)
        return q, r


q_val_cut = np.load('../results/signal_per_delay/cut_q_range.npy')
print("q cut", np.shape(q_val_cut), q_val_cut[:5], q_val_cut[-5:])

num_of_neg_delay = 6  # use estimated real time zero
signal_per_delay = np.load('../results/signal_per_delay/run53.npy')
neg_delay_signal = signal_per_delay[:num_of_neg_delay]
pos_delay_signal = signal_per_delay[num_of_neg_delay:]
neg_tp = np.transpose(neg_delay_signal)
signal_tp = np.transpose(signal_per_delay)

neg_svd = SVDCalc(neg_tp)
whole_svd = SVDCalc(signal_tp)

neg_sing_val = neg_svd.calc_svd()
pos_sing_val = whole_svd.calc_svd()


def plot_singular_value(SVD_obj, plot_title):
    print(SVD_obj.singValVec)
    data_y = SVD_obj.singValVec
    data_x = range(1, len(data_y) + 1)

    plt.title(plot_title)
    color_r = 'tab:red'
    plt.xlabel("index of singular value")
    plt.ylabel("singular value", color=color_r)
    plt.scatter(data_x, data_y, color=color_r)
    plt.plot(data_x, data_y, color=color_r)
    plt.show()


plot_singular_value(neg_svd, "negative")
plot_singular_value(whole_svd, "whole")

num_neg_comp = 2
num_whole_comp = 3
neg_svd.pick_meaningful_data(num_neg_comp)
whole_svd.pick_meaningful_data(num_whole_comp)

neg_svd.plot_left_Vec()
neg_svd.plot_right_Vec()
whole_svd.plot_left_Vec()
whole_svd.plot_right_Vec()
outFp1 = open("../results/svd/run53_whole_LSV.dat", 'w')
outFp2 = open("../results/svd/run53_whole_RSV.dat", 'w')
whole_svd.file_output_singular_vectors(outFp1, outFp2)
outFp1.close()
outFp2.close()

def rsv_time_plot(rsv_data, delay_values):
    rsv_num = rsv_data.shape[1]
    time_delay_points = rsv_data.shape[0]
    plot_boundary = 31

    if time_delay_points < plot_boundary:
        print("two short rsv")

    str_delay = []
    for each_time_val in delay_values:
        str_delay.append(str(each_time_val))

    plt.figure(figsize=(10, 6))
    transRight = np.transpose(rsv_data)
    for data_idx in range(rsv_num):
        plt.plot(delay_values[:plot_boundary], transRight[data_idx][:plot_boundary], label=("rightVec" + str(data_idx + 1)))
    # if v_line_1 != 0.0:
    #     plt.axvline(x=v_line_1, color='r')
    # if v_line_2 != 0.0:
    #     plt.axvline(x=v_line_2, color='r')
    plt.xlabel("ps (time)")
    plt.xscale('symlog', linthresh=1, linscale=1)
    plt.xticks(ticks=delay_values[:plot_boundary], labels=str_delay[:plot_boundary])
    plt.grid(True)
    plt.title("Right singular vectors  - time ")
    plt.legend()
    plt.show()


rsv_time_plot(whole_svd.meanRightVec, time_delay)
nonortho_components = np.concatenate((neg_svd.meanLeftVec, whole_svd.meanLeftVec), axis=1)
print(nonortho_components.shape)


def plot_comps(comp_data, graph_title, labels, is_separate=True):
    global q_val_cut
    num_comp = comp_data.shape[1]
    if is_separate:
        plt.figure(figsize=(6, 10))
        y_max = np.max(comp_data)
        y_min = np.min(comp_data)
        margin = (y_max - y_min) / 20
        for plot_idx in range(num_comp):
            plt.subplot(num_comp, 1, plot_idx + 1)
            if plot_idx == 0:
                plt.title(graph_title)
            plt.plot(q_val_cut, comp_data[:, plot_idx], label=labels[plot_idx])
            plt.axhline(y=0, color='gray', ls='--', alpha=0.6)
            plt.ylim((y_min - margin, y_max + margin))
            plt.legend()
        plt.xlabel("q")
        plt.show()

    else:
        for o_idx in range(num_comp):
            plt.plot(q_val_cut, comp_data[:, o_idx], label=labels[o_idx])

        plt.xlabel("q")
        plt.title(graph_title)
        plt.legend()
        plt.show()


num_nonortho_comp = nonortho_components.shape[1]
c_label = ["C_" + str(idx + 1) for idx in range(num_nonortho_comp)]
print(c_label)
plot_comps(nonortho_components, graph_title="spectral components", labels=c_label)

q_orthonorm, r_uptri = mgs(nonortho_components)
print("q shape", q_orthonorm.shape)
print("r shape", r_uptri.shape)

num_orthonorm_comp = q_orthonorm.shape[1]
o_label = ["O_" + str(idx + 1) for idx in range(num_orthonorm_comp)]
print(o_label)
plot_comps(q_orthonorm, graph_title="orthonormalized components", labels=o_label)

print(signal_per_delay.shape)
num_t_delay = signal_per_delay.shape[0]

alpha_weight = np.zeros((num_t_delay, num_nonortho_comp))
signal_tp = np.transpose(signal_per_delay)
for delay_idx in range(num_t_delay):
    for comp_idx in range(num_nonortho_comp):
        temp = np.dot(signal_tp[:, delay_idx], q_orthonorm[:, comp_idx]) / r_uptri[comp_idx][comp_idx]
        # print(temp)
        alpha_weight[delay_idx][comp_idx] = temp

print(alpha_weight)



def plot_chronogram(alpha_data, time_delay_ps, graph_title, labels, is_separate=True):
    # x_ticks = [-5, 0, 5, 10, 20, 50, 100, 1000]
    x_ticks = [-5, -1, 0, 1, 10, 100, 1000]
    str_ticks = []
    for each_time_val in x_ticks:
        str_ticks.append(str(each_time_val))

    num_comp = alpha_data.shape[1]
    if is_separate:
        plt.figure(figsize=(6, 10))
        y_max = np.max(alpha_data)
        y_min = np.min(alpha_data)
        margin = (y_max - y_min) / 20
        for plot_idx in range(num_comp):
            plt.subplot(num_comp, 1, plot_idx + 1)
            if plot_idx == 0:
                plt.title(graph_title)
            plt.plot(time_delay_ps, alpha_data[:, plot_idx], '.-', label=labels[plot_idx])
            plt.axhline(y=0, color='gray', ls='-', alpha=0.6)
            plt.ylim((y_min - margin, y_max + margin))
            plt.xlim(-5, 5e3)
            plt.xscale('symlog', linthresh=1, linscale=1.5)
            plt.xticks(ticks=x_ticks, labels=str_ticks)
            plt.grid(True)
            plt.legend()
        plt.xlabel("time (ps)")
        plt.show()

    else:
        for o_idx in range(num_comp):
            plt.plot(time_delay_ps, alpha_data[:, o_idx], label=labels[o_idx])

        plt.xlabel("time (ps)")
        plt.title(graph_title)
        plt.xlim(-5, 5e3)
        plt.xscale('symlog', linthresh=1, linscale=1.5)
        plt.xticks(ticks=x_ticks, labels=str_ticks)
        plt.grid(True)
        plt.legend()
        plt.show()

alpha_label = ["a_" + str(idx + 1) for idx in range(num_orthonorm_comp)]
print(alpha_label)
plot_chronogram(alpha_weight, time_delay_ps=time_delay, graph_title="chronogram (alpha for C_N)", labels=alpha_label, is_separate=True)

def artifact_filtering(before_data, num_time_delay, components, weights, num_arti):
    reconstructed = []
    for delay_idx in range(num_time_delay):
        delay_data = copy.deepcopy(before_data[delay_idx])
        # remove artifact from original data
        for arti_idx in range(num_arti):
            delay_data -= components[:,arti_idx] * weights[delay_idx][arti_idx]
        reconstructed.append(delay_data)

    return np.array(reconstructed)

filtered_data = artifact_filtering(signal_per_delay, num_t_delay, nonortho_components, alpha_weight, num_neg_comp)

# print(signal_per_delay - np.array(filtered_data))

def compare_filter_plot(before_data, after_data, num_time_delay, graph_per_fig, graph_title, time_delay_ps):
    global q_val_cut
    y_max = np.max([before_data, after_data])
    y_min = np.min([before_data, after_data])
    margin = (y_max - y_min) / 20

    for delay_idx in range(num_time_delay):
        if (delay_idx % graph_per_fig) == 0:
            plt.figure(figsize=(6, 18))
        plt.subplot(graph_per_fig, 1, (delay_idx % graph_per_fig) + 1)
        if (delay_idx % graph_per_fig) == 0:
            plt.title(graph_title)
        plt.plot()
        plt.plot(q_val_cut, before_data[delay_idx], label=str(time_delay_ps[delay_idx]) + "ps", color='k')
        plt.plot(q_val_cut, after_data[delay_idx], label="filtered " + str(time_delay_ps[delay_idx]) + "ps", color='r')
        plt.axhline(y=0, color='gray', ls='--', alpha=0.6)
        plt.ylim((y_min - margin, y_max + margin))
        plt.legend()
        if ((delay_idx + 1) % graph_per_fig) == 0:
            plt.xlabel("q")
            plt.show()
    if (num_time_delay % graph_per_fig) != 0:
        plt.xlabel("q")
        plt.show()


compare_filter_plot(signal_per_delay, filtered_data, num_t_delay, 10, graph_title="filter compare", time_delay_ps=time_delay)

np.save('../results/sanod/filtered_run53.npy', np.transpose(filtered_data))