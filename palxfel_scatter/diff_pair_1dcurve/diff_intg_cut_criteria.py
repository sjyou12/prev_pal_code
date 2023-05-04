import numpy as np
import matplotlib.pyplot as plt
import math

whole_run_before_cutted_diff = np.load("../results/whole_run_diff/whole_run_diff_run53-54.npy", allow_pickle=True)
q_val = np.load("../results/q_val_run53-54.npy", allow_pickle=True)

print(np.shape(whole_run_before_cutted_diff))


def additional_process_diff_test(before_cutted_diff, test_plot_in_one_graph=3):
    num_of_delay_in_one_run = 54
    num_of_run = 2
    num_of_whole_delay = num_of_delay_in_one_run * num_of_run
    intg_start_idx = 93
    intg_end_idx = -180
    cutoff_criteria = [5E4 for _ in range(num_of_whole_delay)]

    whole_run_sum_list_before_cut = []
    whole_run_sum_list_after_cut = []
    for run_idx, each_run_diff_list in enumerate(before_cutted_diff):
        now_run_sum_list = []
        now_run_cutted_sum_list = []
        for delay_idx, each_delay_diff_list in enumerate(each_run_diff_list):
            each_delay_diff_list = np.array(each_delay_diff_list)
            now_delay_sum_list = []
            for diff_data in each_delay_diff_list:
                now_sum = np.sum(np.abs(diff_data[intg_start_idx:intg_end_idx]))
                now_delay_sum_list.append(now_sum)
            now_delay_sum_list = np.array(now_delay_sum_list)
            now_run_sum_list.append(now_delay_sum_list)

            now_delay_cutted_diff = each_delay_diff_list[now_delay_sum_list < cutoff_criteria[delay_idx]]

            # print("remove ", len(each_delay_diff_list[now_delay_sum_list >= cutoff_criteria[delay_idx]]),
            #       " in", delay_idx, "-th delay")

            now_delay_sum_list_after_cut = []
            for diff_data in now_delay_cutted_diff:
                now_sum = np.sum(np.abs(diff_data[intg_start_idx:intg_end_idx]))
                now_delay_sum_list_after_cut.append(now_sum)
            now_delay_sum_list_after_cut = np.array(now_delay_sum_list_after_cut)
            now_run_cutted_sum_list.append(now_delay_sum_list_after_cut)

        whole_run_sum_list_before_cut.extend(now_run_sum_list)
        whole_run_sum_list_after_cut.extend(now_run_cutted_sum_list)


    total_graph_num = math.ceil(num_of_whole_delay / test_plot_in_one_graph)
    print(total_graph_num)

    for graph_idx in range(total_graph_num):
        start_delay_idx = graph_idx * test_plot_in_one_graph
        plt.figure(figsize=(10, 12))
        plt.suptitle("diff integration compare graph before/after")

        for row_idx in range(test_plot_in_one_graph):
            now_delay_idx = start_delay_idx + row_idx
            if now_delay_idx > (num_of_whole_delay - 1):
                print("end of draw")
                break
            in_run_delay_idx = now_delay_idx % num_of_delay_in_one_run
            run_idx = now_delay_idx // num_of_delay_in_one_run
            plt.subplot(test_plot_in_one_graph, 2, 2*row_idx + 1)
            plt.hist(whole_run_sum_list_before_cut[now_delay_idx], bins=200)
            plt.title(str(in_run_delay_idx) + "-th delay of " + str(run_idx) + "-th run")
            plt.xlabel("integration value")
            plt.ylabel("frequency")
            plt.axvline(x=cutoff_criteria[in_run_delay_idx], color='r')

            plt.subplot(test_plot_in_one_graph, 2, 2*(row_idx + 1))
            plt.hist(whole_run_sum_list_after_cut[now_delay_idx], bins=200)
            plt.title(str(in_run_delay_idx) + "-th delay of " + str(run_idx) + "-th run")
            plt.xlabel("integration value")
            plt.ylabel("frequency")
            plt.axvline(x=cutoff_criteria[in_run_delay_idx], color='r')
        plt.show()

        if (graph_idx % 5) == 0:
            print("plot {0} of {1}".format(graph_idx + 1, total_graph_num))

additional_process_diff_test(whole_run_before_cutted_diff)
