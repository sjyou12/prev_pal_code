import numpy as np
import matplotlib.pyplot as plt
import configparser
import math
import ast

svd_start_idx = 93
svd_end_idx = -180

def find_neareast_pair(compare_arr, item):
    temp_arr = np.asarray(compare_arr)
    pair_idx = (np.abs(temp_arr - item)).argmin()
    return compare_arr[pair_idx]

def match_water_near_intensity_pair(data_list, laser_on_water_idx, laser_off_water_idx):
    compare_longer_one = None
    compare_shoter_one = None
    if len(laser_on_water_idx) > len(laser_off_water_idx):
        compare_longer_one = laser_on_water_idx
        compare_shoter_one = laser_off_water_idx
    else:
        compare_longer_one = laser_off_water_idx
        compare_shoter_one = laser_on_water_idx

    cmp_long_norm_sum = []
    cmp_short_norm_sum = []
    # for each_idx in compare_longer_one:
    #     cmp_long_norm_sum.append(data_list[each_idx].norm_range_sum)
    # for each_idx in compare_shoter_one:
    #     cmp_short_norm_sum.append(data_list[each_idx].norm_range_sum)
    for each_idx in compare_longer_one:
        cmp_long_norm_sum.append(data_list[each_idx].pair_range_sum)
    for each_idx in compare_shoter_one:
        cmp_short_norm_sum.append(data_list[each_idx].pair_range_sum)

    neareast_int_pair = []
    for int_idx, each_int_sum in enumerate(cmp_long_norm_sum):
        most_similar_sum = find_neareast_pair(cmp_short_norm_sum, each_int_sum)
        sim_sum_idx = cmp_short_norm_sum.index(most_similar_sum)
        # print(each_int_sum, "(idx:", compare_longer_one[int_idx], ")",
        #       most_similar_sum, "(idx:", compare_shoter_one[sim_sum_idx], ")")
        neareast_int_pair.append((compare_longer_one[int_idx], compare_shoter_one[sim_sum_idx]))

    # print("pairing with nearest integrated intensity logic")
    return neareast_int_pair


def plot_sum_for_criteria(data, graph_title, v_line_1=0.0, v_line_2=0.0, v_line_3=0.0):
    """
    plot ice sum / water sum tendency for decide criteria of water/ice data separation
    """
    plt.hist(data, bins=200, log=True)
    # plt.hist(data, bins=200)
    plt.title(graph_title)
    plt.xlabel("integration value")
    plt.ylabel("frequency")
    if v_line_1 != 0.0:
        plt.axvline(x=v_line_1, color='r')
    if v_line_2 != 0.0:
        plt.axvline(x=v_line_2, color='r')
    if v_line_3 != 0.0:
        plt.axvline(x=v_line_3, color='g')
    plt.show()
