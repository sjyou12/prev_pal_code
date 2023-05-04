from codes.DataClasses import tth_to_q, IceType, WaterType, find_neareast_pair
from codes.PulseData import PulseData
from codes.old_SVDCalc import SVDCalc
from scipy import stats
import re
import matplotlib.pyplot as plt
import configparser
import numpy as np

# Possible RunCondition : SingleOnOff, TempDiffTest, IceFilterCheck
RunCondition = "TempDiffTest"
# RunCondition = "SingleOnOff"

# common constant
CheckFrontFileNum = 10000
TotalCheckFileNum = 0
openFpList = []
FileInfoDictList = []
ChiFileFolderCommonRoot = "../resources/2017_ref_data/"
if RunCondition == "TempDiffTest":
    FileInfoDictList.append({'FileFolder': "run132/chi/", 'FileStartIdx': 1000001,
                             'NumOfFiles': 9999, 'OpenOption': "laser_on"})
    FileInfoDictList.append({'FileFolder': "run134/chi/", 'FileStartIdx': 1000001,
                             'NumOfFiles': 10000, 'OpenOption': "laser_off"})
elif RunCondition == "SingleOnOff":
    FileInfoDictList.append({'FileFolder': "run132/chi/", 'FileStartIdx': 1000001,
                             'NumOfFiles': 9999, 'OpenOption': "self_onoff"})
    # FileInfoDictList.append({'FileFolder': "run134/chi/", 'FileStartIdx': 1000001,
    #                          'NumOfFiles': 10000, 'OpenOption': "self_onoff"})

FileOutOption = {'IntPairDiffAvg': True, 'IntPairDiffSE': True, 'OnWholeAvg': True, 'OffWholeAvg': True,
                 'WholeDiff': True, 'SVDresults': True}
FileOutFolder = '../results/pre_code/'
common_tth_val = []
data_int_val_list = []  # list of PulseData format data object

# read constant from config file
config = configparser.ConfigParser()
config.read('anal_cond.cfg')

LowerBoundOfIce = float(config['DEFAULT']['LowerBoundOfIce'])
UpperBoundOfNotIce = float(config['DEFAULT']['UpperBoundOfNotIce'])
LowerBoundOfWater = float(config['DEFAULT']['LowerBoundOfWater'])
UpperBoundOfNotWater = float(config['DEFAULT']['UpperBoundOfNotWater'])


def chi_file_to_DataSet(chi_file):
    global common_tth_val
    now_int_val = []
    front_ignore_lines = 4
    num_of_total_line = 494
    for i in range(front_ignore_lines):
        chi_file.readline()
    for idx in range(num_of_total_line):
        nowStr = chi_file.readline()
        pre_sp_ln = nowStr.strip()
        sp_input = re.findall(r"[\w'-.]+", pre_sp_ln)
        now_tth = float(sp_input[0])
        now_int = float(sp_input[1])
        if len(common_tth_val) < num_of_total_line:
            # for first file read, fill tth value arr
            common_tth_val.append(now_tth)
        else:
            # other case, check tth val is correct
            try:
                tth_diff = abs(now_tth - common_tth_val[idx])
                if tth_diff != 0:
                    # if tth_diff > 0.00001:
                    raise ValueError
            except ValueError:
                print("two theta value is different in " + str(i + 1 + front_ignore_lines) +
                      "th line of", str(chi_file.name))
                print("now tth is ", now_tth, "1st files value is ", common_tth_val[idx])
        now_int_val.append(now_int)
    now_data = PulseData(now_int_val)
    return now_data


for each_info in FileInfoDictList:
    NowCheckFileNum = CheckFrontFileNum
    if NowCheckFileNum > each_info['NumOfFiles']:
        NowCheckFileNum = each_info['NumOfFiles']
    TotalCheckFileNum += NowCheckFileNum
    # FileNameRange = range(FileStartIdx, FileStartIdx + NumOfFiles)
    FileNameRange = range(each_info['FileStartIdx'], each_info['FileStartIdx'] + each_info['NumOfFiles'])
    ChiFileCommonRoot = ChiFileFolderCommonRoot + each_info['FileFolder']
    print("Open files in ", ChiFileCommonRoot)
    for file_idx in range(NowCheckFileNum):
        temp_file_name_idx = FileNameRange[file_idx]
        temp_file_root = ChiFileCommonRoot + str(temp_file_name_idx) + ".chi"
        nowFp = open(temp_file_root, "r")
        nowReadData = chi_file_to_DataSet(nowFp)
        nowGenPulseID = 0
        # if laser on -> divided as 24 / off -> 24n + 12
        if RunCondition == "SingleOnOff":
            nowGenPulseID = file_idx * 12
        elif RunCondition == "TempDiffTest":
            if each_info['OpenOption'] == "laser_on":
                nowGenPulseID = file_idx * 24
            else:  # laser_off
                nowGenPulseID = (file_idx * 24) + 12
        nowReadData.check_laser_onoff(nowGenPulseID)
        data_int_val_list.append(nowReadData)
        nowFp.close()
        if (file_idx % 1000) == 0:
            print("open file unitl", FileNameRange[file_idx])

print("end open file & read data")

# calculate q value according to 2theta
common_q_val = tth_to_q(common_tth_val)


def plot_all_given(data_list, x_is_q=False):
    global common_tth_val
    global common_q_val
    num_data_in_one_graph = 5
    num_data = len(data_list)
    if num_data > num_data_in_one_graph:
        now_title = "intensity value of " + str(num_data) + " chi file"

        for data_idx, each_data in enumerate(data_list):
            if x_is_q:
                plt.plot(common_q_val, each_data.intensity_val, label=(str(data_idx + 1) + 'th val'))
            else:
                plt.plot(common_tth_val, each_data.intensity_val, label=(str(data_idx + 1) + 'th val'))
            if ((data_idx + 1) % num_data_in_one_graph) == 0:
                plt.title(now_title)
                if x_is_q:
                    plt.xlabel('q value')
                else:
                    plt.xlabel(r'$2\theta\ value$')
                plt.ylabel("intensity")
                plt.legend()
                plt.show()
        if (num_data % num_data_in_one_graph) != 0:
            plt.title(now_title)
            if x_is_q:
                plt.xlabel('q value')
            else:
                plt.xlabel(r'$2\theta\ value$')
            plt.ylabel("intensity")
            plt.legend()
            plt.show()
    else:
        for data_idx, each_data in enumerate(data_list):
            if x_is_q:
                plt.plot(common_q_val, each_data.intensity_val, label=(str(data_idx + 1) + 'th val'))
            else:
                plt.plot(common_tth_val, each_data.intensity_val, label=(str(data_idx + 1) + 'th val'))
        now_title = "intensity value of " + str(num_data) + "chi file"
        plt.title(now_title)
        if x_is_q:
            plt.xlabel('q value')
        else:
            plt.xlabel(r'$2\theta\ value$')
        plt.ylabel("intensity")
        plt.legend()
        plt.show()


# plot_all_given(data_int_val_list)
# plot_all_given(data_int_val_list, x_is_q=True)

ice_sum_list = []
water_sum_list = []
for file_idx in range(TotalCheckFileNum):
    now_ice_sum, now_water_sum = data_int_val_list[file_idx].classify_data(common_q_val)
    ice_sum_list.append(now_ice_sum)
    water_sum_list.append(now_water_sum)


def plot_sum_for_criteria(data, graph_title, v_line_1=0.0, v_line_2=0.0):
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
    plt.show()


plot_sum_for_criteria(ice_sum_list, "ice sum view", LowerBoundOfIce, UpperBoundOfNotIce)
plot_sum_for_criteria(water_sum_list, "water sum view", LowerBoundOfWater, UpperBoundOfNotWater)


def plot_avg_each_class(data_list):
    ice_counter = 0
    gray_ice_counter = 0
    not_ice_counter = 0
    water_counter = 0
    gray_water_counter = 0
    not_water_counter = 0
    ice_sum = np.zeros_like(data_list[0].intensity_val)
    gray_ice_sum = np.zeros_like(data_list[0].intensity_val)
    not_ice_sum = np.zeros_like(data_list[0].intensity_val)
    water_sum = np.zeros_like(data_list[0].intensity_val)
    gray_water_sum = np.zeros_like(data_list[0].intensity_val)
    not_water_sum = np.zeros_like(data_list[0].intensity_val)
    for each_data in data_list:
        if each_data.ice_type == IceType.ICE:
            ice_counter += 1
            ice_sum += np.array(each_data.intensity_val)
        elif each_data.ice_type == IceType.GRAY_ICE:
            gray_ice_counter += 1
            gray_ice_sum += np.array(each_data.intensity_val)
        else:
            not_ice_counter += 1
            not_ice_sum += np.array(each_data.intensity_val)

        if each_data.water_type == WaterType.WATER:
            water_counter += 1
            water_sum += np.array(each_data.intensity_val)
        elif each_data.water_type == WaterType.GRAY_WATER:
            gray_water_counter += 1
            gray_water_sum += np.array(each_data.intensity_val)
        else:
            not_water_counter += 1
            not_water_sum += np.array(each_data.intensity_val)

    print("print statistic")
    print("ice :", ice_counter, "gray :", gray_ice_counter, "not ice : ", not_ice_counter)
    print("water :", water_counter, "gray :", gray_water_counter, "not water : ", not_water_counter)
    total_ice = ice_counter + gray_ice_counter + not_ice_counter
    total_water = water_counter + gray_water_counter + not_water_counter
    print("total number check (ice)", total_ice, "(water)", total_water)

    division_list = [(ice_sum, ice_counter), (gray_ice_sum, gray_ice_counter), (not_ice_sum, not_ice_counter),
                     (water_sum, water_counter), (gray_water_sum, gray_water_counter),
                     (not_water_sum, not_water_counter)]
    avg_list = []
    for now_sum, now_counter in division_list:
        try:
            now_avg = now_sum / now_counter
        except:
            now_avg = now_sum
        avg_list.append(now_avg)

    ice_avg = avg_list[0]
    gray_ice_avg = avg_list[1]
    not_ice_avg = avg_list[2]
    water_avg = avg_list[3]
    gray_water_avg = avg_list[4]
    not_water_avg = avg_list[5]

    plt.plot(common_q_val, ice_avg, label="ice avg")
    plt.plot(common_q_val, gray_ice_avg, label="gray avg")
    plt.plot(common_q_val, not_ice_avg, label="not ice avg")
    plt.title('average intensity for each ice class')
    plt.xlabel('q value')
    plt.ylabel("average intensity")
    plt.legend()
    plt.show()

    plt.plot(common_q_val, water_avg, label="water avg")
    plt.plot(common_q_val, gray_water_avg, label="gray avg")
    plt.plot(common_q_val, not_water_avg, label="not water avg")
    plt.title('average intensity for each water class')
    plt.xlabel('q value')
    plt.ylabel("average intensity")
    plt.legend()
    plt.show()


plot_avg_each_class(data_int_val_list)


def extract_laser_on_off_water_list(data_list):
    laser_on_water_idx = []
    laser_off_water_idx = []
    for data_idx, each_data in enumerate(data_list):
        if each_data.water_type == WaterType.WATER:
            if each_data.laser_is_on:
                laser_on_water_idx.append(data_idx)
            else:
                laser_off_water_idx.append(data_idx)

    print("laser on droplet : ", len(laser_on_water_idx), "laser off : ", len(laser_off_water_idx))
    return laser_on_water_idx, laser_off_water_idx


def match_water_near_index_pair(data_list, laser_on_water_idx, laser_off_water_idx):
    compare_longer_one = None
    compare_shoter_one = None
    if len(laser_on_water_idx) > len(laser_off_water_idx):
        compare_longer_one = laser_on_water_idx
        compare_shoter_one = laser_off_water_idx
    else:
        compare_longer_one = laser_off_water_idx
        compare_shoter_one = laser_on_water_idx

    neareast_idx_pair = []
    for data_idx in compare_longer_one:
        nearest_idx = find_neareast_pair(compare_shoter_one, data_idx)
        neareast_idx_pair.append((data_idx, nearest_idx))

    print("pairing with nearest index (time order) logic")
    return neareast_idx_pair


def match_water_near_intensity_pair(data_list, laser_on_water_idx, laser_off_water_idx):
    compare_longer_one = None
    compare_shoter_one = None
    if len(laser_on_water_idx) > len(laser_off_water_idx):
        compare_longer_one = laser_on_water_idx
        compare_shoter_one = laser_off_water_idx
    else:
        compare_longer_one = laser_off_water_idx
        compare_shoter_one = laser_on_water_idx

    cmp_long_water_sum = []
    cmp_short_water_sum = []
    for each_idx in compare_longer_one:
        cmp_long_water_sum.append(data_list[each_idx].water_peak_sum)
    for each_idx in compare_shoter_one:
        cmp_short_water_sum.append(data_list[each_idx].water_peak_sum)

    neareast_int_pair = []
    for int_idx, each_int_sum in enumerate(cmp_long_water_sum):
        most_similar_sum = find_neareast_pair(cmp_short_water_sum, each_int_sum)
        sim_sum_idx = cmp_short_water_sum.index(most_similar_sum)
        # print(each_int_sum, "(idx:", compare_longer_one[int_idx], ")",
        #       most_similar_sum, "(idx:", compare_shoter_one[sim_sum_idx], ")")
        neareast_int_pair.append((compare_longer_one[int_idx], compare_shoter_one[sim_sum_idx]))

    print("pairing with nearest integrated intensity logic")
    return neareast_int_pair


water_laser_on_idx, water_laser_off_idx = extract_laser_on_off_water_list(data_int_val_list)
# water_nearest_idx_pair = match_water_near_index_pair(data_int_val_list, water_laser_on_idx, water_laser_off_idx)
water_nearest_int_pair = match_water_near_intensity_pair(data_int_val_list, water_laser_on_idx, water_laser_off_idx)

for each_data in data_int_val_list:
    each_data.norm_given_range()

# print(data_int_val_list)
plot_start_idx = 60
plot_end_idx = -95


def plot_near_idx_diff(data_list, near_idx_pair, draw_each_graph=False):
    global common_q_val
    global plot_start_idx, plot_end_idx
    diff_int_arr = []
    for each_idx_a, each_idx_b in near_idx_pair:
        data_a = data_list[each_idx_a]
        data_b = data_list[each_idx_b]

        laser_on_data = None
        laser_off_data = None
        if data_a.laser_is_on:
            laser_on_data = data_a
            laser_off_data = data_b
        else:
            laser_on_data = data_b
            laser_off_data = data_a

        diff_int = np.array(laser_on_data.intensity_val) - np.array(laser_off_data.intensity_val)
        diff_int_arr.append(diff_int)

    print(len(diff_int_arr))
    if draw_each_graph:
        num_draw_front_diff = 9
        num_data_in_one_graph = 3
        for diff_idx in range(num_draw_front_diff):
            plt.plot(common_q_val[plot_start_idx:plot_end_idx], diff_int_arr[diff_idx][plot_start_idx:plot_end_idx],
                     label=(str(diff_idx + 1) + 'th diff'))
            if ((diff_idx + 1) % num_data_in_one_graph) == 0:
                plt.title("draw each diff - nearest index")
                plt.xlabel('q value')
                plt.ylabel("intensity")
                plt.legend()
                plt.show()

    avg_diff = np.average(diff_int_arr, axis=0)
    plt.plot(common_q_val[plot_start_idx:plot_end_idx], avg_diff[plot_start_idx:plot_end_idx])
    plt.title("average normalized diff - nearest index on/off pair")
    plt.xlabel('q value')
    plt.ylabel("intensity")
    plt.show()
    return avg_diff


def plot_near_int_diff(data_list, near_int_pair, draw_each_graph=False):
    global common_q_val
    global plot_start_idx, plot_end_idx
    diff_int_arr = []
    for each_idx_a, each_idx_b in near_int_pair:
        data_a = data_list[each_idx_a]
        data_b = data_list[each_idx_b]

        laser_on_data = None
        laser_off_data = None
        if data_a.laser_is_on:
            laser_on_data = data_a
            laser_off_data = data_b
        else:
            laser_on_data = data_b
            laser_off_data = data_a

        diff_int = np.array(laser_on_data.intensity_val) - np.array(laser_off_data.intensity_val)
        diff_int_arr.append(diff_int)

    print(len(diff_int_arr))
    if draw_each_graph:
        num_draw_front_diff = 9
        num_data_in_one_graph = 3
        for diff_idx in range(num_draw_front_diff):
            plt.plot(common_q_val[plot_start_idx:plot_end_idx], diff_int_arr[diff_idx][plot_start_idx:plot_end_idx],
                     label=(str(diff_idx + 1) + 'th diff'))
            if ((diff_idx + 1) % num_data_in_one_graph) == 0:
                plt.title("draw each diff - nearest intensity")
                plt.xlabel('q value')
                plt.ylabel("intensity")
                plt.legend()
                plt.show()

    avg_diff = np.average(diff_int_arr, axis=0)
    sem_diff = stats.sem(diff_int_arr, axis=0)
    plt.plot(common_q_val[plot_start_idx:plot_end_idx], avg_diff[plot_start_idx:plot_end_idx])
    plt.title("average normalized diff - similar water integration on/off pair")
    plt.xlabel('q value')
    plt.ylabel("intensity")
    plt.show()
    # plt.figure(figsize=(15, 10))
    plt.errorbar(common_q_val[plot_start_idx:plot_end_idx], avg_diff[plot_start_idx:plot_end_idx],
                 yerr=(sem_diff[plot_start_idx:plot_end_idx] * 1), ecolor='k', fmt='o', markersize=2.8, capsize=4)
    plt.title("average normalized diff with standard error - similar integration pair")
    plt.xlabel('q value')
    plt.ylabel("intensity")
    plt.show()
    return avg_diff, sem_diff


def plot_on_off_whole_avg_diff(data_list, on_idx, off_idx):
    global plot_start_idx, plot_end_idx
    on_int_list = []
    for each_idx in on_idx:
        on_int_list.append(data_list[each_idx].intensity_val)
    on_avg_int = np.average(on_int_list, axis=0)

    off_int_list = []
    for each_idx in off_idx:
        off_int_list.append(data_list[each_idx].intensity_val)
    off_avg_int = np.average(off_int_list, axis=0)

    whole_avg_diff = on_avg_int - off_avg_int
    plt.plot(common_q_val[plot_start_idx:plot_end_idx], whole_avg_diff[plot_start_idx:plot_end_idx])
    plt.title("average normalized diff - whole on/off average diff")
    plt.xlabel('q value')
    plt.ylabel("intensity")
    plt.show()

    plt.plot(common_q_val[plot_start_idx:plot_end_idx], on_avg_int[plot_start_idx:plot_end_idx], label="avg on(highT)")
    plt.plot(common_q_val[plot_start_idx:plot_end_idx], off_avg_int[plot_start_idx:plot_end_idx], label="avg off(lowT)")
    plt.plot(common_q_val[plot_start_idx:plot_end_idx], whole_avg_diff[plot_start_idx:plot_end_idx])
    plt.title("average normalized diff - whole on/off average diff")
    plt.xlabel('q value')
    plt.ylabel("intensity")
    plt.legend()
    plt.show()
    return on_avg_int, off_avg_int, whole_avg_diff


# near_idx_diff = plot_near_idx_diff(data_int_val_list, water_nearest_idx_pair, draw_each_graph=False)
near_int_diff, near_int_diff_err = plot_near_int_diff(data_int_val_list, water_nearest_int_pair, draw_each_graph=True)
whole_on_avg, whole_off_avg, whole_avg_diff = plot_on_off_whole_avg_diff(data_int_val_list, water_laser_on_idx, water_laser_off_idx)

# plt.plot(common_q_val[plot_start_idx:plot_end_idx], near_idx_diff[plot_start_idx:plot_end_idx], label="index pair")
plt.plot(common_q_val[plot_start_idx:plot_end_idx], near_int_diff[plot_start_idx:plot_end_idx], label="intensity pair")
plt.plot(common_q_val[plot_start_idx:plot_end_idx], whole_avg_diff[plot_start_idx:plot_end_idx], label="whole avg")

plt.title("comparison of average normalized diff")
plt.xlabel('q value')
plt.ylabel("intensity")
plt.legend()
plt.show()

"""
SVD of difference part
"""

def pair_to_diff_array(data_list, idx_pair):
    global plot_start_idx, plot_end_idx
    normalized_diff_arr = []
    for each_idx_a, each_idx_b in idx_pair:
        data_a = data_list[each_idx_a]
        data_b = data_list[each_idx_b]

        laser_on_data = None
        laser_off_data = None
        if data_a.laser_is_on:
            laser_on_data = data_a
            laser_off_data = data_b
        else:
            laser_on_data = data_b
            laser_off_data = data_a

        diff_int = np.array(laser_on_data.intensity_val) - np.array(laser_off_data.intensity_val)
        normalized_diff_arr.append(diff_int[plot_start_idx:plot_end_idx])

    return normalized_diff_arr


diff_int_arr = pair_to_diff_array(data_int_val_list, water_nearest_int_pair)

tp_diff_int_arr = np.transpose(np.array(diff_int_arr))
diffSVD = SVDCalc(tp_diff_int_arr)
diffSingVal = diffSVD.calc_svd()

singular_show_num = 50
print(diffSingVal[:singular_show_num])
singular_data_y = diffSingVal[:singular_show_num]
singular_data_y_log = np.log(singular_data_y)
singular_data_x = range(1, len(singular_data_y) + 1)


def plot_singular_value(data_x, data_y, data_y_log):
    color_r = 'tab:red'
    fig, ax1 = plt.subplots()
    ax1.set_xlabel("index of singular value")
    ax1.set_ylabel("singular value", color=color_r)
    ax1.scatter(data_x, data_y, color=color_r)
    ax1.plot(data_x, data_y, color=color_r)
    ax1.tick_params(axis='y', labelcolor=color_r)

    ax2 = ax1.twinx()
    color_b = 'tab:blue'
    ax2.set_ylabel("log scale singular value", color=color_b)
    ax2.scatter(data_x, data_y_log, color=color_b)
    ax2.plot(data_x, data_y_log, color=color_b)
    ax2.tick_params(axis='y', labelcolor=color_b)

    fig.tight_layout()
    plt.show()


plot_singular_value(singular_data_x, singular_data_y, singular_data_y_log)

singular_cut_num = 3
bigSingVal = diffSingVal[:singular_cut_num]
print(bigSingVal)

print("left", diffSVD.leftVec.shape)
print("right", diffSVD.rightVecTrans.shape)
diffSVD.pick_meaningful_data(singular_cut_num)
print("left", diffSVD.meanLeftVec.shape)
print("right", diffSVD.meanRightVec.shape)
diffSVD.plot_left_Vec()
diffSVD.plot_right_Vec()

"""
File out part
"""

file_main_name = RunCondition + "_"
for each_info in FileInfoDictList:
    file_main_name += each_info['FileFolder'].split('/')[0]
    print(each_info['FileFolder'].split('/')[0])
print("now output file main name : ", file_main_name)
print("print option : ", FileOutOption)

data_name = ["Q-value"]
print_arr = [common_q_val]
if FileOutOption['IntPairDiffAvg']:
    data_name.append("PairDiffAVg")
    print_arr.append(near_int_diff)
if FileOutOption['IntPairDiffSE']:
    data_name.append("PairDiffSE")
    print_arr.append(near_int_diff_err)
if FileOutOption['OnWholeAvg']:
    data_name.append("OnAvg")
    print_arr.append(whole_on_avg)
if FileOutOption['OffWholeAvg']:
    data_name.append("OffAvg")
    print_arr.append(whole_off_avg)
if FileOutOption['WholeDiff']:
    data_name.append("WholeAvgDiff")
    print_arr.append(whole_avg_diff)

num_data_column = len(data_name)
real_out_val = np.transpose(np.array(print_arr))
print(real_out_val.shape)

outFileName = FileOutFolder + file_main_name + ".dat"
outFp = open(outFileName, 'w')
for each_name in data_name:
    if each_name == data_name[-1]:
        outFp.write(each_name + "\n")
    else:
        outFp.write(each_name + "\t")
for data_line_idx in range(len(common_q_val)):
    for inline_idx in range(num_data_column):
        if inline_idx == (num_data_column - 1):
            outFp.write("%.5f\n" % real_out_val[data_line_idx][inline_idx])
        else:
            outFp.write("%.5f\t" % real_out_val[data_line_idx][inline_idx])

print("average / se data print and - output file : ", outFileName)

if FileOutOption['SVDresults']:
    # TODO : print SVD result with other file
    SVD_out_file_name1 = FileOutFolder + file_main_name + "_SVD_left.dat"
    SVD_out_file_name2 = FileOutFolder + file_main_name + "_SVD_right.dat"
    SVDOutFp1 = open(SVD_out_file_name1, 'w')
    SVDOutFp2 = open(SVD_out_file_name2, 'w')

    diffSVD.file_output_singular_vectors(SVDOutFp1, SVDOutFp2)

    print("print SVD result with other file")


