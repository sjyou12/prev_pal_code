# TODO: load saved numpy array: avg of each delay for each run
# TODO : do svd for analysis t-dependent feature
import numpy as np
import matplotlib.pyplot as plt
from codes.old_SVDCalc import SVDCalc

# cut neg delay_avg
svd_start_idx = 93
svd_end_idx = -180

run_name_temp_map = dict(run20=260, run21=260,
                         run23=250, run24=250, run25=245, run26=245, run27=240, run28=240, run29=235, run30=235)
temp_list = list(set(run_name_temp_map.values()))
temp_list.sort(reverse=True)
num_delay_in_one_delay_group = 8

def delay_group_avg_SVD():
    """
    apply SVD to each delay group average for given run (n>=1).
    """
    global svd_start_idx, svd_end_idx, run_name_temp_map, temp_list

    delay_1_avg_list = []
    delay_2_avg_list = []
    neg_delay_avg_list = []

    for each_file in run_name_temp_map.keys():
        outDir = "../results/"
        test_file_family_name = each_file
        test_file_index = "00001"
        test_file_main_name = test_file_family_name + "_" + test_file_index

        laser_off_avg_file_name = outDir + "t_dep/laser_off_avg/" + "laser_off_avg_" + test_file_main_name + ".npy"
        delay_group_avg_file_name = outDir + "t_dep/delay_group_avg/" + "delay_group_avg_" + test_file_main_name + ".npy"

        laser_off_avg = np.load(laser_off_avg_file_name)
        delay_group_avg = np.load(delay_group_avg_file_name)

        neg_delay_avg_list.append(laser_off_avg[svd_start_idx:svd_end_idx])
        delay_1_avg_list.append(delay_group_avg[0])
        delay_2_avg_list.append(delay_group_avg[1])

    print(np.shape(neg_delay_avg_list), np.shape(delay_1_avg_list), np.shape(delay_2_avg_list))


    def calc_SVD_and_plot(data, file_out=False, file_name=""):
        tp_data = np.transpose(data)
        avgSVD = SVDCalc(tp_data)
        avgSingVal = avgSVD.calc_svd()

        singular_show_num = 50
        print(avgSingVal[:singular_show_num])
        singular_data_y = avgSingVal[:singular_show_num]
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
        bigSingVal = avgSingVal[:singular_cut_num]
        print(bigSingVal)

        print("left", avgSVD.leftVec.shape)
        print("right", avgSVD.rightVecTrans.shape)
        avgSVD.pick_meaningful_data(singular_cut_num)
        print("left", avgSVD.meanLeftVec.shape)
        print("right", avgSVD.meanRightVec.shape)
        avgSVD.plot_left_Vec(sep_graph=True)
        # delay_1_start = each_delay_count[0]
        # delay_2_start = each_delay_count[0] + each_delay_count[1]
        run_per_each_temp = int(len(run_name_temp_map.keys()) / len(temp_list))
        for temp_idx in range(len(temp_list)):
            plt.axvline(x=(temp_idx*run_per_each_temp), alpha=0.3, color='r')
        avgSVD.plot_right_Vec(sep_graph=False)

        if file_out:
            file_main_name = "temp-dep-avg-"
            left_file_name = "../results/t_dep/delay_group_avg_svd/" + file_main_name + file_name +"_LSV.dat"
            right_file_name = "../results/t_dep/delay_group_avg_svd/" + file_main_name + file_name+ "_RSV.dat"
            singVal_file_name = "../results/t_dep/delay_group_avg_svd/" + file_main_name + file_name + "_SingVal.dat"
            leftFp = open(left_file_name, 'w')
            rightFp = open(right_file_name, 'w')
            svalFp = open(singVal_file_name, 'w')
            avgSVD.file_output_singular_vectors(leftFp, rightFp)
            print("LSV, RSV file print")
            avgSVD.file_output_singular_value(svalFp)
            leftFp.close()
            rightFp.close()
            svalFp.close()

    calc_SVD_and_plot(neg_delay_avg_list, file_out=True, file_name="neg-delay")
    calc_SVD_and_plot(delay_1_avg_list, file_out=True, file_name="100ps")
    calc_SVD_and_plot(delay_2_avg_list, file_out=True, file_name="1ns")

# delay_group_avg_SVD()

def each_delay_avg_SVD():
    """
    apply SVD to concatenated each delay average array for given run(n>=1)
    there are 8 delay for each run & each delay group
    """
    global svd_start_idx, svd_end_idx, run_name_temp_map, num_delay_in_one_delay_group

    delay_1_each_delay_avg_list = []
    delay_2_each_delay_avg_list = []
    neg_delay_each_delay_avg_list = []

    for each_file in run_name_temp_map.keys():
        outDir = "../results/"
        test_file_family_name = each_file
        test_file_index = "00001"
        test_file_main_name = test_file_family_name + "_" + test_file_index

        each_delay_avg_file_name = outDir + "t_dep/each_delay_avg/" + "each_delay_avg_" + test_file_main_name + ".npy"
        laser_off_each_delay_avg_file_name = outDir + "t_dep/off_each_delay_avg/" + "off_each_avg_" + test_file_main_name + ".npy"

        laser_off_avg = np.load(laser_off_each_delay_avg_file_name)
        delay_group_avg = np.load(each_delay_avg_file_name)

        laser_off_middle = np.split(laser_off_avg, (svd_start_idx, svd_end_idx), axis=1)[1]
        neg_delay_each_delay_avg_list.append(laser_off_middle)
        delay_1_each_delay_avg_list.append(delay_group_avg[:num_delay_in_one_delay_group])
        delay_2_each_delay_avg_list.append(delay_group_avg[num_delay_in_one_delay_group:])

    def calc_SVD_and_plot(data, file_out=False, file_name=""):
        tp_data = np.transpose(data)
        avgSVD = SVDCalc(tp_data)
        avgSingVal = avgSVD.calc_svd()

        singular_show_num = 50
        print(avgSingVal[:singular_show_num])
        singular_data_y = avgSingVal[:singular_show_num]
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

        singular_cut_num = 5
        bigSingVal = avgSingVal[:singular_cut_num]
        print(bigSingVal)

        print("left", avgSVD.leftVec.shape)
        print("right", avgSVD.rightVecTrans.shape)
        avgSVD.pick_meaningful_data(singular_cut_num)
        print("left", avgSVD.meanLeftVec.shape)
        print("right", avgSVD.meanRightVec.shape)
        avgSVD.plot_left_Vec(sep_graph=True)
        # delay_1_start = each_delay_count[0]
        # delay_2_start = each_delay_count[0] + each_delay_count[1]
        one_group_num = 8
        if file_name == "neg-delay":
            one_group_num = 16
        run_per_each_temp = int(len(run_name_temp_map.keys()) / len(temp_list))
        for temp_idx in range(len(temp_list)):
            plt.axvline(x=(temp_idx*run_per_each_temp*one_group_num), alpha=0.3, color='r')
        avgSVD.plot_right_Vec(sep_graph=False)

        if file_out:
            file_main_name = "temp-dep-avg-"
            left_file_name = "../results/t_dep/each_delay_svd/" + file_main_name + file_name +"_LSV.dat"
            right_file_name = "../results/t_dep/each_delay_svd/" + file_main_name + file_name+ "_RSV.dat"
            singVal_file_name = "../results/t_dep/each_delay_svd/" + file_main_name + file_name + "_SingVal.dat"
            leftFp = open(left_file_name, 'w')
            rightFp = open(right_file_name, 'w')
            svalFp = open(singVal_file_name, 'w')
            avgSVD.file_output_singular_vectors(leftFp, rightFp)
            print("LSV, RSV file print")
            avgSVD.file_output_singular_value(svalFp)
            leftFp.close()
            rightFp.close()
            svalFp.close()

    neg_delay = np.concatenate(neg_delay_each_delay_avg_list, axis=0)
    delay_1 = np.concatenate(delay_1_each_delay_avg_list, axis=0)
    delay_2 = np.concatenate(delay_2_each_delay_avg_list, axis=0)
    print(np.shape(neg_delay_each_delay_avg_list), np.shape(delay_1_each_delay_avg_list), np.shape(delay_2_each_delay_avg_list))
    print(np.shape(neg_delay), np.shape(delay_1), np.shape(delay_2))


    calc_SVD_and_plot(neg_delay, file_out=True, file_name="neg-delay")
    calc_SVD_and_plot(delay_1, file_out=True, file_name="100ps")
    calc_SVD_and_plot(delay_2, file_out=True, file_name="1ns")

# each_delay_avg_SVD()

def big_run_delay_avg_SVD(run_name):
    """
    SVD analysis for each big run (54-delay run)
    SVD with each delay average array
    """
    """
    apply SVD to each delay group average for given run (n>=1).
    """
    global svd_start_idx, svd_end_idx

    print("svd anal of ", run_name)
    outDir = "../results/"
    test_file_family_name = run_name
    test_file_index = "00001"
    test_file_main_name = test_file_family_name + "_" + test_file_index

    each_delay_avg_file_name = outDir + "t_dep/each_delay_avg/" + "each_delay_avg_" + test_file_main_name + ".npy"
    laser_off_each_delay_avg_file_name = outDir + "t_dep/off_each_delay_avg/" + "off_each_avg_" + test_file_main_name + ".npy"

    laser_off_avg = np.load(laser_off_each_delay_avg_file_name)
    delay_group_avg = np.load(each_delay_avg_file_name)

    laser_off_middle = np.split(laser_off_avg, (svd_start_idx, svd_end_idx), axis=1)[1]
    neg_delay = laser_off_middle
    each_delay = delay_group_avg

    def calc_SVD_and_plot(data, file_out=False, file_name=""):
        tp_data = np.transpose(data)
        avgSVD = SVDCalc(tp_data)
        avgSingVal = avgSVD.calc_svd()

        singular_show_num = 50
        print(avgSingVal[:singular_show_num])
        singular_data_y = avgSingVal[:singular_show_num]
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

        singular_cut_num = 5
        bigSingVal = avgSingVal[:singular_cut_num]
        print(bigSingVal)

        print("left", avgSVD.leftVec.shape)
        print("right", avgSVD.rightVecTrans.shape)
        avgSVD.pick_meaningful_data(singular_cut_num)
        print("left", avgSVD.meanLeftVec.shape)
        print("right", avgSVD.meanRightVec.shape)
        avgSVD.plot_left_Vec(sep_graph=True)

        avgSVD.plot_right_Vec(sep_graph=False)

        if file_out:
            file_main_name = "I0-each-delay-avg-"
            left_file_name = "../results/svd/54_delay_run/" + file_main_name + file_name +"_LSV.dat"
            right_file_name = "../results/svd/54_delay_run/" + file_main_name + file_name+ "_RSV.dat"
            singVal_file_name = "../results/svd/54_delay_run/" + file_main_name + file_name + "_SingVal.dat"
            leftFp = open(left_file_name, 'w')
            rightFp = open(right_file_name, 'w')
            svalFp = open(singVal_file_name, 'w')
            avgSVD.file_output_singular_vectors(leftFp, rightFp)
            print("LSV, RSV file print")
            avgSVD.file_output_singular_value(svalFp)
            leftFp.close()
            rightFp.close()
            svalFp.close()

    print(np.shape(neg_delay), np.shape(each_delay))


    calc_SVD_and_plot(neg_delay, file_out=True, file_name=(run_name +"-neg-delay"))
    calc_SVD_and_plot(each_delay, file_out=True, file_name=(run_name+"-pos-delay"))

big_run_delay_avg_SVD("run49")