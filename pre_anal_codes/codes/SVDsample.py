from codes.old_SVDCalc import SVDCalc
import numpy as np
from matplotlib import pyplot as plt
datatype = "I2"
# datatype = "sythesis"
# datatype = "noise_syn"

syn_group = ["noise_syn","sythesis"]

if datatype == "I2":
    # for I2 files
    folder_name = "I2_MeOH/"
    extention = ""
    file_name_list = ["100ps", "300ps", "700ps", "1ns", "3ns", "7ns", "10ns", "1us"]
    out_file_id = "I2"
elif datatype in syn_group:
    # for my generation data
    if datatype == "sythesis":
        folder_name = "syn_data/"
        out_file_id = "syn_data"
    elif datatype == "noise_syn":
        folder_name = "syn_noise_data/"
        out_file_id = "syn_noise"
    extention = ".txt"
    file_name_list = ["100ps", "300ps", "700ps", "1ns", "3ns", "7ns", "10ns"]


def simple_my_file_format(dataFp, dataArr, idx, temp_file_name):
    while True:
        dataStr = dataFp.readline()
        if dataStr == '':
            break
        newData = dataStr.strip('\n')
        try:
            newFloatData = float(newData)
        except:
            newFloatData = 0.0
            print("float converting error in " + newData + "from file " + temp_file_name)
        dataArr[idx].append(newFloatData)


def sample_data_read(dataArr):
    for idx, file_name in enumerate(file_name_list):
        temp_file_name = "sample_data/" + folder_name + file_name + extention
        dataFp = open(temp_file_name, 'r')
        while True:
            dataStr = dataFp.readline()
            if dataStr == '':
                break
            newData = dataStr.strip('\n')
            try:
                newFloatData = float(newData)
            except:
                newFloatData = 0.0
                print("float converting error in " + newData + "from file " + temp_file_name)
            dataArr[idx].append(newFloatData)


qValArr = []
stdArr = []


def I2_data_read(dataArr):
    global qValArr
    global stdArr
    for idx, file_name in enumerate(file_name_list):
        temp_file_name = "sample_data/" + folder_name + file_name + extention
        dataFp = open(temp_file_name, 'r')

        while True:
            dataStr = dataFp.readline()
            if dataStr == '':
                break
            newData = dataStr.strip('\n')
            if newData == "#":
                continue
            split_temp_data = newData.split()
            print(split_temp_data)
            try:
                newFloatData = float(split_temp_data[1]) * float(split_temp_data[0])
                print(newFloatData)
            except:
                newFloatData = 0.0
                print("float converting error in " + newData + "from file " + temp_file_name)
            qValArr[idx].append(split_temp_data[0])
            dataArr[idx].append(newFloatData)
            stdArr[idx].append(split_temp_data[2])


dataArrFromFile = []
for i in range(0, len(file_name_list)):
    dataArrFromFile.append([])
    qValArr.append([])
    stdArr.append([])
# print(dataArrFromFile)

if datatype == "I2":
    I2_data_read(dataArrFromFile)
elif datatype in syn_group:
    sample_data_read(dataArrFromFile)
# cut front 300 data

orignial_data = None
front_300_cut_data = None


def cut_front_300_data(dataArrFromFile):
    global orignial_data
    global front_300_cut_data
    orignial_data = dataArrFromFile
    backDataArr = []
    for idx in range(0, len(dataArrFromFile)):
        temp_front_cut = dataArrFromFile[idx][300:]
        backDataArr.append(temp_front_cut)
    front_300_cut_data = backDataArr
    return backDataArr


# I2 -> unstable front data
if datatype == "I2":
    dataArrFromFile = cut_front_300_data(dataArrFromFile)
dataArrFromFile = np.array(dataArrFromFile)
dataArrFromFile = np.transpose(dataArrFromFile)
print(dataArrFromFile.shape)

testDataCalc = SVDCalc(dataArrFromFile)
testSingVal = testDataCalc.calc_svd()
print(testSingVal)

singular_data_x = range(1, len(testSingVal) + 1)
singular_data_y = testSingVal
singular_data_y_log = np.log(testSingVal)


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


# manual selection
def manual_num_selection():
    while True:
        bigDataNum = input("choose number of meaningful singular value. (유의미하게 큰 값의 개수를 선택하세요)\n>>")
        try:
            bigDataNumInt = int(bigDataNum)
            break
        except:
            print("please enter integer value")
            print("singular values from exp data : ", testSingVal)
    return bigDataNumInt


bigDataNum = manual_num_selection()
# bigDataNum = 3  # automatic selection
bigSingVal = testSingVal[0:int(bigDataNum)]
print(bigSingVal)

print("left", testDataCalc.leftVec.shape)
print("right", testDataCalc.rightVecTrans.shape)
testDataCalc.pick_meaningful_data(bigDataNum)
print("left", testDataCalc.meanLeftVec.shape)
print("right", testDataCalc.meanRightVec.shape)

testDataCalc.plot_left_Vec()
# testDataCalc.plot_left_Vec(abs=True)
testDataCalc.plot_right_Vec()
# testDataCalc.plot_right_Vec(abs=True)
if datatype == "I2":
    testDataCalc.plot_right_Vec_log()

out_file_name = "sample_data/svd_data_out/svdout" + out_file_id
out_file_name1 = out_file_name + "_left.dat"
out_file_name2 = out_file_name + "_right.dat"
outFp1 = open(out_file_name1, 'w')
outFp2 = open(out_file_name2, 'w')

testDataCalc.file_output_singular_vectors(outFp1, outFp2)
