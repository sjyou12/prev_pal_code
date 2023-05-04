import numpy as np
from matplotlib import pyplot as plt

'''
필요한 기능 
1. 하나의 row로 된 data를 받아와서 기존 matrix에 지속적으로 추가해 matrix를 만들 수 있어야 한다. 
여러 코드에서 생성된 data들을 어떤식으로 합칠건지? 
2. 만들어진 matrix에 대해서 svd 계산 할 수 있어야 한다. 
(추가적인 기준이나 커맨드를을 준다던가? 그것들을 받아서 원하는대로 svd진행)
 
추가적인 fitting 기능
3. S값들 fitting
4. U, V값들 fitting
5. U값들 exponential fitting..등등의 확장기능

다른 사람이 쓸 수 있도록 도큐멘테이션 제대로 할것 

'''
class SVDCalc:
    def __init__(self, data):
        self.originalData = data
        self.leftVec = None
        self.rightVecTrans = None
        self.rightVec = None
        self.singValVec = None
        self.meanLeftVec = None
        self.meanRightVec = None
        self.meanSingValVec = None
        self.meanNum = None

    def calc_svd(self):
        u, s, vh = np.linalg.svd(self.originalData, full_matrices=False)
        self.leftVec = u
        self.rightVecTrans = vh
        self.singValVec = s
        self.rightVec = np.transpose(self.rightVecTrans)
        return s

    @staticmethod
    def choose_n_left_column(array, num):
        tempArr = np.transpose(array)
        tempArr = tempArr[0:num]
        return np.transpose(tempArr)

    def pick_meaningful_data(self, mean_num):
        self.meanNum = mean_num
        self.meanSingValVec = self.singValVec[0:mean_num]
        self.meanLeftVec = self.choose_n_left_column(self.leftVec, mean_num)
        self.meanRightVec = self.choose_n_left_column(self.rightVec, mean_num)
        '''self.meanLeftVec = self.leftVec[0:mean_num]
        self.meanRightVec = self.rightVec[0:mean_num]'''

    def plot_left_Vec(self, abs=False, sep_graph=False):
        transLeft = np.transpose(self.meanLeftVec)
        if abs:  # only work when abs is True
            transLeft = np.abs(transLeft)
        if sep_graph:
            for sp_idx in range(0, self.meanNum):
                plt.plot(transLeft[sp_idx], label=("leftVec" + str(sp_idx + 1)))
                plt.title("Left singular vectors")
                plt.legend()
                plt.show()
        for sp_idx in range(0, self.meanNum):
            plt.plot(transLeft[sp_idx], label=("leftVec" + str(sp_idx + 1)))
        plt.title("Left singular vectors")
        plt.legend()
        plt.show()

    def plot_left_vec_with_x_val(self, graph_title, x_val, sep_graph=False):
        transLeft = np.transpose(self.meanLeftVec)
        if sep_graph:
            for sp_idx in range(0, self.meanNum):
                plt.plot(x_val, transLeft[sp_idx], label=("leftVec" + str(sp_idx + 1)))
                plt.title("Left singular vectors")
                plt.legend()
                plt.show()
        for sp_idx in range(0, self.meanNum):
            plt.plot(x_val, transLeft[sp_idx], label=("leftVec" + str(sp_idx + 1)))
        plt.title(graph_title)
        plt.legend()
        plt.show()

    def plot_right_Vec(self, abs=False, v_line_1=0.0, v_line_2=0.0, sep_graph=False):
        transRight = np.transpose(self.meanRightVec)
        if abs:
            transRight = np.abs(transRight)
        if sep_graph:
            for sp_idx in range(0, self.meanNum):
                plt.plot(transRight[sp_idx], label=("rightVec" + str(sp_idx + 1)))
                if v_line_1 != 0.0:
                    plt.axvline(x=v_line_1, color='r')
                if v_line_2 != 0.0:
                    plt.axvline(x=v_line_2, color='r')
                # plt.ylim(-0.05, 0.05)
                plt.title("Right singular vectors")
                plt.legend()
                plt.show()
        for sp_idx in range(0, self.meanNum):
            plt.plot(transRight[sp_idx], label=("rightVec" + str(sp_idx + 1)))
        if v_line_1 != 0.0:
            plt.axvline(x=v_line_1, color='r')
        if v_line_2 != 0.0:
            plt.axvline(x=v_line_2, color='r')
        # plt.ylim(-0.05, 0.05)
        plt.title("Right singular vectors")
        plt.legend()
        plt.show()

    def plot_right_vec_with_x_text(self, graph_title, x_text, v_line_1=0.0, v_line_2=0.0, sep_graph=False):
        transRight = np.transpose(self.meanRightVec)
        if sep_graph:
            for sp_idx in range(0, self.meanNum):
                plt.plot(transRight[sp_idx], label=("rightVec" + str(sp_idx + 1)))
                if v_line_1 != 0.0:
                    plt.axvline(x=v_line_1, color='r')
                if v_line_2 != 0.0:
                    plt.axvline(x=v_line_2, color='r')
                # plt.ylim(-0.05, 0.05)
                plt.title("Right singular vectors")
                plt.legend()
                plt.show()
        for sp_idx in range(0, self.meanNum):
            value_len = len(transRight[sp_idx])
            plt.plot(transRight[sp_idx], label=("rightVec" + str(sp_idx + 1)))
            try:
                plt.xticks(ticks=range(value_len), labels=x_text[:value_len])
            except:
                print("plot xticks error :", range(value_len), x_text[:value_len])
        if v_line_1 != 0.0:
            plt.axvline(x=v_line_1, color='r')
        if v_line_2 != 0.0:
            plt.axvline(x=v_line_2, color='r')
        # plt.ylim(-0.05, 0.05)
        plt.title(graph_title)
        plt.legend()
        plt.show()

    def file_output_singular_vectors(self, leftFp, rightFp):
        transLeft = np.transpose(self.meanLeftVec)
        transRight = np.transpose(self.meanRightVec)
        # print left
        leftFp.write("value-idx\t")
        for idx in range(0, self.meanNum):
            leftFp.write("leftVec" + str(idx + 1) + "\t")
        leftFp.write("\n")
        for line_num in range(0, self.meanLeftVec.shape[0]):
            leftFp.write(str(line_num + 1))
            leftFp.write("\t")
            for sp_idx in range(0, self.meanNum):
                leftFp.write(str(self.meanLeftVec[line_num][sp_idx]))
                leftFp.write("\t")
            leftFp.write("\n")

        # print right
        rightFp.write("value-idx\t")
        for idx in range(0, self.meanNum):
            rightFp.write("rightVec" + str(idx + 1) + "\t")
        rightFp.write("\n")
        for line_num in range(0, self.meanRightVec.shape[0]):
            rightFp.write(str(line_num + 1))
            rightFp.write("\t")
            for sp_idx in range(0, self.meanNum):
                rightFp.write(str(self.meanRightVec[line_num][sp_idx]))
                rightFp.write("\t")
            rightFp.write("\n")

    def file_output_singular_vectors_with_label(self, leftFp, rightFp, leftLableName, leftLabel, rightLabelName, rightLabel):
        # transLeft = np.transpose(self.meanLeftVec)
        # transRight = np.transpose(self.meanRightVec)

        # print left
        leftFp.write(leftLableName + "\t")
        for idx in range(self.meanNum):
            leftFp.write("leftVec" + str(idx + 1) + "\t")
        leftFp.write("\n")
        for line_num in range(self.meanLeftVec.shape[0]):
            leftFp.write(str(leftLabel[line_num]))
            leftFp.write("\t")
            for sp_idx in range(self.meanNum):
                leftFp.write(str(self.meanLeftVec[line_num][sp_idx]))
                leftFp.write("\t")
            leftFp.write("\n")

        # print right
        rightFp.write(rightLabelName + "\t")
        for idx in range(self.meanNum):
            rightFp.write("rightVec" + str(idx + 1) + "\t")
        rightFp.write("\n")
        for line_num in range(self.meanRightVec.shape[0]):
            rightFp.write(str(rightLabel[line_num]))
            rightFp.write("\t")
            for sp_idx in range(self.meanNum):
                rightFp.write(str(self.meanRightVec[line_num][sp_idx]))
                rightFp.write("\t")
            rightFp.write("\n")

    def file_output_singular_value(self, svalFp):
        svalFp.write("Singular-Value\n")
        for line_num in range(self.meanSingValVec.shape[0]):
            svalFp.write(str(self.meanSingValVec[line_num]) + "\n")