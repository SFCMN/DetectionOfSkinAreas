import os
import cv2
import numpy as np
from detection_of_skin_area import rw


class StatisticsData:

    def __init__(self):
        self.value = []   # [[均值、方差、标准差], [均值、方差、标准差]]

    def get_sample_cb_cr_list(self):
        """
        统计出样本空间中所有样本像素点的Cb、Cr值所组成的列表
        :return: cb_list、cr_list
        """
        # 首先获取到所有样本的绝对路径
        sample_path_list = self.__get_files_name("../SkinSample/")
        # 挨个路径读取样本，将其Cb、Cr值记录到列表中
        cb_list = []
        cr_list = []
        for sample_path in sample_path_list:
            image = cv2.imdecode(np.fromfile(sample_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            y_cr_cb = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
            (y, cr, cb) = cv2.split(y_cr_cb)
            (xb, yb) = cb.shape
            for i in range(0, xb):
                for j in range(0, yb):
                    cb_list.append(cb[i][j])
            (xr, yr) = cr.shape
            for i in range(0, xr):
                for j in range(0, yr):
                    cr_list.append(cr[i][j])
        return cb_list, cr_list

    def __get_files_name(self, file_dir):
        """
        将指定目录的子文件路径列表返回
        :param file_dir: 指定的目录
        :return: 子文件路径列表
        """
        for root, dirs, files in os.walk(file_dir):  # 获取指定目录的子目录名列表及子文件名列表
            for i in range(len(files)):
                # 根目录与文件名组合，形成绝对路径
                files[i] = os.path.join(root, files[i])
        return files

    def calculate_numerical(self, list_x, list_y):
        """
        根据给定的两个数列，计算出均值、方差、标准差等一系列数值
        :param list_x: 数列x
        :param list_y: 数列y
        :return: None
        """
        # 计算均值、方差、标准差
        average_x, variance_x, standard_deviation_x = self.__calculate_average_and_variance_and_standard_deviation(list_x)
        average_y, variance_y, standard_deviation_y = self.__calculate_average_and_variance_and_standard_deviation(list_y)

        # 计算得出期望E（x）、E（y）
        E_x = self.__calculate_expect(list_x)
        E_y = self.__calculate_expect(list_y)
        E_xy = 0
        for i in range(0, len(list_x)):
            E_xy += int(list_x[i]) * int(list_y[i])
        E_xy = E_xy / len(list_x)

        # 计算得出协方差Cov(x,y)
        Cov_xy = E_xy - E_x * E_y

        # 定义协方差矩阵C
        C = np.array([
            [variance_x, Cov_xy],
            [Cov_xy, variance_y]
        ])
        # 定义均值向量
        m = np.array([average_x, average_y])

        # 将计算的出的数据全部写入文件中
        numerical_dict = {
            'Average_X': average_x, 'Average_Y': average_y, 'Variance_X': variance_x, 'Variance_Y': variance_y,
            'Standard_Deviation_X': standard_deviation_x, 'Standard_Deviation_Y': standard_deviation_y,
            'E_X': E_x, 'E_Y': E_y, 'E_XY': E_xy,
            'Cov_XY': Cov_xy,
            'Covariance_Matrix': str(C), 'Average_Vector': str(m)
        }
        io = rw.SequenceTXTIO()
        io.write_sequence_into_txt("../TempInfo/模型计算所得值.txt", numerical_dict)

        # 将均值、方差、标准差写入对象属性
        self.value.append([average_x, variance_x, standard_deviation_x])
        self.value.append([average_y, variance_y, standard_deviation_y])

    def __calculate_average_and_variance_and_standard_deviation(self, value_list):
        """
        计算给定数列的均值、方差、标准差
        :param value_list: 给定的数列
        :return: 均值、方差、标准差
        """
        average = 0     # 均值
        variance = 0    # 方差
        for i in value_list:
            average += i
        average = average / len(value_list)

        for i in value_list:
            variance += pow((i - average), 2)
        variance = variance / (len(value_list) - 1)

        standard_deviation = pow(variance, 0.5)

        return average, variance, standard_deviation

    def __statistics_number_of_value_from_list(self, value_list):
        """
        统计指定列表中值的数量，并生成字典返回
        :param value_list: 指定列表
        :return: 字典
        """
        number_dict = {}
        for i in range(len(value_list)):
            if value_list[i] not in number_dict.keys():
                number_dict[value_list[i]] = 1
            else:
                number_dict[value_list[i]] = number_dict[value_list[i]] + 1
        return number_dict

    def __calculate_expect(self, value_list):
        """
        根据给定的数列，计算期望
        :param value_list: 给定的数列
        :return: 期望
        """
        # 统计数列中不同数字的数量
        number_dict = self.__statistics_number_of_value_from_list(value_list)
        # 统计每个数字出现的概率
        list_len = len(value_list)
        for k, v in number_dict.items():
            number_dict[k] = v / list_len   # 此处将数量字典转变为概率字典
        # 期望 = 数值 * 概率
        E = 0  # 期望
        for k, v in number_dict.items():
            E += (k * v)

        return E

    def get_number_dict(self, value_list1, value_list2):
        """
        统计数量
        :param value_list1: 数列1
        :param value_list2: 数列2
        :return: 统计得到的字典
        """
        dict1 = self.__statistics_number_of_value_from_list(value_list1)
        dict2 = self.__statistics_number_of_value_from_list(value_list2)
        return dict1, dict2

    def calculate_range_of_skin(self):
        """
        根据已经计算出的均值、方差、标准差计算得出肤色范围
        :return: 范围列表
        """
        if self.value is [] or len(self.value) == 1:
            return None
        else:
            skin_range = []  # 范围列表
            value = 1.835
            for i in range(0, 2):
                range_left = int(self.value[i][0] - value * self.value[i][2])
                range_right = int(self.value[i][0] + value * self.value[i][2])
                skin_range.append((range_left, range_right))
            io = rw.SequenceTXTIO()
            io.write_sequence_into_txt("../TempInfo/Cb、Cr范围值.txt", str(skin_range))
            return skin_range

    def calculate_probability(self, C, m, list1, list2):
        """
        计算肤色似然率
        :param C:
        :param m:
        :param list1:
        :param list2:
        :return:
        """
        # 矩阵相乘  np.dot(A,B)
        # 矩阵转置  np.transpose(A)
        # 逆矩阵    np.linalg.inv(A)
        # exp       np.exp

        # 开始计算概率
        list_probability = []
        for i in range(0, len(list1)):
            x = np.array([list1[i], list2[i]])
            probability = np.exp(-0.5 * np.dot(np.dot(np.transpose(x - m), np.linalg.inv(C)), (x - m)))
            list_probability.append(probability)
        max_probability = max(list_probability)
        for i in range(0, len(list_probability)):
            list_probability[i] = list_probability[i] / max_probability
        return list_probability


    #
    # def plot_of_number_dict(self, title, number_dict):
    #     """
    #     根据给定的字典绘制值和数量的折线图
    #     :param title: 折线图标题
    #     :param number_dict: 给定的字典
    #     :return: None
    #     """
    #     number_dict = dict(sorted(number_dict.items(), key=operator.itemgetter(0)))
    #     key_list = list(number_dict.keys())
    #     value_list = list(number_dict.values())
    #     # print(type(key_list))
    #     # print(key_list)
    #     # print(value_list)
    #     plt.title(title)
    #     plt.xlabel("Value")
    #     plt.ylabel("Num")
    #     # 根据给定的散点坐标画折线图
    #     plt.plot(key_list, value_list, linewidth=1)
    #     plt.show()
    #

    #
    #
    # def calculate_range(self, average, standard_deviation):
    #     """
    #     根据给定的均值和标准差计算出范围
    #     :param average: 均值
    #     :param standard_deviation: 标准差
    #     :return: 范围
    #     """
    #     # 2.58、1.96、1.92、1.91、1.85、1.84、1.835、1.83、1.76、1.66、1.46、1
    #     value = 1.835
    #     range_left = int(average - value * standard_deviation)
    #     range_right = int(average + value * standard_deviation)
    #     print("范围是：（" + str(range_left) + "," + str(range_right) + ")")
    #     return range_left, range_right
    #
    # def calculate(self, list_x, list_y):
    #     # 计算均值、方差、标准差
    #     average_x, variance_x, standard_deviation_x = self.calculate_average_and_variance_and_standard_deviation(list_x)
    #     average_y, variance_y, standard_deviation_y = self.calculate_average_and_variance_and_standard_deviation(list_y)
    #
    #     # 计算得出期望E（x）、E（y）
    #     dict_x = self.statistics_number_of_value_from_list(list_x)  # 统计个数
    #     # 统计每个元素出现的概率
    #     len_x = len(list_x)
    #     for k, v in dict_x.items():
    #         dict_x[k] = v / len_x
    #     # print("字典x得出现概率：" + str(dict_x))
    #     E_x = 0   # 期望
    #     for k, v in dict_x.items():
    #         E_x += (k * v)
    #
    #     dict_y = self.statistics_number_of_value_from_list(list_y)  # 统计个数
    #     # 统计每个元素出现的概率
    #     len_y = len(list_y)
    #     for k, v in dict_y.items():
    #         dict_y[k] = v / len_y
    #     # print("字典y得出现概率：" + str(dict_y))
    #     E_y = 0  # 期望
    #     for k, v in dict_y.items():
    #         E_y += (k * v)
    #
    #     # 计算得出E（xy）
    #     E_xy = 0
    #     for i in range(0, len(list_x)):
    #         E_xy += int(list_x[i]) * int(list_y[i])
    #     E_xy = E_xy / len(list_x)
    #
    #     # 计算得出Cov(x,y)
    #     Cov_xy = E_xy - E_x * E_y
    #
    #     # 定义协方差矩阵C
    #     C = np.array([
    #         [variance_x, Cov_xy],
    #         [Cov_xy, variance_y]
    #     ])
    #     # 定义均值向量
    #     m = np.array([average_x, average_y])
    #     # 将计算的出的数据全部写入文件中
    #     # 均值、方差、标准差、期望、协方差、
    #     dict_res = {
    #         '均值x': average_x, '均值y': average_y, '方差x': variance_x, '方差y': variance_y,
    #         '标准差x': standard_deviation_x, '标准差y': standard_deviation_y,
    #         '期望x': E_x, '期望y': E_y, '期望xy': E_xy,
    #         '协方差xy': Cov_xy,
    #         '协方差矩阵C': str(C), '均值向量m': str(m)
    #     }
    #     io = SequenceTXTIO()
    #     io.write_sequence_into_txt("../TempInfo/模型计算所得值.txt", dict_res)
    #
    #
    #
    # def calculate_probability(self, C, m, list1, list2):
    #     # 矩阵相乘  np.dot(A,B)
    #     # 矩阵转置  np.transpose(A)
    #     # 逆矩阵    np.linalg.inv(A)
    #     # exp       np.exp
    #
    #     # 开始计算概率
    #     list_probability = []
    #     for i in range(0, len(list1)):
    #         x = np.array([list1[i], list2[i]])
    #         probability = np.exp(-0.5 * np.dot(np.dot(np.transpose(x - m), np.linalg.inv(C)), (x - m)))
    #         list_probability.append(probability)
    #     max_probability = max(list_probability)
    #     for i in range(0, len(list_probability)):
    #         list_probability[i] = list_probability[i] / max_probability
    #     return list_probability