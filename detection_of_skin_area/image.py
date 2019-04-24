from PIL import Image
import cv2
import numpy as np
from detection_of_skin_area import rw, data
import time
import matplotlib.pyplot as plt
import operator


class ImageProcess:

    def __get_plot_image(self, title, number_dict):
        """
        根据给定的字典绘制值和数量的折线图
        :param title: 折线图标题
        :param number_dict: 给定的字典
        :return: None
        """
        number_dict = dict(sorted(number_dict.items(), key=operator.itemgetter(0)))
        key_list = list(number_dict.keys())
        value_list = list(number_dict.values())
        plt.title(title)
        plt.xlabel("Value")
        plt.ylabel("Num")
        # 根据给定的散点坐标画折线图
        plt.plot(key_list, value_list, linewidth=1)

        plt.savefig("../TempInfo/" + title + ".jpg")
        plt.clf()
        plt.close()

    def plot_of_number_dict(self):
        """
        根据给定的字典绘制值和数量的折线图
        :param title: 折线图标题
        :param number_dict: 给定的字典
        :return: None
        """
        statistics = data.StatisticsData()
        list_cb, list_cr = statistics.get_sample_cb_cr_list()
        dict1, dict2 = statistics.get_number_dict(list_cb, list_cr)
        self.__get_plot_image("Sample Cb-Value Distribution Image", dict1)
        self.__get_plot_image("Sample Cr-Value Distribution Image", dict2)

    def build_model(self, dialog):
        """
        模型建立，计算一系列数值
        :param dialog: 弹窗
        :return: None
        """
        statistics = data.StatisticsData()
        dialog.set_text_lable("数据统计中...")
        list_cb, list_cr = statistics.get_sample_cb_cr_list()
        dialog.set_text_lable("数值计算中...")
        statistics.calculate_numerical(list_cb, list_cr)
        dialog.set_text_lable("肤色范围计算中...")
        skin_range = statistics.calculate_range_of_skin()
        dialog.set_text_lable("模型建立完毕！")
        time.sleep(1)

    def denoise_median_blur(self, image):
        """
        中值滤波去噪
        :param image: 图像
        :return: 滤波后的图像
        """
        img = cv2.medianBlur(image, 3)  # 中值滤波
        return img

    def denoise_blur(self, image):
        """
        均值滤波去噪
        :param image: 图像
        :return: 滤波后的图像
        """
        img = cv2.blur(image, (3, 5))
        return img

    def denoise_gaussian_blur(self, image):
        """
        高斯滤波去噪
        :param image: 图像
        :return: 滤波后的图像
        """
        img = cv2.GaussianBlur(image, (5, 5), cv2.BORDER_DEFAULT)
        return img

    def denoise_bilateral_filter(self, image):
        """
        双边滤波去噪
        :param image: 图像
        :return: 滤波后的图像
        """
        img = cv2.bilateralFilter(image, 9, 75, 75)
        return img

    def illumination_compensation_gray_world(self, image):
        """
        基于灰度世界算法的光照补偿
        :param image: 图像
        :return: 光照补偿后的图像
        """
        # GrayWorld 色彩均衡算法
        # GrayWorld 色彩均衡算法是一种彩色均衡的方法
        # 它基于“灰度世界假设”
        # 即对于一幅有着大量色彩变化的图像，其 R、G、B 三个颜色分量各自的平均值均近似于同一个灰度值
        # 它的基本思想是:
        # 首先分别计算原始图像三个颜色分量的平均值avgR 、avgG、avgB 和原始图像的平均灰度值 avgGray
        # 然后分别调整每个像素的 R、G、B值，使得调整后图像的三个颜色分量的平均值都近似于平均灰度值avgGray

        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # 将传参过来的OpenCV图转换成PIL.Image格式
        # 分离出图像的三个通道
        r, g, b = image.split()
        # 分别将三个通道的图像转化为数组
        rd = np.asarray(r)
        gd = np.asarray(g)
        bd = np.asarray(b)
        # 计算三个颜色分量的平均值
        avg_r = rd.mean()
        avg_g = gd.mean()
        avg_b = bd.mean()
        # 图像的平均灰度值
        avg_gray = np.divide(np.sum([avg_r, avg_g, avg_b]), 3.0)
        # 将图像的平均灰度值与各通道的平均值相除
        a_r = np.divide(avg_gray, avg_r)
        a_g = np.divide(avg_gray, avg_g)
        a_b = np.divide(avg_gray, avg_b)
        # 调整图像中每个像素的RGB通道值
        c_r = np.multiply(rd, a_r)
        c_g = np.multiply(gd, a_g)
        c_b = np.multiply(bd, a_b)
        # 将三个通道的二维数组转换为Image对象
        # 这里必须将Image转换为灰度图像，再进行合并，否则会提示ValueError:mode mismatch
        ar = Image.fromarray(c_r).convert('L')
        ag = Image.fromarray(c_g).convert('L')
        ab = Image.fromarray(c_b).convert('L')
        # 合并图像
        img01 = Image.merge("RGB", (ar, ag, ab))
        # 将PIL.Image转换成OpenCV格式
        img02 = cv2.cvtColor(np.array(img01), cv2.COLOR_RGB2BGR)
        return img02

    def illumination_compensation_reference_white(self, image):
        """
        基于参考白算法的光照补偿
        :param image: 图像
        :return: 光照补偿后的图像
        """
        (b, g, r) = cv2.split(image)  # 分离三通道
        (x, y) = b.shape
        gray_list = []  # 灰度值列表
        for i in range(0, x):
            for j in range(0, y):
                gray_list.append(b[i][j])
                gray_list.append(g[i][j])
                gray_list.append(r[i][j])
        gray_list.sort(reverse=True)  # 从大到小排序
        num = int(x * y * 3 * 0.05)  # 计算前5%的像素个数
        total = 0  # 前5%的灰度值总和
        for i in range(0, num):
            total += gray_list[i]
        average = total / num  # 灰度平均值
        coe = 255 / average  # 光照补偿系数coe
        (b, g, r) = cv2.split(image)
        for i in range(0, x):
            for j in range(0, y):
                b[i][j] = int(b[i][j] * coe)
                g[i][j] = int(g[i][j] * coe)
                r[i][j] = int(r[i][j] * coe)
        img = cv2.merge([b, g, r])
        return img

    def crcb_range_sceening(self, image):
        """
        YCbCr色彩空间，Cb、Cr范围筛选，得出肤色区域的二值化图像
        :param image:
        :return:
        """
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
        (y, cr, cb) = cv2.split(ycrcb)
        skin = np.zeros(cr.shape, dtype=np.uint8)
        io = rw.SequenceTXTIO()
        value_range = io.read_sequence_from_txt("../TempInfo/Cb、Cr范围值.txt")
        (x, y) = cr.shape
        for i in range(0, x):
            for j in range(0, y):
                if (cr[i][j] > value_range[1][0]) and (cr[i][j] < value_range[1][1]) \
                        and (cb[i][j] > value_range[0][0]) and (cb[i][j] < value_range[0][1]):
                    skin[i][j] = 255
                else:
                    skin[i][j] = 0
        dst = skin
        return dst

    def skin_segmentation(self, image, mask):
        """
        图像肤色区域分割
        :param image: 基准图像
        :param mask: 二值化图像
        :return: 肤色区域分割后的图像
        """
        img = cv2.bitwise_and(image, image, mask=mask)
        return img

    def skin_detection(self, dialog, image):
        """
        默认肤色检测
        :param dialog: 弹窗
        :param image: 图像
        :return: 检测得到的三个图像组成的列表
        """
        dialog.set_text_lable("图像光照补偿中...")
        img01 = self.illumination_compensation_gray_world(image)    # 灰度世界
        # img01 = self.illumination_compensation_reference_white(image)   # 基于参考白
        # img01 = image

        dialog.set_text_lable("图像去噪中...")
        img02 = self.denoise_median_blur(img01)

        dialog.set_text_lable("图像肤色检测中...")
        img03 = self.crcb_range_sceening(img02)
        dialog.set_text_lable("图像二值化中...")
        time.sleep(1)

        dialog.set_text_lable("图像肤色区域分割中...")
        img04 = self.skin_segmentation(img02, img03)

        dialog.set_text_lable("图像检测完毕！")
        time.sleep(1)

        return [img02, img03, img04]

    def skin_likelihood(self, image):
        """
        生成肤色似然图且二值化
        :param image: 图像
        :return: None
        """
        img01 = self.illumination_compensation_gray_world(image)    # 光照补偿
        img02 = self.denoise_median_blur(img01)     # 去噪

        # 统计得出 Cb、Cr 值列表
        img03 = cv2.cvtColor(img02, cv2.COLOR_BGR2YCR_CB)
        (Y, Cr, Cb) = cv2.split(img03)  # 图像三通道分离
        (x, y) = Cr.shape
        list1 = []
        list2 = []
        for i in range(0, x):
            for j in range(0, y):
                list1.append(Cb[i][j])
                list2.append(Cr[i][j])

        # 读取模型数值
        io = rw.SequenceTXTIO()
        value_dict = io.read_sequence_from_txt("../TempInfo/模型计算所得值.txt")
        C = value_dict["Covariance_Matrix"]
        m = value_dict["Average_Vector"]
        C = C.replace("[", "").replace("]", "").replace("\n", "")
        m = m.replace("[", "").replace("]", "")
        list_C = C.split(" ")
        list_m = m.split(" ")
        del list_C[4]
        del list_C[0]
        # del list_m[2]
        # del list_m[1]
        C = np.array([
            [float(list_C[0]), float(list_C[1])],
            [float(list_C[2]), float(list_C[3])]
        ])
        m = np.array([float(list_m[0]), float(list_m[1])])

        # 开始计算
        statistics = data.StatisticsData()
        # 计算肤色似然率
        probability_list = statistics.calculate_probability(C, m, list1, list2)
        # 生成肤色似然图
        img04 = cv2.cvtColor(img02, cv2.COLOR_BGR2GRAY)
        (x, y) = img04.shape
        for i in range(0, x):
            for j in range(0, y):
                img04[i][j] = int(probability_list[i * y + j] * 255)
        # 开始生成二值化图像
        img05 = cv2.cvtColor(img02, cv2.COLOR_BGR2GRAY)
        (x, y) = img05.shape
        for i in range(0, x):
            for j in range(0, y):
                if probability_list[i * y + j] < 0.20:
                    img05[i][j] = 0
                else:
                    img05[i][j] = 255

        # 保存图片
        img04_temp = Image.fromarray(cv2.cvtColor(img04, cv2.COLOR_GRAY2RGB))  # 将传参过来的OpenCV图转换成PIL.Image格式
        img04_temp.save("../TempInfo/Skin Likelihood Image.jpg")
        img05_temp = Image.fromarray(cv2.cvtColor(img05, cv2.COLOR_GRAY2RGB))  # 将传参过来的OpenCV图转换成PIL.Image格式
        img05_temp.save("../TempInfo/Skin Likelihood Binarization Image.jpg")

        # 返回结果
        return probability_list

    def ellipse_detection(self, image):
        """
        椭圆模型肤色检测
        :param image: 图像
        :return: None
        """
        img01 = self.illumination_compensation_gray_world(image)

        img02 = self.denoise_median_blur(img01)

        img03 = self.crcb_ellipse_detection(img02)

        img04 = self.skin_segmentation(img02, img03)

        # 保存图片
        img03_temp = Image.fromarray(cv2.cvtColor(img03, cv2.COLOR_GRAY2RGB))  # 将传参过来的OpenCV图转换成PIL.Image格式
        img03_temp.save("../TempInfo/Skin Ellipse Detection Image_temp1.jpg")
        img04_temp = Image.fromarray(cv2.cvtColor(img04, cv2.COLOR_BGR2RGB))  # 将传参过来的OpenCV图转换成PIL.Image格式cv2.cvtColor(img04, cv2.COLOR_RBG2RGB)
        img04_temp.save("../TempInfo/Skin Ellipse Detection Image_temp2.jpg")

    def crcb_ellipse_detection(self, image):
        """
        在YCbCr颜色空间中采用椭圆模型对图像进行肤色检测
        :param image: 图像
        :return: 检测后的图像
        """
        skinCrCbHist = np.zeros((256, 256), dtype=np.uint8)  # 将返回给定形状和类型的新数组，用零填充。
        # 画椭圆——需要输入中心点位置，长轴和短轴的长度，椭圆沿逆时针选择角度，椭圆沿顺时针方向起始角度和结束角度
        img = cv2.ellipse(skinCrCbHist, (113, 155), (23, 15), 43, 0, 360, (255, 255, 255), -1)

        # 保存椭圆模型图
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))  # 将传参过来的OpenCV图转换成PIL.Image格式
        img.save("../TempInfo/ellipse_temp.jpg")

        # cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB) 色彩空间转换
        YCrCb = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
        (y, cr, cb) = cv2.split(YCrCb)  # 图像三通道分离
        skin = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        (x, y) = cr.shape
        for i in range(0, x):  # 利用椭圆皮肤模型进行皮肤检测
            for j in range(0, y):
                CR = YCrCb[i, j, 1]
                CB = YCrCb[i, j, 2]
                if skinCrCbHist[CR, CB] > 0:
                    skin[i, j] = 255
                else:
                    skin[i, j] = 0
        return skin

    def hsv_range_sceening(self, image):
        """
        HSV色彩空间，H、S、V范围筛选，得出肤色区域的二值化图像
        :param image: 图像
        :return: 二值化图像
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        (h, s, v) = cv2.split(hsv)
        skin = np.zeros(h.shape, dtype=np.uint8)
        (x, y) = h.shape

        for i in range(0, x):
            for j in range(0, y):
                if (h[i][j] > 7) and (h[i][j] < 20) and (s[i][j] > 28) and (s[i][j] < 255) and (v[i][j] > 50) and (
                        v[i][j] < 255):
                    skin[i][j] = 255
                else:
                    skin[i][j] = 0
        dst = skin
        return dst

    def hsv_detection(self, image):
        """
        HSV范围筛选肤色检测
        :param image: 图像
        :return: None
        """
        img01 = self.illumination_compensation_gray_world(image)

        img02 = self.denoise_median_blur(img01)

        img03 = self.hsv_range_sceening(img02)

        img04 = self.skin_segmentation(img02, img03)

        # 保存图片
        img03_temp = Image.fromarray(cv2.cvtColor(img03, cv2.COLOR_GRAY2RGB))  # 将传参过来的OpenCV图转换成PIL.Image格式
        img03_temp.save("../TempInfo/Skin HSV Detection Image_temp1.jpg")
        img04_temp = Image.fromarray(cv2.cvtColor(img04, cv2.COLOR_BGR2RGB))  # 将传参过来的OpenCV图转换成PIL.Image格式cv2.cvtColor(img04, cv2.COLOR_RBG2RGB)
        img04_temp.save("../TempInfo/Skin HSV Detection Image_temp2.jpg")

    def cr_otsu_detection(self, image):
        """
        YCrCb颜色空间的Cr分量+Otsu阈值分割
        :param image: 图像
        :return: None
        """
        img01 = self.illumination_compensation_gray_world(image)

        img02 = self.denoise_median_blur(img01)

        img03 = self.cr_otsu(img02)

        img04 = self.skin_segmentation(img02, img03)

        # 保存图片
        img03_temp = Image.fromarray(cv2.cvtColor(img03, cv2.COLOR_GRAY2RGB))  # 将传参过来的OpenCV图转换成PIL.Image格式
        img03_temp.save("../TempInfo/Skin Cr_Otsu Detection Image_temp1.jpg")
        img04_temp = Image.fromarray(cv2.cvtColor(img04, cv2.COLOR_BGR2RGB))  # 将传参过来的OpenCV图转换成PIL.Image格式cv2.cvtColor(img04, cv2.COLOR_RBG2RGB)
        img04_temp.save("../TempInfo/Skin Cr_Otsu Detection Image_temp2.jpg")

    def cr_otsu(self, image):
        """
        YCrCb颜色空间的Cr分量+Otsu阈值分割
        :param image: 图像
        :return: 检测出的二值化图像
        """
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)

        (y, cr, cb) = cv2.split(ycrcb)
        cr1 = cv2.GaussianBlur(cr, (5, 5), 0)
        _, skin = cv2.threshold(cr1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        dst = skin
        return dst

    def rgb(self, image):
        """
        RGB色彩空间肤色检测
        :param image: 图像
        :return: 二值化图像
        """
        (b, g, r) = cv2.split(image)
        skin = np.zeros(b.shape, dtype=np.uint8)
        (x, y) = b.shape
        for i in range(0, x):
            for j in range(0, y):
                R = int(r[i][j])
                G = int(g[i][j])
                B = int(b[i][j])
                if (abs(R - G) > 15) and (R > G) and (R > B):
                    if (R > 95) and (G > 40) and (B > 20) and (max(R, G, B) - min(R, G, B) > 15):
                        skin[i][j] = 255
                    elif (R > 220) and (G > 210) and (B > 170):
                        skin[i][j] = 255
                else:
                    skin[i][j] = 0
        dst = skin
        return dst

    def rgb_detection(self, image):
        """
        RGB颜色空间的肤色检测
        :param image: 图像
        :return: None
        """
        img01 = self.illumination_compensation_gray_world(image)

        img02 = self.denoise_median_blur(img01)

        img03 = self.rgb(img02)

        img04 = self.skin_segmentation(img02, img03)

        # 保存图片
        img03_temp = Image.fromarray(cv2.cvtColor(img03, cv2.COLOR_GRAY2RGB))  # 将传参过来的OpenCV图转换成PIL.Image格式
        img03_temp.save("../TempInfo/Skin RGB Detection Image_temp1.jpg")
        img04_temp = Image.fromarray(cv2.cvtColor(img04, cv2.COLOR_BGR2RGB))  # 将传参过来的OpenCV图转换成PIL.Image格式cv2.cvtColor(img04, cv2.COLOR_RBG2RGB)
        img04_temp.save("../TempInfo/Skin RGB Detection Image_temp2.jpg")
