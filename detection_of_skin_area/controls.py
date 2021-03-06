import wx
import wx.grid as grid
import wx.lib.buttons as buttons
from PIL import Image
import cv2
import time
import threading
import numpy as np
import webbrowser   # 默认浏览器打开网页
import pyperclip    # 剪切板模块
from detection_of_skin_area import image


class ImageView(wx.Panel):
    def __init__(self, parent, image=None, pos=wx.DefaultPosition, size=wx.DefaultSize, label="Unnamed"):
        """
        ImageView
        :param parent: 父控件
        :param image: 图片
        :param pos: 位置
        :param size: 大小
        :param label: 标题
        """
        image_temp = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image_temp.save("../TempInfo/image_temp_" + label + ".jpg")
        self.image = image
        self.label = label
        self.selected = True  # True代表本窗口被选中，False代表本窗口未被选中
        self.pattern = False   # False代表全屏模式，True代表列表模式
        wx.Panel.__init__(self, parent, pos=pos, size=size)
        self.SetBackgroundColour("white")

        # 初始化子控件
        width, heigth = self.GetSize()
        self.selected_box = wx.CheckBox(self, wx.ID_ANY, label=self.label, pos=(2, 3), size=(width - 57, 17),
                                       style=0, name="CheckBox")
        self.full_screen = buttons.GenBitmapButton(self, bitmap=wx.Bitmap("../images/full01.png"), size=(25, 25),
                                        pos=(width - 27, 0), name="FullScreen")
        self.download = buttons.GenBitmapButton(self, bitmap=wx.Bitmap("../images/download01.png"),
                                                size=(25, 25), pos=(width - 57, 0), name="Download")
        self.image_view = wx.StaticBitmap(self, wx.ID_ANY, bitmap=wx.Bitmap("../TempInfo/image_temp_" + label + ".jpg"), pos=(0, 0),
                                         size=(width, heigth), style=0, name="图片" + self.label)
        self.__set_button_properties(self.full_screen)
        self.__set_button_properties(self.download)
        self.selected_box.Hide()
        self.full_screen.Hide()
        self.download.Hide()
        self.SetBackgroundColour('white')
        self.image_view.SetBackgroundColour('white')
        self.__reset_image()
        self.Bind(wx.EVT_SIZE, self.OnSize, self)

    def __reset_image(self):
        """
        重置图片的大小
        :return: None
        """
        # self.image_view.SetSize(self.GetSize()[0], self.GetSize()[1])
        image = Image.fromarray(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))    # 将传参过来的OpenCV图转换成PIL.Image格式
        w, h = self.image_view.GetSize()
        image_view = image.resize((w, h))    # 修改图片大小
        image_view.save("../TempInfo/image_view.jpg")
        self.image_view.SetBitmap(wx.Bitmap("../TempInfo/image_view.jpg"))

    def OnSize(self, event):
        """
        当视图大小改变时，改变相应图片的位置及大小
        :param event:
        :return: None
        """
        self.selected_box.SetSize(self.GetSize()[0] - 57, 17)
        self.full_screen.SetPosition(wx.Point(self.GetSize()[0] - 27, 0))
        self.download.SetPosition(wx.Point(self.GetSize()[0] - 57, 0))
        if self.pattern is False:   # 全屏模式
            self.image_view.SetSize(self.GetSize()[0], self.GetSize()[1])
            self.image_view.SetPosition(wx.Point(0, 0))
            self.__reset_image()
        else:   # 列表模式
            self.image_view.SetSize(self.GetSize()[0], self.GetSize()[1] - 26)
            self.image_view.SetPosition(wx.Point(0, 26))
            self.__reset_image()

    def set_status(self, boolean):
        """
        将视图的状态设为给定状态
        :param boolean: 给定的状态
        :return: None
        """
        self.selected = boolean
        self.selected_box.SetValue(boolean)

    def get_status(self):
        """
        返回视图的状态
        :return: 视图的状态
        """
        return self.selected_box.GetValue()

    def __get_image_view_size_and_position(self):
        """
        根据图像比例，计算视图的尺寸和位置
        :return: 尺寸、位置
        """
        if self.selected:   # 全屏模式
            # 获取主窗口大小
            w, h = self.GetSize()
            img_w = self.image.shape[1]  # 获取到图片大小
            img_h = self.image.shape[0]
            view_w = 0
            view_h = 0
            if img_w > img_h:  # 图片长
                # 图片长，那么，长度为其限制
                view_w = w
                view_h = int(img_h * view_w / img_w)
                if view_h > h:  # 图片过高
                    view_h = h
                    view_w = int(img_w * view_h / img_h)
            else:  # 图片高
                # 图片高，那么，高度为其限制
                view_h = h
                view_w = int(img_w * view_h / img_h)
                if view_w > w:  # 图片过长
                    view_w = w
                    view_h = int(img_h * view_w / img_w)

            return view_w, view_h, (w - view_w) / 2, (h - view_h) / 2
        else:   # 列表模式
            pass

    def set_image(self, image):
        """
        重置图像
        :param image:
        :return:
        """
        self.image = image
        self.__reset_image()

    def set_pattern(self, boolean):
        self.pattern = boolean

    def conver(self):
        if self.pattern is False:   # 全屏模式
            self.selected_box.Hide()    # 将按钮等隐藏
            self.full_screen.Hide()
            self.download.Hide()
            self.image_view.SetSize(self.GetSize()[0], self.GetSize()[1])
            self.image_view.SetPosition(wx.Point(0, 0))
            self.__reset_image()
        else:       # 列表模式
            self.selected_box.Show()    # 将按钮等显示
            self.full_screen.Show()
            self.download.Show()
            self.image_view.SetSize(self.GetSize()[0], self.GetSize()[1] - 26)
            self.image_view.SetPosition(wx.Point(0, 26))
            self.__reset_image()

    def __set_button_properties(self, btn):
        """
        设置给定按钮的属性
        :param btn: 给定的按钮
        :return: None
        """
        btn.SetBackgroundColour('white')
        btn.SetBezelWidth(0)
        btn.SetUseFocusIndicator(False)
        font = wx.Font(16, wx.DECORATIVE, wx.NORMAL, wx.NORMAL)
        btn.SetFont(font)


class Loading(wx.Dialog):
    def __init__(self, parent=None, size=wx.DefaultSize):
        """
        初始化对话框
        :param parent: 父控件
        :param size: 大小
        """
        # 初始化窗口并将其位置居中
        wx.Dialog.__init__(self, parent, -1, size=size,
                           style=wx.FRAME_SHAPED | wx.SIMPLE_BORDER | wx.FRAME_NO_TASKBAR | wx.FRAME_FLOAT_ON_PARENT)
        self.Centre(wx.BOTH)

        self.panel = None   # 主面板
        self.image = None   # 加载动画
        self.text = None    # 静态文本
        # 加载动画路径
        self.imagePathList = ["../images/Loading_GIF/loading-01.jpg", "../images/Loading_GIF/loading-02.jpg",
                              "../images/Loading_GIF/loading-03.jpg", "../images/Loading_GIF/loading-04.jpg",
                              "../images/Loading_GIF/loading-05.jpg", "../images/Loading_GIF/loading-06.jpg",
                              "../images/Loading_GIF/loading-07.jpg", "../images/Loading_GIF/loading-08.jpg",
                              "../images/Loading_GIF/loading-09.jpg", "../images/Loading_GIF/loading-10.jpg",
                              "../images/Loading_GIF/loading-11.jpg", "../images/Loading_GIF/loading-12.jpg",
                              "../images/Loading_GIF/loading-13.jpg", "../images/Loading_GIF/loading-14.jpg"]
        self.i = 1  # 加载动画定位游标
        self.run = True  # 线程是否运行的标志，True代表线程运行，False代表线程结束
        self.color = 'white'    # 主题色彩

        # 设置对话框圆角
        # linux平台
        if wx.Platform == "__WXGTK__":
            self.Bind(wx.EVT_WINDOW_CREATE, self.set_balloon_shape)
        else:
            self.set_balloon_shape()

        # 设置对话框色彩及子控件
        self.set_attribute_and_child_controls()

        # 建立线程，刷新加载动画
        self.thread = threading.Thread(target=self.timing)
        self.thread.start()   # 启动线程

    def set_balloon_shape(self, event=None):
        """
        设置窗口圆角
        :param event: Linux平台需进行事件绑定
        :return: None
        """
        width, height = self.GetSize()
        bmp = wx.Bitmap(width, height)
        dc = wx.BufferedDC(None, bmp)
        dc.SetBackground(wx.Brush(wx.Colour(0, 0, 0), wx.SOLID))
        dc.Clear()
        dc.DrawRoundedRectangle(0, 0, width - 1, height - 1, 8)
        region = wx.Region(bmp, wx.Colour(0, 0, 0))
        self.SetShape(region)

    def set_attribute_and_child_controls(self):
        """
        设置对话框色彩及子控件
        :return: None
        """
        # 窗口背景色
        self.SetBackgroundColour(self.color)

        # 添加子控件并设置子控件属性
        width, height = self.GetSize()
        self.panel = wx.Panel(self, pos=(0, 0), size=(width, height), name="主面板")
        self.image = wx.StaticBitmap(self.panel, wx.ID_ANY, bitmap=wx.Bitmap(self.imagePathList[0]),
                                     pos=((width - 32) / 2, 54),  size=(32, 32), style=0, name="加载动画")
        self.text = wx.StaticText(self.panel, label="Loading...",
                                  pos=(5, 130), size=(width - 10, 20), name="加载文本", style=wx.ALIGN_CENTER)
        self.panel.SetBackgroundColour(self.color)
        self.set_text_attribute()

    def set_text_attribute(self):
        """
        设置静态文本的属性
        :return:None
        """
        self.text.SetBackgroundColour(self.color)
        font = wx.Font(16, wx.DECORATIVE, wx.NORMAL, wx.NORMAL)
        self.text.SetFont(font)

    def refresh_image(self):
        """
        改变加载动画的图片
        :return: None
        """
        self.image.SetBitmap(wx.Bitmap(self.imagePathList[self.i]))
        # 当一次循环结束时，重定位至开始位置，否则依次后移
        if self.i != len(self.imagePathList) - 1:
            self.i += 1
        else:
            self.i = 0

    def timing(self):
        """
        定时器，每100ms刷新一次加载动画
        :return: None
        """
        while self.run:
            self.refresh_image()
            time.sleep(0.1)  # 每个100ms改变一次图片

    def close(self):
        """
        关闭弹窗
        :return: None
        """
        self.Destroy()  # 销毁窗口
        self.run = False    # 结束线程

    def set_color(self, color):
        """
        设置弹窗的主题色彩
        :param color: 颜色
        :return: None
        """
        self.color = color
        self.panel.SetBackgroundColour(self.color)
        self.image.SetBackgroundColour(self.color)
        self.text.SetBackgroundColour(self.color)

    def set_text_lable(self, label):
        """
        设置文本内容
        :param label: 文本内容
        :return: None
        """
        self.text.SetLabelText(label)

    def set_text_color(self, color):
        """
        设置文本字体颜色
        :param color: 字体颜色
        :return: None
        """
        self.text.SetForegroundColour(color)

    def set_loading_image(self, image_path_list):
        """
        设置加载动画为给定路径列表所指向的动画序列
        :param image_path_list: GIF组图路径列表
        :return: None
        """
        self.imagePathList = image_path_list


class Tip(wx.Dialog):
    def __init__(self, parent=None, size=(200, 100), label="窗口暂时无法关闭！"):
        """
        初始化对话框
        :param parent: 父控件
        :param size: 大小
        """
        # 初始化窗口并将其位置居中
        wx.Dialog.__init__(self, parent, -1, size=size,
                           style=wx.FRAME_SHAPED | wx.SIMPLE_BORDER | wx.FRAME_NO_TASKBAR | wx.FRAME_FLOAT_ON_PARENT)
        self.Centre(wx.BOTH)

        self.panel = wx.Panel(self, size=size)

        self.text = wx.StaticText(self.panel, label=label, size=(size[0], 20), name="加载文本", style=wx.ALIGN_CENTER)

        # 设置对话框圆角
        # linux平台
        if wx.Platform == "__WXGTK__":
            self.Bind(wx.EVT_WINDOW_CREATE, self.set_balloon_shape)
        else:
            self.set_balloon_shape()

        # 设置窗口及控件属性
        self.set_control_attribute()

        self.Show()

        # 建立线程，稍后关闭
        thread_temp = threading.Thread(target=self.__tip)
        thread_temp.start()  # 启动线程

    def set_balloon_shape(self, event=None):
        """
        设置窗口圆角
        :param event: Linux平台需进行事件绑定
        :return: None
        """
        width, height = self.GetSize()
        bmp = wx.Bitmap(width, height)
        dc = wx.BufferedDC(None, bmp)
        dc.SetBackground(wx.Brush(wx.Colour(0, 0, 0), wx.SOLID))
        dc.Clear()
        dc.DrawRoundedRectangle(0, 0, width - 1, height - 1, 8)
        region = wx.Region(bmp, wx.Colour(0, 0, 0))
        self.SetShape(region)

    def set_control_attribute(self):
        """
        设置窗口及控件的属性
        :return:None
        """
        font = wx.Font(16, wx.DECORATIVE, wx.NORMAL, wx.NORMAL)
        self.text.SetFont(font)
        self.__set_transparent(200)
        self.SetBackgroundColour('black')
        self.panel.SetBackgroundColour('black')
        self.text.SetBackgroundColour('black')
        self.text.SetForegroundColour('white')
        self.text.Center(wx.BOTH)

    def __set_transparent(self, value):
        """
        设置窗口及控件的透明度
        :param value: 透明度
        :return: None
        """
        self.SetTransparent(value)
        self.panel.SetTransparent(value)
        self.text.SetTransparent(value)

    def __tip(self):
        """
        弹窗提示暂时无法关闭窗口
        :return: None
        """
        time.sleep(0.2)
        self.__set_transparent(160)
        time.sleep(0.1)
        self.__set_transparent(120)
        time.sleep(0.1)
        self.__set_transparent(80)
        time.sleep(0.1)
        self.__set_transparent(40)
        time.sleep(0.1)
        self.Destroy()


class ShowImage(wx.Frame):
    def __init__(self, parent=None, title="Unnamed", size=wx.DefaultSize):
        """
        初始化对话框
        :param parent: 父控件
        :param size: 大小
        """
        self.imagePathList = ["../images/Loading_GIF/loading-01.jpg", "../images/Loading_GIF/loading-02.jpg",
                              "../images/Loading_GIF/loading-03.jpg", "../images/Loading_GIF/loading-04.jpg",
                              "../images/Loading_GIF/loading-05.jpg", "../images/Loading_GIF/loading-06.jpg",
                              "../images/Loading_GIF/loading-07.jpg", "../images/Loading_GIF/loading-08.jpg",
                              "../images/Loading_GIF/loading-09.jpg", "../images/Loading_GIF/loading-10.jpg",
                              "../images/Loading_GIF/loading-11.jpg", "../images/Loading_GIF/loading-12.jpg",
                              "../images/Loading_GIF/loading-13.jpg", "../images/Loading_GIF/loading-14.jpg"]
        self.i = 1  # 加载动画定位游标
        self.run = True  # 线程是否运行的标志，True代表线程运行，False代表线程结束

        # 初始化属性
        self.title = title

        # 初始化窗口并将其位置居中
        wx.Frame.__init__(self, None, wx.ID_ANY, title=self.title, size=size, style=wx.CAPTION | wx.CLOSE_BOX | wx.MINIMIZE_BOX)
        self.panel = wx.Panel(self, pos=(0, 0), size=(self.GetSize()[0], self.GetSize()[1]), name="主面板")
        self.image = wx.StaticBitmap(self.panel, wx.ID_ANY, bitmap=wx.Bitmap(self.imagePathList[0]),
                                     pos=((self.panel.GetSize()[0] - 32) / 2, (self.panel.GetSize()[1] - 80) / 2), size=(32, 32), style=0, name="加载动画")
        self.image1 = wx.StaticBitmap(self.panel, wx.ID_ANY, bitmap=wx.Bitmap(self.imagePathList[0]),
                                    pos=(0, 0), size=(100, 100), style=0, name="显示图片")
        self.image2 = wx.StaticBitmap(self.panel, wx.ID_ANY, bitmap=wx.Bitmap(self.imagePathList[0]),
                                     pos=(100, 0), size=(100, 100), style=0, name="显示图片")
        self.image3 = wx.StaticBitmap(self.panel, wx.ID_ANY, bitmap=wx.Bitmap(self.imagePathList[0]),
                                      pos=(0, 100), size=(100, 100), style=0, name="显示图片")
        self.image4 = wx.StaticBitmap(self.panel, wx.ID_ANY, bitmap=wx.Bitmap(self.imagePathList[0]),
                                      pos=(100, 100), size=(100, 100), style=0, name="显示图片")
        self.grid = grid.Grid(self.panel, wx.ID_ANY, size=(self.panel.GetSize()[0], self.panel.GetSize()[1]))
        self.text = wx.StaticText(self.panel, label="NO TEXT", pos=(0, 0),
                                  size=(self.panel.GetSize()[0], self.panel.GetSize()[1]), name="TEXT",
                                style=wx.ALIGN_LEFT)
        self.img_l = wx.StaticBitmap(self.text, wx.ID_ANY, bitmap=wx.Bitmap("../images/parentheses_left.png"),
                                     pos=(218.5, 408), size=(71, 142), style=0, name="显示图片")
        self.img_r = wx.StaticBitmap(self.text, wx.ID_ANY, bitmap=wx.Bitmap("../images/parentheses_right.png"),
                                     pos=(465, 408), size=(71, 142), style=0, name="显示图片")
        self.icon = wx.StaticBitmap(self.panel, wx.ID_ANY, bitmap=wx.Bitmap("../images/icon03_64.ico"),
                               pos=((self.panel.GetSize()[0] - 664) / 2, 25), size=(64, 64), style=0, name="显示图片")
        self.title = wx.StaticText(self.panel, label="NO TEXT", pos=(self.icon.GetPosition()[0] + 64, 33), size=(600, 48), name="TEXT",
                                   style=wx.ALIGN_CENTER)
        self.divider = wx.Panel(self.panel, pos=(0, 95), size=(self.panel.GetSize()[0], 1), name="分割线")
        self.soft_info = wx.TextCtrl(self.panel, value="NO TEXT", pos=((self.panel.GetSize()[0] - 500) / 2, 200),
                                     size=(500, self.panel.GetSize()[1] - 350), name="TEXT",
                                     style=wx.TE_LEFT | wx.TE_MULTILINE | wx.TE_NO_VSCROLL | wx.BORDER_NONE)
        self.author_info = wx.TextCtrl(self.panel, value="NO TEXT", pos=(0, 0),
                                     size=(self.panel.GetSize()[0], self.panel.GetSize()[1]), name="TEXT",
                                     style=wx.TE_LEFT | wx.TE_MULTILINE | wx.TE_NO_VSCROLL | wx.BORDER_NONE | wx.TE_RICH2)
        self.author = wx.StaticBitmap(self.author_info, wx.ID_ANY, bitmap=wx.Bitmap("../images/AuthorPersonalPhoto.jpg"),
                                    pos=(35, 20), size=(170, 250), style=0, name="作者")
        self.package = wx.StaticBitmap(self, wx.ID_ANY, bitmap=wx.Bitmap("../images/package02.png"),
                                    pos=(25, 40), size=(128, 128), style=0, name="显示图片")
        self.btn1 = buttons.GenButton(self, id=201, label="在线查看", pos=(165, 50), size=(self.GetSize()[0] - 200, 40),
                                      style=wx.BORDER_SIMPLE)
        self.btn2 = buttons.GenButton(self, id=202, label="下载到本地", pos=(165, 118), size=(self.GetSize()[0] - 200, 40),
                                      style=wx.BORDER_SIMPLE)

        # 设置对话框色彩及子控件
        self.set_attribute_and_child_controls()

        # 建立线程，刷新加载动画
        self.thread = threading.Thread(target=self.timing)
        self.thread.start()  # 启动线程

    def set_attribute_and_child_controls(self):
        """
        设置对话框色彩及子控件
        :return: None
        """
        self.Centre(wx.BOTH)
        self.Bind(wx.EVT_CLOSE, self.OnCloseFrame, self)    # 关闭窗口前停掉线程
        # 窗口背景色
        self.SetBackgroundColour('white')
        # 主面板背景色
        self.panel.SetBackgroundColour('white')

        # 设置表格
        self.grid.EnableEditing(False)
        self.grid.EnableDragGridSize(False)
        self.grid.EnableDragColSize(False)
        self.grid.EnableDragRowSize(False)
        self.grid.SetDefaultCellAlignment(wx.ALIGN_RIGHT, wx.ALIGN_CENTRE)
        self.grid.SetColLabelSize(20)
        self.grid.SetRowLabelSize(50)
        self.grid.SetDefaultColSize(50)
        self.grid.SetDefaultRowSize(20)
        self.grid.SetScrollLineY(200)
        self.grid.SetScrollLineX(400)

        # 设置文本框
        self.text.SetBackgroundColour('white')
        self.set_text_attribute(self.text)
        self.title.SetBackgroundColour('white')
        self.divider.SetBackgroundColour('gray')
        self.set_text_attribute(self.soft_info)
        self.soft_info.SetEditable(False)
        self.set_text_attribute(self.author_info)
        self.author_info.SetEditable(False)
        self.author_info.Bind(wx.EVT_LEFT_DOWN, self.OnLink)
        self.__set_button_properties(self.btn1)
        self.__set_button_properties(self.btn2)
        self.Bind(wx.EVT_BUTTON, self.OnButton, self.btn1)
        self.Bind(wx.EVT_BUTTON, self.OnButton, self.btn2)

        self.image1.Hide()
        self.image2.Hide()
        self.image3.Hide()
        self.image4.Hide()
        self.text.Hide()
        self.grid.Hide()
        self.img_l.Hide()
        self.img_r.Hide()
        self.icon.Hide()
        self.title.Hide()
        self.divider.Hide()
        self.soft_info.Hide()
        self.author_info.Hide()
        self.author.Hide()
        self.package.Hide()
        self.btn1.Hide()
        self.btn2.Hide()

    def timing(self):
        """
        定时器，每100ms刷新一次加载动画
        :return: None
        """
        while self.run:
            self.refresh_image()
            time.sleep(0.1)  # 每个100ms改变一次图片

    def close(self):
        """
        关闭弹窗
        :return: None
        """
        self.image.Hide()
        self.run = False  # 结束线程

    def refresh_image(self):
        """
        改变加载动画的图片
        :return: None
        """
        self.image.SetBitmap(wx.Bitmap(self.imagePathList[self.i]))
        # 当一次循环结束时，重定位至开始位置，否则依次后移
        if self.i != len(self.imagePathList) - 1:
            self.i += 1
        else:
            self.i = 0

    def set_icon(self, icon_path):
        """
        设置窗口的图标
        :param icon_path:
        :return:
        """
        icon = wx.Icon()
        icon.LoadFile(icon_path, wx.BITMAP_TYPE_ICO)
        self.SetIcon(icon)

    def set_show_image(self, image_paths, frame):
        """
        显示图片
        :return: None
        """
        if len(image_paths) == 1:
            size, pos = self.__get_size_and_pos(1, image_paths)
            self.image1.SetSize(size[0], size[1])
            self.image1.SetPosition(wx.Point(pos[0][0], pos[0][1]))
            self.image1.SetBitmap(wx.Bitmap(image_paths[0][0:(len(image_paths[0]) - 4)] + "_temp.jpg"))

            # 关闭等待图，显示图片
            frame.close()
            self.image1.Show()
        elif len(image_paths) == 2:
            size, pos = self.__get_size_and_pos(2, image_paths)
            self.image1.SetSize(size[0], size[1])
            self.image1.SetPosition(wx.Point(pos[0][0], pos[0][1]))
            self.image1.SetBitmap(wx.Bitmap(image_paths[0][0:(len(image_paths[0]) - 4)] + "_temp.jpg"))
            self.image2.SetSize(size[0], size[1])
            self.image2.SetPosition(wx.Point(pos[1][0], pos[1][1]))
            self.image2.SetBitmap(wx.Bitmap(image_paths[1][0:(len(image_paths[1]) - 4)] + "_temp.jpg"))

            # 关闭等待图，显示图片
            frame.close()
            self.image1.Show()
            self.image2.Show()
        elif len(image_paths) == 4:
            size, pos = self.__get_size_and_pos(4, image_paths)
            self.image1.SetSize(size[0], size[1])
            self.image1.SetPosition(wx.Point(pos[0][0], pos[0][1]))
            self.image1.SetBitmap(wx.Bitmap(image_paths[0][0:(len(image_paths[0]) - 4)] + "_temp.jpg"))
            self.image2.SetSize(size[0], size[1])
            self.image2.SetPosition(wx.Point(pos[1][0], pos[1][1]))
            self.image2.SetBitmap(wx.Bitmap(image_paths[1][0:(len(image_paths[1]) - 4)] + "_temp.jpg"))
            self.image3.SetSize(size[0], size[1])
            self.image3.SetPosition(wx.Point(pos[2][0], pos[2][1]))
            self.image3.SetBitmap(wx.Bitmap(image_paths[2][0:(len(image_paths[2]) - 4)] + "_temp.jpg"))
            self.image4.SetSize(size[0], size[1])
            self.image4.SetPosition(wx.Point(pos[3][0], pos[3][1]))
            self.image4.SetBitmap(wx.Bitmap(image_paths[3][0:(len(image_paths[3]) - 4)] + "_temp.jpg"))
            # 关闭等待图，显示图片
            frame.close()
            self.image1.Show()
            self.image2.Show()
            self.image3.Show()
            self.image4.Show()

    def __get_size_and_pos(self, num, image_paths):
        """
        根据面板大小及图片数量、大小，计算出位置和大小
        :param num: 图片数量
        :param image_paths: 图片路径
        :return: 大小、位置列表
        """
        image = cv2.imdecode(np.fromfile(image_paths[0], dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if num == 1:
            w, h = self.panel.GetSize()
            # 获取图片大小
            img_w = image.shape[1]  # 获取到图片大小
            img_h = image.shape[0]
            view_w = 0
            view_h = 0
            if img_w > img_h:  # 图片长
                # 图片长，那么，长度为其限制
                view_w = w
                view_h = int(img_h * view_w / img_w)
                if view_h > h:  # 图片过高
                    view_h = h
                    view_w = int(img_w * view_h / img_h)
            else:  # 图片高
                # 图片高，那么，高度为其限制
                view_h = h
                view_w = int(img_w * view_h / img_h)
                if view_w > w:  # 图片过长
                    view_w = w
                    view_h = int(img_h * view_w / img_w)
            size = (view_w, view_h)
            pos_temp = (int((w - view_w) / 2), int((h - view_h) / 2))
            pos = []
            pos += [pos_temp]
            img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # 将传参过来的OpenCV图转换成PIL.Image格式
            img_temp = img.resize((size[0], size[1]))  # 修改图片大小
            img_temp.save(image_paths[0][0:(len(image_paths[0]) - 4)] + "_temp.jpg")
            return size, pos
        elif num == 2:
            pos = []
            w = int((self.panel.GetSize()[0] - 15) / 2)
            h = self.panel.GetSize()[1]
            # 获取图片大小
            img_w = image.shape[1]  # 获取到图片大小
            img_h = image.shape[0]
            view_w = 0
            view_h = 0
            if img_w > img_h:  # 图片长
                # 图片长，那么，长度为其限制
                view_w = w
                view_h = int(img_h * view_w / img_w)
                if view_h > h:  # 图片过高
                    view_h = h
                    view_w = int((img_w - 26) * view_h / img_h)
            else:  # 图片高
                # 图片高，那么，高度为其限制
                view_h = h
                view_w = int(img_w * view_h / img_h)
                if view_w > w:  # 图片过长
                    view_w = w
                    view_h = int(img_h * view_w / img_w)

            size = (view_w, view_h)
            for i in range(0, 2):
                pos_temp = ((w - view_w) / 2 + (i + 1) * 5 + i * w, (h - view_h) / 2 + 5)
                pos += [pos_temp]
            img1 = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # 将传参过来的OpenCV图转换成PIL.Image格式
            img_temp1 = img1.resize((size[0], size[1]))  # 修改图片大小
            img_temp1.save(image_paths[0][0:(len(image_paths[0]) - 4)] + "_temp.jpg")
            image2 = cv2.imdecode(np.fromfile(image_paths[1], dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            img2 = Image.fromarray(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))  # 将传参过来的OpenCV图转换成PIL.Image格式
            img_temp2 = img2.resize((size[0], size[1]))  # 修改图片大小
            img_temp2.save(image_paths[1][0:(len(image_paths[1]) - 4)] + "_temp.jpg")
            return size, pos
        elif num == 4:
            pos = []
            w = int((self.panel.GetSize()[0] - 15) / 2)
            h = int((self.panel.GetSize()[1] - 15) / 2)
            # 获取图片大小
            img_w = image.shape[1]  # 获取到图片大小
            img_h = image.shape[0]
            view_w = 0
            view_h = 0
            if img_w > img_h:  # 图片长
                # 图片长，那么，长度为其限制
                view_w = w
                view_h = int(img_h * view_w / img_w)
                if view_h > h:  # 图片过高
                    view_h = h
                    view_w = int((img_w - 26) * view_h / img_h)
            else:  # 图片高
                # 图片高，那么，高度为其限制
                view_h = h
                view_w = int(img_w * view_h / img_h)
                if view_w > w:  # 图片过长
                    view_w = w
                    view_h = int(img_h * view_w / img_w)

            size = (view_w, view_h)
            for i in range(0, 4):
                if i < 2:
                    pos_temp = ((w - view_w) / 2 + (i + 1) * 5 + i * w, (h - view_h) / 2 + 5)
                else:
                    i1 = i % 2
                    pos_temp = ((w - view_w) / 2 + (i1 + 1) * 5 + i1 * w, (h - view_h) / 2 + h + 10)
                pos += [pos_temp]
            for i in range(0, 4):
                image = cv2.imdecode(np.fromfile(image_paths[i], dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # 将传参过来的OpenCV图转换成PIL.Image格式
                img_temp = img.resize((size[0], size[1]))  # 修改图片大小
                img_temp.save(image_paths[i][0:(len(image_paths[i]) - 4)] + "_temp.jpg")
            return size, pos
        else:
            return None, None

    def show_skin_probability(self, probability_list, image, frame):
        """
        显示肤色似然概率
        :param probability_list:
        :param image:
        :return:
        """
        img01 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        (x, y) = img01.shape
        self.grid.CreateGrid(x, y)
        for i in range(0, y):
            self.grid.SetColLabelValue(i, str(i + 1))
        for i in range(0, x):
            for j in range(0, y):
                value = int(probability_list[i * y + j] * 100)
                self.grid.SetCellValue(i, j, str(value))

        # 关闭等待图，显示概率
        frame.close()
        self.grid.Show()

    def OnCloseFrame(self, event):
        """
        关闭窗口之前将动态图停掉
        :param event: 事件源
        :return: None
        """
        # 原方案，但会引起线程无法全部关闭的异常
        # if self.run:    # 线程还在运行
        #     self.close()    # 停掉线程
        # self.Destroy()  # 销毁窗体

        # 方案1，在线程运行完之前无法关闭窗口
        if self.run is False:   # 如果线程已经不再运行，关闭窗口
            self.Destroy()
        else:       # 如果线程还在运行，弹窗提示窗口无法关闭
            # 弹窗提示
            Tip(size=(200, 100), label="窗口暂时无法关闭！")

    def set_show_value(self, value, frame):
        """
        显示模型数值
        :return: None
        """
        # 取值
        C = value["Covariance_Matrix"]
        m = value["Average_Vector"]
        C = C.replace("[", "").replace("]", "").replace("\n", "")
        m = m.replace("[", "").replace("]", "")
        list_C = C.split(" ")
        list_m = m.split(" ")
        # 直接删除
        del list_C[5]
        del list_C[2]
        del list_C[0]
        # print(list_C)
        self.text.SetLabelText("\n均值(Cb):\t" + str(round(value['Average_X'], 1)) + "\t    均值(Cr):\t" +
                               str(round(value['Average_Y'], 1)) + "\n\n方差(Cb):\t" + str(round(value['Variance_X'], 1))
                               + "\t    方差(Cr):\t" + str(round(value['Variance_Y'], 1)) +
                               "\n\n标准差(Cb):\t" + str(round(value['Standard_Deviation_X'], 1)) +
                               "\t    标准差(Cr):\t" + str(round(value['Standard_Deviation_Y'], 1)) +
                               "\n\n期望E(Cb):\t" + str(round(value['E_X'], 1)) +
                               "\t    期望E(Cr):\t" + str(round(value['E_Y'], 1)) +
                               "\n\n期望E(CbCr):\t" + str(round(value['E_XY'], 1)) +
                               "\n\n均值向量m:\t(" + str(round(float(list_m[0]), 1)) +
                               ",\t" + str(round(float(list_m[1]), 1)) +
                               ")\n\n\t\t  " + str(round(float(list_C[0]), 1)) +
                               "\t" + str(round(float(list_C[1]), 1)) +
                               "\n协方差矩阵C：\n\t\t  " + str(round(float(list_C[2]), 1)) +
                               "\t" + str(round(float(list_C[3]), 1)) + "")

        # 关闭等待图，显示数值
        frame.close()
        self.text.Show()
        self.img_l.Show()
        self.img_r.Show()

    def set_text_attribute(self, control):
        """
        设置静态文本的属性
        :return:None
        """
        control.SetBackgroundColour('white')
        font = wx.Font(24, wx.DECORATIVE, wx.NORMAL, wx.NORMAL)
        control.SetFont(font)

    def destory(self):
        """
        销毁此窗口
        :return: None
        """
        self.Destroy()

    def set_about(self, type, frame):
        """
        设置Web文档界面
        :param type: 控件类型
        :return: None
        """
        if type == 1:   # 关于软件
            self.icon.SetBackgroundColour('white')
            self.title.SetForegroundColour('#1296DB')
            self.title.SetBackgroundColour('white')
            font = wx.Font(36, wx.DECORATIVE, wx.NORMAL, wx.NORMAL)
            self.title.SetFont(font)
            self.title.SetLabelText("彩色图像肤色区域检测系统")
            text = "作者：\t\t王博\n版本：\t\t0.1.0\n更新日期：\t2019-04-22\n反馈邮箱：\twsfcmn@163.com"
            self.soft_info.SetValue(text)

            # 关闭等待图，显示软件信息
            frame.close()
            self.icon.Show()
            self.title.Show()
            self.divider.Show()
            self.soft_info.Show()
        elif type == 2:     # 关于作者
            # 显示作者信息
            text = "\n\t\t\t\t\t\t姓名：\t王博\n\n\t\t\t\t\t\t学校：\t"
            self.author_info.SetValue(text)
            self.author_info.SetDefaultStyle(wx.TextAttr("#1296DB"))
            self.author_info.AppendText("西安理工大学")
            text = "\n\n\t\t\t\t\t\t邮箱：\twsfcmn@163.com\n\n    GitHub：\t\t"
            self.author_info.SetDefaultStyle(wx.TextAttr(wx.BLACK))
            self.author_info.AppendText(text)
            self.author_info.SetDefaultStyle(wx.TextAttr("#1296DB"))
            self.author_info.AppendText("https://github.com/SFCMN")

            # 关闭等待图，显示作者信息
            frame.close()
            self.author_info.Show()
            self.author.Show()
        else:       # 源码
            # 关闭等待图，显示源码
            frame.close()
            self.panel.Hide()
            self.package.Show()
            self.btn1.Show()
            self.btn2.Show()

    def OnLink(self, event):
        """
        鼠标点击打开网站
        :param event: 事件源
        :return: None
        """
        (x, y) = event.GetPosition()    # 获取当前鼠标位置
        # print((x, y))
        if x >= 430 and x <= 650 and y >= 230 and y<= 250:
            pyperclip.copy("wsfcmn@163.com")
            Tip(size=(250, 100), label="邮箱已复制到剪切板！")
        if x >= 440 and x <= 620 and y >= 140 and y<= 160:
            webbrowser.open("http://www.xaut.edu.cn/")
        if x >= 290 and x <= 670 and y >= 310 and y<= 330:
            webbrowser.open("https://github.com/SFCMN")

    def __set_button_properties(self, btn):
        """
        设置给定按钮的属性
        :param btn: 给定的按钮
        :return: None
        """
        btn.SetBackgroundColour('white')
        btn.SetTransparent(200)
        btn.SetBezelWidth(0)
        btn.SetUseFocusIndicator(False)
        font = wx.Font(16, wx.DECORATIVE, wx.NORMAL, wx.NORMAL)
        btn.SetFont(font)

    def OnButton(self, event):
        """
        点击按钮跳转到源码网页或下载到本地
        :param event: 事件源
        :return: None
        """
        id = event.GetId()
        if id == 201:       # 在线查看
            webbrowser.open("https://github.com/SFCMN/DetectionOfSkinAreas")
        else:               # 下载到本地
            # 使用浏览器下载源码
            webbrowser.open("https://github.com/SFCMN/DetectionOfSkinAreas/archive/master.zip")


class MultiSchemeFrame(wx.Frame):

    def __init__(self, image, parent=None, title="Unnamed", size=(350, 300)):
        self.image = image
        self.title = title

        # 初始化窗口并将其位置居中
        wx.Frame.__init__(self, None, wx.ID_ANY, title=self.title, size=size,
                          style=wx.CAPTION | wx.CLOSE_BOX | wx.MINIMIZE_BOX)
        self.__set_icon("../images/compare02.ico")
        self.Center()
        self.panel = wx.Panel(self, pos=(0, 0), size=(self.GetSize()[0], self.GetSize()[1]), name="主面板")
        self.panel.SetBackgroundColour('white')
        self.btn1 = buttons.GenButton(self.panel, label='不同色彩空间', pos=(18, 20), size=(300, 40),
                                      style=wx.BORDER_SIMPLE)
        self.btn2 = buttons.GenButton(self.panel, label='不同预处理算法', pos=(18, 80), size=(300, 40),
                                      style=wx.BORDER_SIMPLE)
        self.btn3 = buttons.GenButton(self.panel, label='不同检测方法(YCbCr空间)', pos=(18, 140), size=(300, 40),
                                      style=wx.BORDER_SIMPLE)
        self.btn4 = buttons.GenButton(self.panel, label='不同Cb、Cr值范围', pos=(18, 200), size=(300, 40),
                                      style=wx.BORDER_SIMPLE)

        self.background_panel = wx.Panel(self.panel, pos=(340, 20), size=(330, 220), name="背景面板")
        self.background_panel.SetBackgroundColour('#ADD8E6')
        self.background_panel.Hide()

        self.__set_button_properties(self.btn1)
        self.__set_button_properties(self.btn2)
        self.__set_button_properties(self.btn3)
        self.__set_button_properties(self.btn4)

        self.Bind(wx.EVT_BUTTON, self.OnClick1, self.btn1)
        self.Bind(wx.EVT_BUTTON, self.OnClick2, self.btn2)
        self.Bind(wx.EVT_BUTTON, self.OnClick3, self.btn3)
        self.Bind(wx.EVT_BUTTON, self.OnClick4, self.btn4)

    def __set_icon(self, icon_path):
        """
        设置窗口的图标
        :param icon_path:
        :return:
        """
        icon = wx.Icon()
        icon.LoadFile(icon_path, wx.BITMAP_TYPE_ICO)
        self.SetIcon(icon)

    def __set_button_properties(self, btn):
        """
        设置给定按钮的属性
        :param btn: 给定的按钮
        :return: None
        """
        btn.SetBackgroundColour('white')
        btn.SetTransparent(200)
        btn.SetBezelWidth(0)
        btn.SetUseFocusIndicator(False)
        font = wx.Font(16, wx.DECORATIVE, wx.NORMAL, wx.NORMAL)
        btn.SetFont(font)

    def __set_button_properties2(self, btn):
        """
        设置给定按钮的属性
        :param btn: 给定的按钮
        :return: None
        """
        self.__set_button_properties(btn)
        btn.SetBackgroundColour('#D5F0EF')

    def __set_button_properties3(self, btn):
        """
        设置给定按钮的属性
        :param btn: 给定的按钮
        :return: None
        """
        self.__set_button_properties2(btn)
        font = wx.Font(14, wx.DECORATIVE, wx.NORMAL, wx.NORMAL)
        btn.SetFont(font)

    def __set_button_properties4(self, btn, bool):
        """
        设置给定按钮的属性
        :param btn: 给定的按钮
        :return: None
        """
        self.__set_button_properties(btn)
        if bool:
            btn.SetBackgroundColour('#F0E68C')
            btn.SetForegroundColour('red')
        else:
            btn.SetBackgroundColour('#D5F0EF')
            btn.SetForegroundColour('black')

    def __set_background(self, btn):
        self.background_panel.DestroyChildren()
        self.btn1.SetBackgroundColour('white')
        self.btn2.SetBackgroundColour('white')
        self.btn3.SetBackgroundColour('white')
        self.btn4.SetBackgroundColour('white')
        btn.SetBackgroundColour('#D5F0EF')
        self.update()

    def update(self):
        self.panel.Hide()
        self.panel.Show()

    def __reset_size(self):
        """
        重新设置窗口大小及位置
        :return: None
        """
        self.SetSize(700, 300)
        self.background_panel.Show()
        self.Center()

    def OnClick1(self, event):
        """
        按钮点击事件
        :param event: 事件源
        :return: None
        """
        self.__set_background(self.btn1)
        self.__reset_size()
        btn1 = buttons.GenButton(self.background_panel, id=101, label='RGB色彩空间', pos=(15, 30), size=(300, 40),
                                      style=wx.BORDER_SIMPLE)
        btn2 = buttons.GenButton(self.background_panel, id=102, label='HSV色彩空间', pos=(15, 90), size=(300, 40),
                                 style=wx.BORDER_SIMPLE)
        btn3 = buttons.GenButton(self.background_panel, id=103, label='YCbCr色彩空间', pos=(15, 150), size=(300, 40),
                                 style=wx.BORDER_SIMPLE)
        self.__set_button_properties2(btn1)
        self.__set_button_properties2(btn2)
        self.__set_button_properties2(btn3)

        self.Bind(wx.EVT_BUTTON, self.OnClick11, btn1)
        self.Bind(wx.EVT_BUTTON, self.OnClick11, btn2)
        self.Bind(wx.EVT_BUTTON, self.OnClick11, btn3)

    def OnClick2(self, event):
        """
        按钮点击事件
        :param event: 事件源
        :return: None
        """
        self.__set_background(self.btn2)
        self.__reset_size()
        btn_light = buttons.GenButton(self.background_panel, id=2001, label='光照补偿', pos=(15, 10), size=(135, 40),
                                 style=wx.BORDER_SIMPLE)
        btn_denoise = buttons.GenButton(self.background_panel, id=2002, label='去噪', pos=(180, 10), size=(135, 40),
                                 style=wx.BORDER_SIMPLE)

        btn1 = buttons.GenButton(self.background_panel, id=201, label='不进行光照补偿', pos=(15, 70), size=(300, 40),
                                 style=wx.BORDER_SIMPLE)
        btn2 = buttons.GenButton(self.background_panel, id=202, label='GrayWorld色彩均衡算法', pos=(15, 120), size=(300, 40),
                                 style=wx.BORDER_SIMPLE)
        btn3 = buttons.GenButton(self.background_panel, id=203, label='基于参考白的算法', pos=(15, 170), size=(300, 40),
                                 style=wx.BORDER_SIMPLE)
        btn4 = buttons.GenButton(self.background_panel, id=204, label='中值滤波', pos=(15, 90), size=(135, 40),
                                 style=wx.BORDER_SIMPLE)
        btn5 = buttons.GenButton(self.background_panel, id=205, label='均值滤波', pos=(180, 90), size=(135, 40),
                                 style=wx.BORDER_SIMPLE)
        btn6 = buttons.GenButton(self.background_panel, id=206, label='高斯滤波', pos=(15, 150), size=(135, 40),
                                 style=wx.BORDER_SIMPLE)
        btn7 = buttons.GenButton(self.background_panel, id=207, label='双边滤波', pos=(180, 150), size=(135, 40),
                                 style=wx.BORDER_SIMPLE)

        # 按钮样式设置
        self.__set_button_properties2(btn_light)
        self.__set_button_properties2(btn_denoise)
        self.__set_button_properties2(btn1)
        self.__set_button_properties2(btn2)
        self.__set_button_properties2(btn3)
        self.__set_button_properties2(btn4)
        self.__set_button_properties2(btn5)
        self.__set_button_properties2(btn6)
        self.__set_button_properties2(btn7)

        # 按钮事件绑定
        self.Bind(wx.EVT_BUTTON, self.OnClick201, btn_light)
        self.Bind(wx.EVT_BUTTON, self.OnClick201, btn_denoise)
        self.Bind(wx.EVT_BUTTON, self.OnClick21, btn1)
        self.Bind(wx.EVT_BUTTON, self.OnClick21, btn2)
        self.Bind(wx.EVT_BUTTON, self.OnClick21, btn3)
        self.Bind(wx.EVT_BUTTON, self.OnClick21, btn4)
        self.Bind(wx.EVT_BUTTON, self.OnClick21, btn5)
        self.Bind(wx.EVT_BUTTON, self.OnClick21, btn6)
        self.Bind(wx.EVT_BUTTON, self.OnClick21, btn7)

        # 后4个按钮隐藏
        btn4.Hide()
        btn5.Hide()
        btn6.Hide()
        btn7.Hide()

        # 光照补偿按钮点亮
        self.__set_button_properties4(btn_light, True)

    def OnClick3(self, event):
        """
        按钮点击事件
        :param event: 事件源
        :return: None
        """
        self.__set_background(self.btn3)
        self.__reset_size()
        btn1 = buttons.GenButton(self.background_panel, id=301, label='Cb、Cr值范围筛选法', pos=(15, 30), size=(300, 40),
                                 style=wx.BORDER_SIMPLE)
        btn2 = buttons.GenButton(self.background_panel, id=302, label='椭圆肤色模型检测法', pos=(15, 90), size=(300, 40),
                                 style=wx.BORDER_SIMPLE)
        btn3 = buttons.GenButton(self.background_panel, id=303, label='Cr分量+Otsu阈值分割法', pos=(15, 150), size=(300, 40),
                                 style=wx.BORDER_SIMPLE)
        self.__set_button_properties2(btn1)
        self.__set_button_properties2(btn2)
        self.__set_button_properties2(btn3)

        self.Bind(wx.EVT_BUTTON, self.OnClick31, btn1)
        self.Bind(wx.EVT_BUTTON, self.OnClick31, btn2)
        self.Bind(wx.EVT_BUTTON, self.OnClick31, btn3)

    def OnClick4(self, event):
        """
        按钮点击事件
        :param event: 事件源
        :return: None
        """
        self.__set_background(self.btn4)
        self.__reset_size()
        btn1 = buttons.GenButton(self.background_panel, id=401, label='公认:Cb∈[77,127],Cr∈[133，173]', pos=(5, 30), size=(320, 40),
                                 style=wx.BORDER_SIMPLE)
        btn2 = buttons.GenButton(self.background_panel, id=402, label='学者:Cb∈[90，135],Cr∈[137,167]', pos=(5, 90), size=(320, 40),
                                 style=wx.BORDER_SIMPLE)
        btn3 = buttons.GenButton(self.background_panel, id=403, label='作者:Cb∈[94,126],Cr∈[133，165]', pos=(5, 150), size=(320, 40),
                                 style=wx.BORDER_SIMPLE)
        self.__set_button_properties3(btn1)
        self.__set_button_properties3(btn2)
        self.__set_button_properties3(btn3)

        self.Bind(wx.EVT_BUTTON, self.OnClick41, btn1)
        self.Bind(wx.EVT_BUTTON, self.OnClick41, btn2)
        self.Bind(wx.EVT_BUTTON, self.OnClick41, btn3)

    def OnClick11(self, event):
        """
        按钮点击事件
        :param event: 事件源
        :return: None
        """
        id = event.GetId()
        if id == 101:
            compare = image.ImageCompare()
            compare.rgb(self.image)
            frame = ShowImage2(num=3, title="RGB色彩空间肤色检测",
                               title_list=["原图像", "肤色检测结果图像", "二值化图像"],
                               image_path_list=["../TempInfo/rgb_compare_1.jpg", "../TempInfo/rgb_compare_2.jpg",
                                                "../TempInfo/rgb_compare_3.jpg"])
            frame.set_icon("../images/likelihoodimage01.ico")
        elif id == 102:
            compare = image.ImageCompare()
            compare.hsv(self.image)
            frame = ShowImage2(num=3, title="HSV色彩空间肤色检测",
                               title_list=["原图像", "肤色检测结果图像", "二值化图像"],
                               image_path_list=["../TempInfo/hsv_compare_1.jpg", "../TempInfo/hsv_compare_2.jpg",
                                                "../TempInfo/hsv_compare_3.jpg"])
            frame.set_icon("../images/likelihoodimage01.ico")
        else:
            compare = image.ImageCompare()
            compare.ycbcr(self.image)
            frame = ShowImage2(num=3, title="YCbCr色彩空间肤色检测",
                               title_list=["原图像", "肤色检测结果图像", "二值化图像"],
                               image_path_list=["../TempInfo/ycbcr_compare_1.jpg", "../TempInfo/ycbcr_compare_2.jpg",
                                                "../TempInfo/ycbcr_compare_3.jpg"])
            frame.set_icon("../images/likelihoodimage01.ico")

    def OnClick201(self, event):
        """
        按钮点击事件
        :param event: 事件源
        :return: None
        """
        id = event.GetId()
        btn_list = self.background_panel.GetChildren()
        if id == 2001:
            self.__set_button_properties4(btn_list[0], True)
            self.__set_button_properties4(btn_list[1], False)
            for i in range(2, 9):
                if i < 5:
                    btn_list[i].Show()
                else:
                    btn_list[i].Hide()
        elif id == 2002:
            self.__set_button_properties4(btn_list[0], False)
            self.__set_button_properties4(btn_list[1], True)
            for i in range(2, 9):
                if i < 5:
                    btn_list[i].Hide()
                else:
                    btn_list[i].Show()
        self.update()

    def OnClick21(self, event):
        """
        按钮点击事件
        :param event: 事件源
        :return: None
        """
        id = event.GetId()
        if id == 201:
            compare = image.ImageCompare()
            compare.nonelight(self.image)
            frame = ShowImage2(num=4, title="不进行光照补偿肤色检测",
                               title_list=["原图像", "未进行光照补偿的图像", "肤色检测结果图像", "二值化图像"],
                               image_path_list=["../TempInfo/nonelight_compare_1.jpg", "../TempInfo/nonelight_compare_2.jpg",
                                                "../TempInfo/nonelight_compare_3.jpg", "../TempInfo/nonelight_compare_4.jpg"])
            frame.set_icon("../images/likelihoodimage01.ico")
        elif id == 202:
            compare = image.ImageCompare()
            compare.grayworld(self.image)
            frame = ShowImage2(num=4, title="GrayWorld色彩均衡算法肤色检测",
                               title_list=["原图像", "GrayWorld光照补偿图像", "肤色检测结果图像", "二值化图像"],
                               image_path_list=["../TempInfo/grayworld_compare_1.jpg", "../TempInfo/grayworld_compare_2.jpg",
                                                "../TempInfo/grayworld_compare_3.jpg", "../TempInfo/grayworld_compare_4.jpg"])
            frame.set_icon("../images/likelihoodimage01.ico")
        elif id == 203:
            compare = image.ImageCompare()
            compare.referencewhite(self.image)
            frame = ShowImage2(num=4, title="基于参考白的算法肤色检测",
                               title_list=["原图像", "基于参考白光照补偿图像", "肤色检测结果图像", "二值化图像"],
                               image_path_list=["../TempInfo/referencewhite_compare_1.jpg",
                                                "../TempInfo/referencewhite_compare_2.jpg",
                                                "../TempInfo/referencewhite_compare_3.jpg",
                                                "../TempInfo/referencewhite_compare_4.jpg"])
            frame.set_icon("../images/likelihoodimage01.ico")
        elif id == 204:
            compare = image.ImageCompare()
            compare.medianblur(self.image)
            frame = ShowImage2(num=4, title="中值滤波肤色检测",
                               title_list=["原图像", "中值滤波图像", "肤色检测结果图像", "二值化图像"],
                               image_path_list=["../TempInfo/medianblur_compare_1.jpg",
                                                "../TempInfo/medianblur_compare_2.jpg",
                                                "../TempInfo/medianblur_compare_3.jpg",
                                                "../TempInfo/medianblur_compare_4.jpg"])
            frame.set_icon("../images/likelihoodimage01.ico")
        elif id == 205:
            compare = image.ImageCompare()
            compare.blur(self.image)
            frame = ShowImage2(num=4, title="均值滤波肤色检测",
                               title_list=["原图像", "均值滤波图像", "肤色检测结果图像", "二值化图像"],
                               image_path_list=["../TempInfo/blur_compare_1.jpg",
                                                "../TempInfo/blur_compare_2.jpg",
                                                "../TempInfo/blur_compare_3.jpg",
                                                "../TempInfo/blur_compare_4.jpg"])
            frame.set_icon("../images/likelihoodimage01.ico")
        elif id == 206:
            compare = image.ImageCompare()
            compare.gaussianblur(self.image)
            frame = ShowImage2(num=4, title="高斯滤波肤色检测",
                               title_list=["原图像", "高斯滤波图像", "肤色检测结果图像", "二值化图像"],
                               image_path_list=["../TempInfo/gaussianblur_compare_1.jpg",
                                                "../TempInfo/gaussianblur_compare_2.jpg",
                                                "../TempInfo/gaussianblur_compare_3.jpg",
                                                "../TempInfo/gaussianblur_compare_4.jpg"])
            frame.set_icon("../images/likelihoodimage01.ico")
        else:
            compare = image.ImageCompare()
            compare.bilateralfilter(self.image)
            frame = ShowImage2(num=4, title="双边滤波肤色检测",
                               title_list=["原图像", "双边滤波图像", "肤色检测结果图像", "二值化图像"],
                               image_path_list=["../TempInfo/bilateralfilter_compare_1.jpg",
                                                "../TempInfo/bilateralfilter_compare_2.jpg",
                                                "../TempInfo/bilateralfilter_compare_3.jpg",
                                                "../TempInfo/bilateralfilter_compare_4.jpg"])
            frame.set_icon("../images/likelihoodimage01.ico")

    def OnClick31(self, event):
        """
        按钮点击事件
        :param event: 事件源
        :return: None
        """
        id = event.GetId()
        if id == 301:
            compare = image.ImageCompare()
            compare.cbcrrange(self.image)
            frame = ShowImage2(num=3, title="Cb、Cr值范围筛选法肤色检测",
                               title_list=["原图像", "肤色检测结果图像", "二值化图像"],
                               image_path_list=["../TempInfo/cbcrrange_compare_1.jpg",
                                                "../TempInfo/cbcrrange_compare_2.jpg",
                                                "../TempInfo/cbcrrange_compare_3.jpg"])
            frame.set_icon("../images/likelihoodimage01.ico")
        elif id == 302:
            compare = image.ImageCompare()
            compare.ellipse(self.image)
            frame = ShowImage2(num=3, title="椭圆模型检测法肤色检测",
                               title_list=["原图像", "肤色检测结果图像", "二值化图像"],
                               image_path_list=["../TempInfo/ellipse_compare_1.jpg",
                                                "../TempInfo/ellipse_compare_2.jpg",
                                                "../TempInfo/ellipse_compare_3.jpg"])
            frame.set_icon("../images/likelihoodimage01.ico")
        else:
            compare = image.ImageCompare()
            compare.crotsu(self.image)
            frame = ShowImage2(num=3, title="Cr分量+Otsu阈值分割法肤色检测",
                               title_list=["原图像", "肤色检测结果图像", "二值化图像"],
                               image_path_list=["../TempInfo/crotsu_compare_1.jpg",
                                                "../TempInfo/crotsu_compare_2.jpg",
                                                "../TempInfo/crotsu_compare_3.jpg"])
            frame.set_icon("../images/likelihoodimage01.ico")

    def OnClick41(self, event):
        """
        按钮点击事件
        :param event: 事件源
        :return: None
        """
        id = event.GetId()
        if id == 401:
            compare = image.ImageCompare()
            compare.gongren(self.image)
            frame = ShowImage2(num=3, title="公认:Cb∈[77,127],Cr∈[133，173]肤色检测",
                               title_list=["原图像", "肤色检测结果图像", "二值化图像"],
                               image_path_list=["../TempInfo/gongren_compare_1.jpg",
                                                "../TempInfo/gongren_compare_2.jpg",
                                                "../TempInfo/gongren_compare_3.jpg"])
            frame.set_icon("../images/likelihoodimage01.ico")
        elif id == 402:
            compare = image.ImageCompare()
            compare.xuezhe(self.image)
            frame = ShowImage2(num=3, title="学者:Cb∈[90，135],Cr∈[137,167]肤色检测",
                               title_list=["原图像", "肤色检测结果图像", "二值化图像"],
                               image_path_list=["../TempInfo/xuezhe_compare_1.jpg",
                                                "../TempInfo/xuezhe_compare_2.jpg",
                                                "../TempInfo/xuezhe_compare_3.jpg"])
            frame.set_icon("../images/likelihoodimage01.ico")
        else:
            compare = image.ImageCompare()
            compare.zuozhe(self.image)
            frame = ShowImage2(num=3, title="作者:Cb∈[94,126],Cr∈[133，165]肤色检测",
                               title_list=["原图像", "肤色检测结果图像", "二值化图像"],
                               image_path_list=["../TempInfo/zuozhe_compare_1.jpg",
                                                "../TempInfo/zuozhe_compare_2.jpg",
                                                "../TempInfo/zuozhe_compare_3.jpg"])
            frame.set_icon("../images/likelihoodimage01.ico")


class ShowImage2(wx.Frame):
    def __init__(self, image_path_list, title_list, parent=None, num=1, title="未加载", size=wx.DefaultSize):
        """
        初始化对话框
        :param parent: 父控件
        :param size: 大小
        """
        self.title = title

        # 初始化窗口并将其位置居中
        wx.Frame.__init__(self, None, wx.ID_ANY, title=self.title, size=(318, 467),
                          style=wx.CAPTION | wx.CLOSE_BOX | wx.MINIMIZE_BOX)
        # 主面板
        self.panel = wx.Panel(self)
        self.panel.SetBackgroundColour('white')

        # 图像标题
        self.text1 = wx.StaticText(self.panel, label='未加载', pos=(11, 0), size=(280, 25), name="加载文本",
                                   style=wx.ALIGN_CENTER)
        self.text2 = wx.StaticText(self.panel, label='未加载', pos=(305, 0), size=(280, 25), name="加载文本",
                                   style=wx.ALIGN_CENTER)
        self.text3 = wx.StaticText(self.panel, label='未加载', pos=(599, 0), size=(280, 25), name="加载文本",
                                   style=wx.ALIGN_CENTER)
        self.text4 = wx.StaticText(self.panel, label='未加载', pos=(893, 0), size=(280, 25), name="加载文本",
                                   style=wx.ALIGN_CENTER)
        self.text2.Hide()
        self.text3.Hide()
        self.text4.Hide()

        # 图片
        self.image1 = wx.StaticBitmap(self.panel, wx.ID_ANY, bitmap=wx.Bitmap("../images/loading02.png"),
                                      pos=(11, 31), size=(280, 391), style=0, name="图片1")
        self.image2 = wx.StaticBitmap(self.panel, wx.ID_ANY, bitmap=wx.Bitmap("../images/loading02.png"),
                                      pos=(305, 31), size=(280, 391), style=0, name="图片2")
        self.image3 = wx.StaticBitmap(self.panel, wx.ID_ANY, bitmap=wx.Bitmap("../images/loading02.png"),
                                      pos=(599, 31), size=(280, 391), style=0, name="图片3")
        self.image4 = wx.StaticBitmap(self.panel, wx.ID_ANY, bitmap=wx.Bitmap("../images/loading02.png"),
                                      pos=(893, 31), size=(280, 391), style=0, name="图片4")
        self.image2.Hide()
        self.image3.Hide()
        self.image4.Hide()

        self.Center()
        self.Show()
        self.__set_image(num, image_path_list, title_list)

    def __set_image(self, num, image_path_list, title_list):
        """
        设置图像标题
        :param title_list:
        :return:
        """
        image_size = self.__get_image_size(image_path_list[0])
        frame_size = (38 + image_size[0] * num + 14 * (num - 1), 76 + image_size[1])
        self.SetSize(frame_size)
        self.Center()
        controls_list = self.panel.GetChildren()
        len_title = len(image_path_list)
        # print(image_path_list)
        # print(title_list)
        for i in range(0, len_title):
            controls_list[i].Show()
            controls_list[i].SetLabelText(title_list[i])
        for i in range(4, 4 + len_title):
            controls_list[i].Show()
            controls_list[i].SetBitmap(wx.Bitmap(image_path_list[i - 4]))

    def set_icon(self, icon_path):
        """
        设置窗口的图标
        :param icon_path:
        :return:
        """
        icon = wx.Icon()
        icon.LoadFile(icon_path, wx.BITMAP_TYPE_ICO)
        self.SetIcon(icon)

    def __get_image_size(self, image_path):
        """
        根据图像得到图像大小
        :param image_path: 图像尺寸
        :return: 图像尺寸
        """
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        return (image.shape[1], image.shape[0])
