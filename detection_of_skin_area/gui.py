import wx
import wx.lib.buttons as buttons
from detection_of_skin_area import controls, rw, image
import cv2
import numpy as np
import threading
import os
from PIL import Image
import sys


class MainFrame(wx.Frame):
    def __init__(self, label_dict_lists, personal_settings):
        self.window_selected = None     # 窗口(全屏)模式下被选中的视图
        self.list_selected = []     # 列表模式下被选中的视图
        self.file_name = None   # 导入图像的文件名
        self.dir_name = None    # 导入图像所在的目录路径
        self.image = None   # 导入的图像
        self.controls_list = []     # 控件列表
        self.image_list = []    # 图像列表
        self.pattern = False    # False代表全屏模式， True代表列表模式
        self.title_list = []    # 标题列表
        self.window_list = []   # 视图列表
        self.dialog = None  # 弹窗
        self.child_frame = None     # 子窗口
        self.probability_list = []      # 肤色似然率列表
        self.skin_control_type = -1     # 肤色似然控件类型
        self.model_value = None     # 模型计算所得值
        self.thread1 = None     # 子线程
        self.thread2 = None
        self.help_control_type = -1     # 关于控件类型
        self.language_style = personal_settings['language_style']   # 语言风格
        self.label_dict = None    # GUI控件label集合
        if self.language_style == 'cn':
            self.label_dict = label_dict_lists[0]
        else:
            self.label_dict = label_dict_lists[1]
        # 初始化一个窗口对象
        wx.Frame.__init__(self, None, wx.ID_ANY, self.label_dict['MainFrameTitle'], size=(1005, 700))   # 主窗口
        self.menu_bar = wx.MenuBar()  # 菜单栏
        self.tool_bar = self.CreateToolBar()  # 工具栏
        self.status_bar = self.CreateStatusBar()    #状态栏
        self.main_panel = wx.Panel(self)   # 主面板
        self.fun_panel = wx.Panel(self.main_panel, name="FunctionPanel", pos=(0, 0), size=(271, 588))
        self.show_panel = wx.Panel(self.main_panel, name="ShowPanel", pos=(271, 0), size=(718, 588))
        self.build_model_text = wx.StaticText(self.fun_panel, label=self.label_dict['Build Model Text'], pos=(5, 5), size=(251, 30),
                                         name="ModelBuildText", style=wx.ALIGN_LEFT)
        self.btn1 = buttons.GenButton(self.fun_panel, label=self.label_dict['Build Model Btn'], pos=(5, 35), size=(251, 40))    # 建立模型按钮
        self.io_text = wx.StaticText(self.fun_panel, label=self.label_dict['IO Text'], pos=(5, 80), size=(251, 30),
                                     name="输入输出文本", style=wx.ALIGN_LEFT)
        self.btn2 = buttons.GenButton(self.fun_panel, label=self.label_dict['Open Btn'], pos=(5, 110), size=(251, 40))
        self.btn3 = buttons.GenButton(self.fun_panel, label=self.label_dict['Step Delete Btn'], pos=(5, 155), size=(251, 40))
        self.btn4 = buttons.GenButton(self.fun_panel, label=self.label_dict['Clear Btn'], pos=(5, 200), size=(251, 40))
        self.skin_detection_text = wx.StaticText(self.fun_panel, label=self.label_dict['Skin Detection Text'], pos=(5, 245), size=(251, 30),
                                                 name="肤色检测文本", style=wx.ALIGN_LEFT)
        self.btn5 = buttons.GenButton(self.fun_panel, label=self.label_dict['Skin Detection Btn'], pos=(5, 275), size=(251, 40))
        self.step_detection_text = wx.StaticText(self.fun_panel, label=self.label_dict['Step Detection Text'], pos=(5, 320), size=(251, 30),
                                                 name="单步检测文本", style=wx.ALIGN_LEFT)
        self.btn6 = buttons.GenButton(self.fun_panel, label=self.label_dict['Step Illumination Compensation Btn'], pos=(5, 350), size=(251, 40))
        # 下拉列表选项列表
        self.ic_text = wx.StaticText(self.fun_panel, -1, label=self.label_dict['IC Text'], pos=(5, 396), size=(60, 18), style=wx.ALIGN_LEFT)
        self.choice1 = wx.Choice(self.fun_panel, -1, pos=(70, 393), size=(186, 26), choices=eval(self.label_dict['IC Method']))
        # 默认选中第一个
        self.choice1.SetSelection(0)
        self.dn_text = wx.StaticText(self.fun_panel, -1, label=self.label_dict['DN Text'], pos=(5, 425), size=(60, 18), style=wx.ALIGN_LEFT)
        self.choice2 = wx.Choice(self.fun_panel, -1, pos=(70, 422), size=(186, 26), choices=eval(self.label_dict['DN Method']))
        self.choice2.SetSelection(0)
        self.btn7 = buttons.GenButton(self.fun_panel, label=self.label_dict['Step Denoise Btn'], pos=(5, 451), size=(251, 40))
        self.btn8 = buttons.GenButton(self.fun_panel, label=self.label_dict['Step Detection Btn'], pos=(5, 496), size=(251, 40))
        self.btn9 = buttons.GenButton(self.fun_panel, label=self.label_dict['Step Binarization Btn'], pos=(5, 541), size=(251, 40))
        self.title_panel = wx.Panel(self.show_panel, pos=(5, 5), size=(708, 30), name="标题面板")
        self.window_panel = wx.Panel(self.show_panel, pos=(5, 40), size=(708, 543), name="窗口面板")
        self.title_welcome = wx.StaticText(self.title_panel, -1, label=self.label_dict['Welcome Title'], pos=(0, 0), style=wx.ALIGN_LEFT)  # ▷Welcome
        self.title_original_image = buttons.GenButton(self.title_panel, label=self.label_dict['Original Title'], pos=(0, 0))  # ▷Original
        self.title_preprocess_image = buttons.GenButton(self.title_panel, label=self.label_dict['Preprocess Title'], pos=(65, 0))  # ▷Preprocess
        self.title_binarization_image = buttons.GenButton(self.title_panel, label=self.label_dict['Segmentation Title'], pos=(150, 0))  # ▷Binarization
        self.title_segmentation_image = buttons.GenButton(self.title_panel, label=self.label_dict['Binarization Title'], pos=(235, 0))  # ▷Segmentation
        self.title_window_2_list = buttons.GenBitmapButton(self.title_panel, bitmap=wx.Bitmap("../images/list01.png"),
                                           size=(30, 30), name="PatternConver1")
        self.title_list_2_window = buttons.GenBitmapButton(self.title_panel, bitmap=wx.Bitmap("../images/full01.png"),
                                                           size=(30, 30), name="PatternConver2")
        image_welcome = cv2.imdecode(np.fromfile("../images/welcome.png", dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        self.window_welcome = controls.ImageView(self.window_panel, image=image_welcome, label=self.label_dict['Welcome Window'], pos=(0, 0), size=(708, 543))
        self.window_original = controls.ImageView(self.window_panel, image=image_welcome, label=self.label_dict['Original Window'], pos=(0, 0), size=(708, 543))
        self.window_preprocess = controls.ImageView(self.window_panel, image=image_welcome, label=self.label_dict['Preprocess Window'], pos=(0, 0), size=(708, 543))
        self.window_binarization = controls.ImageView(self.window_panel, image=image_welcome, label=self.label_dict['Binarization Window'], pos=(0, 0), size=(708, 543))
        self.window_segmentation = controls.ImageView(self.window_panel, image=image_welcome, label=self.label_dict['Segmentation Window'], pos=(0, 0), size=(708, 543))

        # 设置控件属性
        self.__set_control_properties()

    def __set_control_properties(self):
        """
        设置子控件的属性
        :return: None
        """
        # 主窗口设置
        self.Center()  # 窗口居中
        self.SetMinSize((1005, 700))  # 窗口的最小尺寸
        # 为主窗口设置图标
        icon = wx.Icon()
        icon.LoadFile("../images/icon03_64.ico", wx.BITMAP_TYPE_ICO)
        self.SetIcon(icon)

        # 菜单栏设置
        self.SetMenuBar(self.menu_bar)    # 将菜单栏添加至主窗口
        self.__set_menu_bar()

        # 工具栏设置
        self.__set_tool_bar()

        # 状态栏设置
        self.__set_status_bar()

        # 主面板设置
        self.__set_main_panel()

        # 控件入栈
        self.__add_controls()

        self.Show(True)  # 显示主窗口
        self.__update_main_frame()

    def __set_menu_bar(self):
        """
        设置菜单栏的属性并为其添加子控件
        :return: None
        """
        self.file_menu = wx.Menu()  # 文件菜单
        self.edit_menu = wx.Menu()  # 编辑菜单
        self.view_menu = wx.Menu()  # 视图菜单
        self.settings_menu = wx.Menu()  # 设置菜单
        self.window_menu = wx.Menu()  # 窗口菜单
        self.help_menu = wx.Menu()  # 帮助菜单

        # 将菜单添加到菜单栏上
        self.menu_bar.Append(self.file_menu, self.label_dict['FileMenu'])
        self.menu_bar.Append(self.edit_menu, self.label_dict['EditMenu'])
        self.menu_bar.Append(self.view_menu, self.label_dict['ViewMenu'])
        self.menu_bar.Append(self.settings_menu, self.label_dict['SettingsMenu'])
        # self.menu_bar.Append(self.window_menu, self.label_dict['WindowMenu'])
        self.menu_bar.Append(self.help_menu, self.label_dict['HelpMenu'])

        # 初始化文件菜单项
        self.file_open = wx.MenuItem(self.file_menu, wx.ID_OPEN, self.label_dict['Open'])
        self.file_save = wx.MenuItem(self.file_menu, wx.ID_SAVE, self.label_dict['Save'])
        self.file_save_all = wx.MenuItem(self.file_menu, wx.ID_ANY, self.label_dict['Save All'])
        self.file_save_as = wx.MenuItem(self.file_menu, wx.ID_SAVEAS, self.label_dict['Save As'])
        self.file_save_all_as = wx.MenuItem(self.file_menu, wx.ID_ANY, self.label_dict['Save All As'])
        self.file_delete = wx.MenuItem(self.file_menu, wx.ID_CLOSE, self.label_dict['Delete'])
        self.file_delete_all = wx.MenuItem(self.file_menu, wx.ID_ANY, self.label_dict['Delete All'])
        self.file_exit = wx.MenuItem(self.file_menu, wx.ID_EXIT, self.label_dict['Exit'])
        # 初始化编辑菜单项
        self.edit_build_model = wx.MenuItem(self.edit_menu, wx.ID_ANY, self.label_dict['Build Model'])
        self.edit_skin_detection = wx.MenuItem(self.edit_menu, wx.ID_ANY, self.label_dict['Skin Detection'])
        self.edit_step_preprocess = wx.MenuItem(self.edit_menu, wx.ID_ANY, self.label_dict['Step Preprocess'])
        self.edit_step_skin_detection = wx.MenuItem(self.edit_menu, wx.ID_ANY, self.label_dict['Step Skin Detection'])
        self.edit_step_binarization = wx.MenuItem(self.edit_menu, wx.ID_ANY, self.label_dict['Binarization'])
        self.edit_step_skin_segmentation = wx.MenuItem(self.edit_menu, wx.ID_ANY, self.label_dict['Skin Segmentation'])
        # 初始化视图菜单项
        self.view_sample_data_distribution = wx.MenuItem(self.view_menu, wx.ID_ANY, self.label_dict['Sample Data Distribution'])
        self.view_model_value = wx.MenuItem(self.view_menu, wx.ID_ANY, self.label_dict['Model Value'])
        self.view_skin_likelihood_probability = wx.MenuItem(self.view_menu, 201, self.label_dict['Skin Likelihood Probability'])
        self.view_skin_likelihood_image = wx.MenuItem(self.view_menu, 202, self.label_dict['Skin Likelihood Image'])
        self.view_skin_likelihood_binarization_image = wx.MenuItem(self.view_menu, 203, self.label_dict['Skin Likelihood Binarization Image'])
        # 初始化设置菜单项
        self.settings_cn = wx.MenuItem(self.settings_menu, wx.ID_ANY, self.label_dict['Chinese'], kind=wx.ITEM_RADIO)
        self.settings_en = wx.MenuItem(self.settings_menu, wx.ID_ANY, self.label_dict['English'], kind=wx.ITEM_RADIO)
        # 初始化窗口菜单项
        self.window_new = wx.MenuItem(self.window_menu, wx.ID_ANY, self.label_dict['New Window'])
        # 初始化帮助菜单项
        self.help_soft = wx.MenuItem(self.help_menu, 501, self.label_dict['About Software'])
        self.help_author = wx.MenuItem(self.help_menu, 502, self.label_dict['About Author'])
        self.help_source_code = wx.MenuItem(self.help_menu, 503, self.label_dict['Source Code'])

        # 设置菜单项背景色为白色
        self.file_open.SetBackgroundColour('white')
        self.file_save.SetBackgroundColour('white')
        self.file_save_all.SetBackgroundColour('white')
        self.file_save_as.SetBackgroundColour('white')
        self.file_save_all_as.SetBackgroundColour('white')
        self.file_delete.SetBackgroundColour('white')
        self.file_delete_all.SetBackgroundColour('white')
        self.file_exit.SetBackgroundColour('white')
        self.edit_build_model.SetBackgroundColour('white')
        self.edit_skin_detection.SetBackgroundColour('white')
        self.edit_step_preprocess.SetBackgroundColour('white')
        self.edit_step_skin_detection.SetBackgroundColour('white')
        self.edit_step_binarization.SetBackgroundColour('white')
        self.edit_step_skin_segmentation.SetBackgroundColour('white')
        self.view_sample_data_distribution.SetBackgroundColour('white')
        self.view_model_value.SetBackgroundColour('white')
        self.view_skin_likelihood_probability.SetBackgroundColour('white')
        self.view_skin_likelihood_image.SetBackgroundColour('white')
        self.view_skin_likelihood_binarization_image.SetBackgroundColour('white')
        self.settings_cn.SetBackgroundColour('white')
        self.settings_en.SetBackgroundColour('white')
        self.window_new.SetBackgroundColour('white')
        self.help_soft.SetBackgroundColour('white')
        self.help_author.SetBackgroundColour('white')
        self.help_source_code.SetBackgroundColour('white')

        # 为菜单项添加图标
        self.file_open.SetBitmap(wx.Bitmap('../images/open01.png'))  # 添加一个图标
        self.file_save.SetBitmap(wx.Bitmap('../images/save01.png'))  # 添加一个图标
        self.file_save_as.SetBitmap(wx.Bitmap('../images/saveas01.png'))  # 添加一个图标
        self.file_delete.SetBitmap(wx.Bitmap('../images/clear01.png'))  # 添加一个图标
        self.file_exit.SetBitmap(wx.Bitmap('../images/exit01.png'))  # 添加一个图标
        self.edit_build_model.SetBitmap(wx.Bitmap('../images/model01.png'))  # 添加一个图标
        self.edit_step_preprocess.SetBitmap(wx.Bitmap('../images/preprocess02.png'))  # 添加一个图标
        self.edit_skin_detection.SetBitmap(wx.Bitmap('../images/detection02.png'))  # 添加一个图标
        self.edit_step_binarization.SetBitmap(wx.Bitmap('../images/binarization02.png'))  # 添加一个图标
        self.edit_step_skin_segmentation.SetBitmap(wx.Bitmap('../images/segmentation02.png'))  # 添加一个图标
        self.view_sample_data_distribution.SetBitmap(wx.Bitmap('../images/distribution01.png'))  # 添加一个图标
        self.view_model_value.SetBitmap(wx.Bitmap('../images/number01.png'))  # 添加一个图标
        self.view_skin_likelihood_probability.SetBitmap(wx.Bitmap('../images/probability01.png'))  # 添加一个图标
        self.view_skin_likelihood_image.SetBitmap(wx.Bitmap('../images/likelihoodimage01.png'))  # 添加一个图标
        self.window_new.SetBitmap(wx.Bitmap('../images/newwindow01.png'))  # 添加一个图标
        self.help_soft.SetBitmap(wx.Bitmap('../images/soft01.png'))  # 添加一个图标
        self.help_author.SetBitmap(wx.Bitmap('../images/author01.png'))  # 添加一个图标
        self.help_source_code.SetBitmap(wx.Bitmap('../images/code01.png'))  # 添加一个图标

        # 将菜单项添加到文件菜单上
        self.file_menu.Append(self.file_open)
        self.file_menu.Append(self.file_save)
        self.file_menu.Append(self.file_save_all)
        self.file_menu.Append(self.file_save_as)
        self.file_menu.Append(self.file_save_all_as)
        self.file_menu.Append(self.file_delete)
        self.file_menu.Append(self.file_delete_all)
        self.file_menu.Append(self.file_exit)
        # 将菜单项添加到编辑菜单上
        self.edit_menu.Append(self.edit_build_model)
        self.edit_menu.Append(self.edit_skin_detection)
        self.edit_separator = self.edit_menu.AppendSeparator()
        self.edit_separator.SetBackgroundColour('white')
        self.edit_menu.Append(self.edit_step_preprocess)
        self.edit_menu.Append(self.edit_step_skin_detection)
        self.edit_menu.Append(self.edit_step_binarization)
        self.edit_menu.Append(self.edit_step_skin_segmentation)
        # 将菜单项添加到视图菜单上
        self.view_menu.Append(self.view_sample_data_distribution)
        self.view_menu.Append(self.view_model_value)
        self.view_menu.Append(self.view_skin_likelihood_probability)
        self.view_menu.Append(self.view_skin_likelihood_image)
        self.view_menu.Append(self.view_skin_likelihood_binarization_image)
        # 将菜单项添加到设置菜单上
        self.settings_menu.Append(self.settings_cn)
        self.settings_menu.Append(self.settings_en)
        # 将菜单项添加到窗口菜单上
        self.window_menu.Append(self.window_new)
        # 将菜单项添加到帮助菜单上
        self.help_menu.Append(self.help_soft)
        self.help_menu.Append(self.help_author)
        self.help_menu.Append(self.help_source_code)

        # 绑定菜单项事件
        self.Bind(wx.EVT_MENU, self.OnOpen, self.file_open)
        # self.Bind(wx.EVT_MENU, self.OnSave, self.file_save)
        # self.Bind(wx.EVT_MENU, self.OnSave, self.file_save_all)
        # self.Bind(wx.EVT_MENU, self.OnSaveAs, self.file_save_as)
        # self.Bind(wx.EVT_MENU, self.OnSaveAllAs, self.file_save_all_as)
        self.Bind(wx.EVT_MENU, self.OnStepDelete, self.file_delete)
        self.Bind(wx.EVT_MENU, self.OnDeleteAll, self.file_delete_all)
        self.Bind(wx.EVT_MENU, self.OnExit, self.file_exit)
        self.Bind(wx.EVT_MENU, self.OnDetection, self.edit_skin_detection)
        self.Bind(wx.EVT_MENU, self.OnCreateModel, self.edit_build_model)
        self.Bind(wx.EVT_MENU, self.OnDrawPlot, self.view_sample_data_distribution)
        self.Bind(wx.EVT_MENU, self.OnModelValue, self.view_model_value)
        self.Bind(wx.EVT_MENU, self.OnGetSkinLikelihood, self.view_skin_likelihood_probability)
        self.Bind(wx.EVT_MENU, self.OnGetSkinLikelihood, self.view_skin_likelihood_image)
        self.Bind(wx.EVT_MENU, self.OnGetSkinLikelihood, self.view_skin_likelihood_binarization_image)
        self.Bind(wx.EVT_MENU, self.OnAbout, self.help_soft)
        self.Bind(wx.EVT_MENU, self.OnAbout, self.help_author)
        self.Bind(wx.EVT_MENU, self.OnAbout, self.help_source_code)

    def __set_tool_bar(self):
        """
        设置工具栏的属性并为其添加子工具
        :return: None
        """
        # 设置 工具栏 背景色
        self.tool_bar.SetBackgroundColour('white')

        # 为工具栏添加一系列工具对象
        self.tool1 = self.tool_bar.AddTool(wx.ID_ANY, "", wx.Bitmap('../images/open01.png'), self.label_dict['Open Tool'])
        self.tool2 = self.tool_bar.AddTool(wx.ID_ANY, "", wx.Bitmap('../images/save01.png'), self.label_dict['Sava Tool'])
        self.tool3 = self.tool_bar.AddTool(wx.ID_ANY, "", wx.Bitmap('../images/saveas01.png'), self.label_dict['Save As Tool'])
        self.tool4 = self.tool_bar.AddTool(504, "", wx.Bitmap('../images/code01.png'), self.label_dict['Source Code Tool'])
        self.tool5 = self.tool_bar.AddTool(wx.ID_ANY, "", wx.Bitmap('../images/settings01.png'), self.label_dict['Settings Tool'])
        self.tool6 = self.tool_bar.AddTool(wx.ID_ANY, "", wx.Bitmap('../images/model01.png'), self.label_dict['Build Model Tool'])
        self.tool7 = self.tool_bar.AddTool(wx.ID_ANY, "", wx.Bitmap('../images/distribution01.png'), self.label_dict['Sample Data Distribution Tool'])
        self.tool8 = self.tool_bar.AddTool(wx.ID_ANY, "", wx.Bitmap('../images/number01.png'), self.label_dict['Model Value Tool'])
        self.tool9 = self.tool_bar.AddTool(301, "", wx.Bitmap('../images/probability01.png'), self.label_dict['Skin Likelihood Probability Tool'])
        self.tool10 = self.tool_bar.AddTool(302, "", wx.Bitmap('../images/likelihoodimage01.png'), self.label_dict['Skin Likelihood Image Tool'])
        self.tool11 = self.tool_bar.AddTool(303, "", wx.Bitmap('../images/binarizationimage01.png'), self.label_dict['Skin Likelihood Binarization Image Tool'])
        self.tool16 = self.tool_bar.AddTool(wx.ID_ANY, "", wx.Bitmap('../images/binarizationimage01.png'), self.label_dict['Ellipse Tool'])
        self.tool21 = self.tool_bar.AddTool(wx.ID_ANY, "", wx.Bitmap('../images/ellipse01.png'), self.label_dict['Ellipse Model Tool'])
        # self.tool18 = self.tool_bar.AddTool(wx.ID_ANY, "", wx.Bitmap('../images/binarizationimage01.png'), self.label_dict['Cr_Otsu Tool'])
        self.tool19 = self.tool_bar.AddTool(wx.ID_ANY, "", wx.Bitmap('../images/binarizationimage01.png'), self.label_dict['RGB Tool'])
        self.tool17 = self.tool_bar.AddTool(wx.ID_ANY, "", wx.Bitmap('../images/binarizationimage01.png'), self.label_dict['HSV Tool'])
        self.tool20 = self.tool_bar.AddTool(wx.ID_ANY, "", wx.Bitmap('../images/compare01.png'), self.label_dict['Compare Tool'])
        self.tool12 = self.tool_bar.AddTool(wx.ID_ANY, "", wx.Bitmap('../images/back01.png'), self.label_dict['Step Delete Tool'])
        self.tool13 = self.tool_bar.AddTool(wx.ID_ANY, "", wx.Bitmap('../images/clear01.png'), self.label_dict['Clear Tool'])
        self.tool14 = self.tool_bar.AddTool(wx.ID_ANY, "", wx.Bitmap('../images/detection02.png'), self.label_dict['Skin Detection Tool'])
        self.tool15 = self.tool_bar.AddTool(wx.ID_ANY, "", wx.Bitmap('../images/next01.png'), self.label_dict['Step Tool'])
        self.tool22 = self.tool_bar.AddTool(wx.ID_ANY, "", wx.Bitmap('../images/compare01.png'), self.label_dict['Multi-scheme Comparison Tool'])

        # 显现工具栏
        self.tool_bar.Realize()

        # 事件绑定
        self.Bind(wx.EVT_TOOL, self.OnOpen, self.tool1)
        self.Bind(wx.EVT_TOOL, self.OnAbout, self.tool4)
        self.Bind(wx.EVT_TOOL, self.OnCreateModel, self.tool6)
        self.Bind(wx.EVT_TOOL, self.OnDrawPlot, self.tool7)
        self.Bind(wx.EVT_TOOL, self.OnModelValue, self.tool8)
        self.Bind(wx.EVT_TOOL, self.OnGetSkinLikelihood, self.tool9)
        self.Bind(wx.EVT_TOOL, self.OnGetSkinLikelihood, self.tool10)
        self.Bind(wx.EVT_TOOL, self.OnGetSkinLikelihood, self.tool11)
        self.Bind(wx.EVT_TOOL, self.OnStepDelete, self.tool12)
        self.Bind(wx.EVT_TOOL, self.OnDeleteAll, self.tool13)
        self.Bind(wx.EVT_TOOL, self.OnDetection, self.tool14)
        self.Bind(wx.EVT_TOOL, self.OnEllipse, self.tool16)
        self.Bind(wx.EVT_TOOL, self.OnHSV, self.tool17)
        # self.Bind(wx.EVT_TOOL, self.OnCr_Otsu, self.tool18)
        self.Bind(wx.EVT_TOOL, self.OnRGB, self.tool19)
        self.Bind(wx.EVT_TOOL, self.OnCompare, self.tool20)
        self.Bind(wx.EVT_TOOL, self.OnEllipseModel, self.tool21)
        self.Bind(wx.EVT_TOOL, self.OnMultiScheme, self.tool22)

    def __set_status_bar(self):
        """
        设置状态栏的初始属性
        :return: None
        """
        self.status_bar.SetBackgroundColour('white')
        self.status_bar.SetFieldsCount(2)
        self.status_bar.SetStatusWidths([-7, -1])
        self.status_bar.SetStatusText(self.label_dict['Steps Status'], 0)
        self.status_bar.SetStatusText(self.label_dict['Author Status'], 1)

    def __set_main_panel(self):
        """
        设置主面板的属性并为其添加子控件
        :return: None
        """
        # Main Panel Settings
        self.main_panel.SetBackgroundColour('white')
        # print("主窗口的尺寸：" + str(self.GetSize()))  #(1005, 700)
        # print("主面板的尺寸：" + str(self.main_panel.GetSize()))   # (989, 588)
        self.fun_panel.SetBackgroundColour('white')
        self.show_panel.SetBackgroundColour('white')
        # Function Text and Button Settings
        self.__set_text_properties(self.build_model_text)
        self.__set_text_properties(self.io_text)
        self.__set_text_properties(self.skin_detection_text)
        self.__set_text_properties(self.step_detection_text)
        self.__set_button_properties(self.btn1)
        self.__set_button_properties(self.btn2)
        self.__set_button_properties(self.btn3)
        self.__set_button_properties(self.btn4)
        self.__set_button_properties(self.btn5)
        self.__set_button_properties(self.btn6)
        self.__set_button_properties(self.btn7)
        self.__set_button_properties(self.btn8)
        self.__set_button_properties(self.btn9)
        self.title_panel.SetBackgroundColour('#D5F0EF')
        self.window_panel.SetBackgroundColour('#D5F0EF')
        # Title Text and Button Settings
        self.__set_text_properties(self.title_welcome)
        self.title_welcome.SetBackgroundColour('#D5F0EF')
        self.__set_button_properties(self.title_original_image)
        self.__set_button_properties(self.title_preprocess_image)
        self.__set_button_properties(self.title_binarization_image)
        self.__set_button_properties(self.title_segmentation_image)
        if self.language_style == 'cn':
            self.title_welcome.SetPosition(wx.Point(0, 5))
            self.title_original_image.SetPosition(wx.Point(0, 0))
            self.title_preprocess_image.SetPosition(wx.Point(65, 0))
            self.title_binarization_image.SetPosition(wx.Point(150, 0))
            self.title_segmentation_image.SetPosition(wx.Point(295, 0))
            self.title_welcome.SetSize(60, 20)
            self.title_original_image.SetSize(60, 30)
            self.title_preprocess_image.SetSize(80, 30)
            self.title_binarization_image.SetSize(140, 30)
            self.title_segmentation_image.SetSize(80, 30)
        else:
            self.title_welcome.SetPosition(wx.Point(0, 5))
            self.title_original_image.SetPosition(wx.Point(0, 0))
            self.title_preprocess_image.SetPosition(wx.Point(115, 0))
            self.title_binarization_image.SetPosition(wx.Point(250, 0))
            self.title_segmentation_image.SetPosition(wx.Point(405, 0))
            self.title_welcome.SetSize(100, 20)
            self.title_original_image.SetSize(110, 30)
            self.title_preprocess_image.SetSize(130, 30)
            self.title_binarization_image.SetSize(150, 30)
            self.title_segmentation_image.SetSize(150, 30)
        self.title_original_image.Hide()
        self.title_preprocess_image.Hide()
        self.title_binarization_image.Hide()
        self.title_segmentation_image.Hide()
        self.title_window_2_list.SetPosition(wx.Point(self.title_panel.GetSize()[0] - 35, 0))
        self.title_list_2_window.SetPosition(wx.Point(self.title_panel.GetSize()[0] - 35, 0))
        self.__set_button_properties(self.title_window_2_list)
        self.__set_button_properties(self.title_list_2_window)
        self.title_window_2_list.Hide()
        self.title_list_2_window.Hide()
        # Window View Settings
        self.window_original.Hide()
        self.window_preprocess.Hide()
        self.window_binarization.Hide()
        self.window_segmentation.Hide()
        self.title_list += [self.title_original_image, self.title_preprocess_image, self.title_binarization_image, self.title_segmentation_image]
        self.window_list += [self.window_original, self.window_preprocess, self.window_binarization, self.window_segmentation]

        # mainPanel 绑定大小改变事件
        self.main_panel.Bind(wx.EVT_SIZE, self.OnSize, self.main_panel)
        # 按钮绑定响应事件
        self.Bind(wx.EVT_BUTTON, self.OnCreateModel, self.btn1)
        self.Bind(wx.EVT_BUTTON, self.OnOpen, self.btn2)
        self.Bind(wx.EVT_BUTTON, self.OnStepDelete, self.btn3)
        self.Bind(wx.EVT_BUTTON, self.OnDeleteAll, self.btn4)
        self.Bind(wx.EVT_BUTTON, self.OnDetection, self.btn5)
        self.Bind(wx.EVT_BUTTON, self.OnConver, self.title_window_2_list)
        self.Bind(wx.EVT_BUTTON, self.OnConver, self.title_list_2_window)
        for i in range(0, len(self.title_list)):
            self.Bind(wx.EVT_BUTTON, self.OnChangeSelected, self.title_list[i])
            self.Bind(wx.EVT_CHECKBOX, self.OnChangeCheckBoxStatus, self.window_list[i].selected_box)
            self.Bind(wx.EVT_BUTTON, self.OnConver, self.window_list[i].full_screen)

    def __set_text_properties(self, text):
        """
        设置给定静态文本的属性
        :param text: 给定的静态文本
        :return: None
        """
        text.SetBackgroundColour('white')
        font = wx.Font(16, wx.DECORATIVE, wx.NORMAL, wx.NORMAL)
        text.SetFont(font)

    def __set_button_properties(self, btn):
        """
        设置给定按钮的属性
        :param btn: 给定的按钮
        :return: None
        """
        btn.SetBackgroundColour('#D5F0EF')
        btn.SetBezelWidth(0)
        btn.SetUseFocusIndicator(False)
        font = wx.Font(16, wx.DECORATIVE, wx.NORMAL, wx.NORMAL)
        btn.SetFont(font)

    def __add_controls(self):
        """
        将所有可应用控件加入到一个列表中
        :return: None
        """
        self.controls_list += [
            self.file_open, self.file_save, self.file_save_all, self.file_save_as, self.file_save_all_as,
            self.file_delete, self.file_delete_all, self.file_exit, self.edit_build_model, self.edit_skin_detection,
            self.edit_step_preprocess, self.edit_step_skin_detection, self.edit_step_binarization,
            self.edit_step_skin_segmentation, self.view_sample_data_distribution, self.view_model_value,
            self.view_skin_likelihood_probability, self.view_skin_likelihood_image,
            self.view_skin_likelihood_binarization_image, self.settings_cn, self.settings_en, self.window_new,
            self.help_soft, self.help_author, self.help_source_code, self.tool1, self.tool2, self.tool3, self.tool4,
            self.tool5, self.tool6, self.tool7, self.tool8, self.tool9, self.tool10, self.tool11, self.tool12,
            self.tool13, self.tool14, self.tool15, self.btn1, self.btn2, self.btn3, self.btn4, self.btn5,
            self.btn6, self.choice1, self.choice2, self.btn7, self.btn8, self.btn9
        ]
        # 将一系列控件封禁
        for i in range(1, 51):
            if i in [7, 8, 9, 19, 20, 21, 22, 23, 24, 25, 28, 29, 30, 40, 41]:
                continue
            else:
                self.controls_list[i].Enable(False)
        # 当高斯模型已经建立的情况下，解禁一些控件
        if os.path.exists("../TempInfo/模型计算所得值.txt"):
            for i in [14, 15, 31, 32]:
                self.controls_list[i].Enable(True)

    def OnSize(self, event):
        """
        当主面板大小改变时，改变子控件的大小及位置
        :param event:
        :return:
        """
        width, height = event.GetSize()
        self.fun_panel.SetSize(271, height)
        self.show_panel.SetSize(width - 271, height)
        show_width, show_height = self.show_panel.GetSize()
        self.title_panel.SetSize(show_width - 10, 30)
        self.window_panel.SetSize(show_width - 10, show_height - 45)

        self.title_window_2_list.SetPosition(wx.Point(self.title_panel.GetSize()[0] - 35, 0))
        self.title_list_2_window.SetPosition(wx.Point(self.title_panel.GetSize()[0] - 35, 0))
        if len(self.image_list) == 0:    # 当图像还未导入时
            self.window_welcome.SetSize(show_width - 10, show_height - 45)
        else:       # 图像已导入
            # if self.pattern is False:  # 全屏模式
            size, pos = self.__get_size_and_position()
            for i in range(0, len(self.image_list)):
                self.window_list[i].SetSize(size[0], size[1])
                self.window_list[i].SetPosition(wx.Point(pos[i][0], pos[i][1]))
            # else:   # 列表模式
            #     # 计算各视图大小及位置并重置显示
            #     size, pos = self.__get_size_and_position()
            #     for i in range(0, len(self.image_list)):
            #         self.window_list[i].SetSize(size[0], size[1])
            #         self.window_list[i].SetPosition(wx.Point(pos[i][0], pos[i][1]))

    def OnOpen(self, event):
        """
        打开图像事件
        :param event: 事件源
        :return: None
        """
        # wx.FileDialog语法：(self, parent, message, defaultDir, defaultFile,wildcard, style, pos)
        # .ai .webp 格式图像暂不支持
        wildcard = "PNG files (*.png)|*.png|" "JPG files (*.jpg)|*.jpg|" "JPEG files (*.jpeg)|*.jpeg|" \
                   "BMP files (*.bmp)|*.bmp|" "PPM files (*.ppm)|*.ppm|" "PNM files (*.pnm)|*.pnm|" \
                   "PBM files (*.pbm)|*.pbm|" "TIF files (*.tiff)|*.tiff|" "CDR files (*.cdr)|*.cdr|" \
                   "ODD files (*.odd)|*.odd|" "FPX files (*.fpx)|*.fpx|" "PBM files (*.pbm)|*.pbm|" \
                   "PGM files (*.pgm)|*.pgm|" "All files (*.*)|*.*"

        file_dialog = wx.FileDialog(self, message=self.label_dict['Import Image Window'], wildcard=wildcard, style=wx.FD_OPEN)
        file_path = ""
        if file_dialog.ShowModal() == wx.ID_OK:
            # 获取到文件名
            self.file_name = file_dialog.GetFilename()
            # 获取到目录名
            self.dir_name = file_dialog.GetDirectory()
            # 组合出完整文件路径
            file_path = self.dir_name + "\\" + self.file_name
        file_dialog.Destroy()
        if file_path is not None and "" != file_path:
            for i in [1, 2, 3, 4, 6, 9, 10, 16, 17, 18, 26, 27, 33, 34, 35, 37, 38, 39, 43, 44, 45, 46, 47]:
                self.controls_list[i].Enable(True)   # 如果导入了图像,将一些控件解禁
            for i in [0, 25, 41]:
                self.controls_list[i].Enable(False)   # 如果导入了图像,将一些控件封禁
            self.image = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            self.image_list += [self.image]

            # 将welcome页面隐藏
            self.title_welcome.Hide()
            self.window_welcome.Hide()
            # 显示原图
            self.title_original_image.Show()
            self.window_original.Show()
            self.window_original.set_image(self.image)

            self.window_selected = self.window_original
            self.title_window_2_list.Show()
            size, pos = self.__get_size_and_position()
            self.window_original.SetSize(size[0], size[1])
            self.window_original.SetPosition(wx.Point(pos[0][0], pos[0][1]))

    def OnConver(self, event):
        """
        转换视图显示模式
        :param event: 事件源
        :return: None
        """

        boo = False
        if event is not None:
            btn = event.GetButtonObj()
            time = -1
            for i in [self.window_list[0].full_screen, self.window_list[1].full_screen,
                      self.window_list[2].full_screen, self.window_list[3].full_screen]:
                time += 1
                if btn is i:
                    self.list_selected.append(self.title_list[time])
                    boo = True
                    break
        if self.pattern is False:   # 全屏模式
            space = self.window_list.index(self.window_selected)
            self.list_selected.append(self.title_list[space])
            self.window_selected = None
            self.pattern = True
            self.title_window_2_list.Hide()
            self.title_list_2_window.Show()
            for i in self.window_list:
                i.set_pattern(True)
            self.window_list[space].set_status(True)
            for i in range(0, len(self.image_list)):
                self.title_list[i].Show()
                self.window_list[i].Show()
            self.__change_title_color()     # 将标题栏颜色改变
        else:       # 列表模式
            # 将列表模式下所有已选中的视图置为False,并将已选中的列表清空，同时将标题全部置黑
            for i in self.list_selected:
                space_t = self.title_list.index(i)
                self.window_list[space_t].set_status(False)

            for i in self.window_list:
                i.set_pattern(False)
                i.Hide()
            if boo:
                # print(boo)
                # print(time)
                self.window_selected = self.window_list[time]
            else:
                # 获取到第一个选中视图，将其显示
                space_tt = -1
                for i in range(0, len(self.title_list)):
                    for j in self.list_selected:
                        if j is self.title_list[i]:
                            space_tt = i
                            break
                    if space_tt != -1:
                        break
                self.window_selected = self.window_list[space_tt]
            self.window_selected.Show()
            # 将当前模式改为全屏、并改变转换按钮
            self.pattern = False
            self.title_list_2_window.Hide()
            self.title_window_2_list.Show()
            # 根据当前状态设置标题栏颜色，同时将列表模式下的已选中列表清空
            self.__change_title_color()
            self.list_selected.clear()
        # 重置所有视图的大小及位置
        size, pos = self.__get_size_and_position()
        for i in range(0, len(self.image_list)):
            self.window_list[i].SetSize(size[0], size[1])
            self.window_list[i].SetPosition(wx.Point(pos[i][0], pos[i][1]))
            self.window_list[i].conver()

    def __get_size_and_position(self):
        """
        根据模式以及图像的尺寸比例，计算视图的大小和位置
        :return: 视图尺寸、视图位置列表
        """
        if self.pattern is False:   # 全屏模式
            w, h = self.window_panel.GetSize()
            # 获取图片大小
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
            size = (view_w, view_h)
            pos_temp = (int((w - view_w) / 2), int((h - view_h) / 2))
            pos = []
            for i in range(0, len(self.image_list)):
                pos += [pos_temp]
            return size, pos
        else:       # 列表模式
            pos = []
            w = int((self.window_panel.GetSize()[0] - 15) / 2)
            h = int((self.window_panel.GetSize()[1] - 15) / 2)
            # 获取图片大小
            img_w = self.image.shape[1]  # 获取到图片大小
            img_h = self.image.shape[0]
            view_w = 0
            view_h = 0
            if img_w > img_h:  # 图片长
                # 图片长，那么，长度为其限制
                view_w = w
                view_h = int(img_h * view_w / img_w + 26)
                # print("图片长度：" + str(imgWidth1))
                # print("图片高度：" + str(imgHeight1))
                if view_h > h:  # 图片过高
                    view_h = h
                    view_w = int((img_w - 26) * view_h / img_h)
            else:  # 图片高
                # 图片高，那么，高度为其限制
                view_h = h - 26
                view_w = int(img_w * view_h / img_h)
                if view_w > w:  # 图片过长
                    view_w = w
                    view_h = int(img_h * view_w / img_w + 26)

            size = (view_w, view_h)
            for i in range(0, len(self.image_list)):
                if i < 2:
                    pos_temp = ((w - view_w) / 2 + (i + 1) * 5 + i * w, (h - view_h) / 2 + 5)
                else:
                    i1 = i % 2
                    pos_temp = ((w - view_w) / 2 + (i1 + 1) * 5 + i1 * w, (h - view_h) / 2 + h + 10)
                pos += [pos_temp]
            return size, pos

    def OnDeleteAll(self, event):
        """
        清空图像
        :param event: 事件源
        :return: None
        """
        # 将标题和视图隐藏
        for i in range(0, len(self.image_list)):
            self.title_list[i].Hide()
            self.window_list[i].Hide()
        self.pattern = False
        self.title_window_2_list.Hide()
        self.title_list_2_window.Hide()
        # 将图像清空
        self.image = None
        self.image_list.clear()
        # 将一些控件封禁
        for i in [1, 2, 3, 4, 6, 9, 10, 16, 17, 18, 26, 27, 33, 34, 35, 37, 38, 39, 43, 44, 45, 46, 47]:
            self.controls_list[i].Enable(False)
        # 将导入图像按钮解禁
        for i in [0, 25, 41]:
            self.controls_list[i].Enable(True)
        # 将欢迎页面显示出来
        self.title_welcome.Show()
        self.window_welcome.Show()
        # 欢迎页面大小重置
        self.window_welcome.SetSize(self.window_panel.GetSize()[0], self.window_panel.GetSize()[1])
        # 所有视图状态重置
        for i in self.window_list:
            i.set_pattern(False)
            i.set_status(False)
            i.conver()

    def OnCreateModel(self, event):
        """
        建立模型
        :param event: 事件源
        :return: None
        """
        # 将一些控件解禁
        self.controls_list[14].Enable(True)
        self.controls_list[15].Enable(True)

        self.dialog = controls.Loading(size=(300, 200))  # 初始化对话框
        self.dialog.Show()  # 显示对话框
        self.Enable(False)  # 将主窗口置为不可用

        # 建立线程，开始肤色检测
        create_model_thread = threading.Thread(target=self.__create_model)
        create_model_thread.start()  # 启动线程

    def __create_model(self):
        """
        开始建立模型
        :return: None
        """
        # 统计数据
        image_process = image.ImageProcess()
        image_process.build_model(self.dialog)

        # 数据统计完毕，关闭弹窗
        self.dialog.close()
        self.Enable(True)  # 将主窗口置为可用

    def OnDetection(self, event):
        """
        肤色检测
        :param event: 事件源
        :return: None
        """
        # 将一些控件封禁
        for i in [10, 38, 39, 44, 45, 46, 47]:
            self.controls_list[i].Enable(False)

        self.dialog = controls.Loading(size=(300, 200))  # 初始化对话框
        self.dialog.Show()  # 显示对话框
        self.Enable(False)  # 将主窗口置为不可用

        # 建立线程，开始肤色检测
        detection_thread = threading.Thread(target=self.__detection)
        detection_thread.start()  # 启动线程

    def __detection(self):
        """
        开始肤色检测
        :return: None
        """
        process = image.ImageProcess()
        image_list = process.skin_detection(self.dialog, self.image)
        self.image_list += image_list

        # 肤色检测完毕，关闭弹窗
        self.dialog.close()
        self.Enable(True)  # 将主窗口置为可用

        size, pos = self.__get_size_and_position()
        # 将标题显示出来，将所有图片重置，将除原图外的所有图隐藏，将所有图像大小位置重置
        for i in range(1, len(self.image_list)):
            self.title_list[i].Show()
            self.window_list[i].set_image(self.image_list[i])
            self.window_list[i].Hide()
            self.window_list[i].SetSize(size[0], size[1])
            self.window_list[i].SetPosition(wx.Point(pos[i][0], pos[i][1]))
        if self.pattern is False:   # 全屏模式下，将前3个图像隐藏，将最后一个图像显示,选中末图
            for i in range(0, len(self.image_list)):
                self.window_list[i].Hide()
            self.window_segmentation.Show()
            self.window_selected = self.window_segmentation
        else:       # 列表模式下，将所有图像显示
            for i in range(0, len(self.image_list)):
                self.window_list[i].Show()
            self.window_selected = None

        # 标题颜色重置
        self.__change_title_color()

    def __change_title_color(self):
        """
        根据当前模式及选中状态，将相应图像的标题置为红色
        :return: None
        """
        if self.pattern is False:   # 全屏
            # 全屏模式下，将选中的图像标题置为红色，其余置为黑色
            for i in self.title_list:
                i.SetForegroundColour('black')
            space = self.window_list.index(self.window_selected)
            self.title_list[space].SetForegroundColour('red')
            self.update_title()
        else:   # 列表模式
            # 列表模式下，初始时，所有标题全是黑色的
            for i in self.title_list:
                if i in self.list_selected:
                    i.SetForegroundColour('red')
                else:
                    i.SetForegroundColour('black')

            self.update_title()

    def update_title(self):
        """
        刷新标题栏
        :return: None
        """
        self.title_panel.Hide()
        self.title_panel.Show()

    def OnChangeSelected(self, event):
        """
        切换视图
        :param event: 事件源
        :return: None
        """
        btn = event.GetButtonObj()
        if self.pattern is False:   # 全屏模式
            self.window_selected.Hide()
            space = self.title_list.index(btn)
            self.window_selected = self.window_list[space]
            self.__change_title_color()
            self.window_selected.Show()
        else:   # 列表模式
            space = self.title_list.index(btn)
            if self.window_list[space].get_status():
                self.window_list[space].set_status(False)
                self.list_selected.remove(self.title_list[space])
            else:
                self.window_list[space].set_status(True)
                self.list_selected.append(self.title_list[space])

            self.__change_title_color()

    def __set_pattern(self, boolean):
        """
        改变显示模式
        :param boolean: 模式
        :return: None
        """
        pass

    def OnChangeCheckBoxStatus(self, event):
        """
        当视图状态发生改变时，改变相应的颜色
        :param event:
        :return:
        """
        check_box = event.GetEventObject()
        window = check_box.GetParent()
        status = check_box.GetValue()
        space = self.window_list.index(window)
        if status:  # 状态改为选中
            self.list_selected.append(self.title_list[space])
        else:   # 状态改为未选中
            if self.title_list[space] in self.list_selected:
                self.list_selected.remove(self.title_list[space])
        self.__change_title_color()

    def OnStepDelete(self, event):
        """
        单步清除
        :param event: 事件源
        :return: None
        """
        # # 单步清除时，将模式置为列表模式，同时暂时不可改变模式
        # if self.pattern is False:
        #     self.OnConver(None)
        # # 当前图像列表不为空时，清除最后一个图像，将最后一个视图及其标题隐藏
        # image_len = len(self.image_list)
        # if image_len != 0:
        #     self.image_list.pop()
        #     self.title_list[image_len - 1].Hide()
        #     self.window_list[image_len - 1].Hide()
        #     # 当所有图像清空时，恢复初始状态（调用清空图像函数）
        #     if len(self.image_list) == 0:
        #         self.OnDeleteAll(None)
        #     else:
        #         if len(self.image_list) == 1:
        #             self.OnConver(None)
        print("暂时没写")

    def OnDrawPlot(self, event):
        """
        生成折线图并显示
        :param event:
        :return:
        """
        self.child_frame = controls.ShowImage(title="样本分布(Cb、Cr)图", size=(1200, 700))  # 初始化对话框
        self.child_frame.Show()  # 显示对话框
        self.child_frame.set_icon("../images/distribution01.ico")

        # 建立线程，开始显示折线图
        draw_plot_thread = threading.Thread(target=self.__draw_plot)
        draw_plot_thread.start()  # 启动线程

    def __draw_plot(self):
        """
        开始显示折线图
        :param frame:
        :return:
        """
        draw_thread = threading.Thread(target=self.__draw)
        draw_thread.start()  # 启动线程
        draw_thread.join()
        # 开始显示图片
        self.child_frame.set_show_image(["../TempInfo/Sample Cb-Value Distribution Image.jpg",
                                         "../TempInfo/Sample Cr-Value Distribution Image.jpg"], self.child_frame)

    def __draw(self):
        """
        生成折线图
        :return: None
        """
        image_process = image.ImageProcess()
        image_process.plot_of_number_dict()

    def OnGetSkinLikelihood(self, event):
        """
        计算肤色似然度，得到肤色似然图与肤色二值化图
        :return: 似然概率列表、肤色似然图、肤色似然二值化图
        """

        # 分辨事件源
        control = event.GetId()
        if control == 201 or control == 301:
            self.skin_control_type = 1
            title = "肤色似然概率"
            size = (816, 459)
            path = "../images/probability01.ico"
        elif control == 202 or control == 302:
            self.skin_control_type = 2
            title = "肤色似然图"
            size = (800, 560)
            path = "../images/likelihoodimage01.ico"
        else:
            self.skin_control_type = 3
            title = "肤色似然二值化图"
            size = (800, 560)
            path = "../images/likelihoodimage01.ico"

        # 开始弹窗，计算
        self.child_frame = controls.ShowImage(title=title, size=(size[0], size[1]))  # 初始化对话框
        self.child_frame.Show()  # 显示对话框
        self.child_frame.set_icon(path)

        # 建立线程，开始计算肤色似然度，同时生成肤色似然图与二值化图像
        skin_likelihood_thread = threading.Thread(target=self.__skin_likelihood)
        skin_likelihood_thread.start()  # 启动线程

    def __skin_likelihood(self):
        """
        开始显示图片
        :return: None
        """
        skin_thread = threading.Thread(target=self.__skin)
        skin_thread.start()  # 启动线程
        skin_thread.join()
        # 开始显示图片
        if self.skin_control_type == 1:
            self.child_frame.show_skin_probability(self.probability_list, self.image, self.child_frame)
        elif self.skin_control_type == 2:
            self.child_frame.set_show_image(["../TempInfo/Skin Likelihood Image.jpg"], self.child_frame)
        else:
            self.child_frame.set_show_image(["../TempInfo/Skin Likelihood Binarization Image.jpg"], self.child_frame)

        # 清除控件类型
        self.skin_control_type = -1

    def __skin(self):
        """
        开始计算与生成图片
        :return:
        """
        process = image.ImageProcess()
        probability_list = process.skin_likelihood(self.image)
        self.probability_list = probability_list

    def OnModelValue(self, event):
        """
        显示建模过程中计算得到的一系列数值
        :param event: 事件源
        :return: None
        """
        # 开始弹窗，显示
        self.child_frame = controls.ShowImage(title="模型数值", size=(856, 600))  # 初始化对话框
        self.child_frame.Show()  # 显示对话框
        self.child_frame.set_icon("../images/number01.ico")

        # 建立线程，开始计算肤色似然度，同时生成肤色似然图与二值化图像
        value_thread = threading.Thread(target=self.__model_value)
        value_thread.start()  # 启动线程

    def __model_value(self):
        """
        开始显示
        :return: None
        """
        value_thread = threading.Thread(target=self.__value)
        value_thread.start()  # 启动线程
        value_thread.join()

        # 开始显示图片
        self.child_frame.set_show_value(self.model_value, self.child_frame)

        # 清除数值
        self.model_value = None

    def __value(self):
        """
        显示
        :return: None
        """
        io = rw.SequenceTXTIO()
        self.model_value = io.read_sequence_from_txt("../TempInfo/模型计算所得值.txt")

    def __update_main_frame(self):
        """
        刷新主窗口
        :return: None
        """
        self.Hide()
        self.Show()

    def OnEllipse(self, event):
        """
        显示椭圆模型肤色检测并进行二值化的图像
        :param event: 事件源
        :return: None
        """
        # 开始弹窗，检测
        self.child_frame = controls.ShowImage(title="椭圆模型肤色检测图像", size=(800, 560))  # 初始化对话框
        self.child_frame.Show()  # 显示对话框
        self.child_frame.set_icon("../images/likelihoodimage01.ico")

        # 建立线程，开始计算肤色似然度，同时生成肤色似然图与二值化图像
        ellipse_thread = threading.Thread(target=self.__ellipse)
        ellipse_thread.start()  # 启动线程

    def __ellipse(self):
        process = image.ImageProcess()
        process.ellipse_detection(self.image)

        # self.child_frame.set_show_image(["../TempInfo/Skin Ellipse Detection Image_temp1.jpg",
        #                                  "../TempInfo/Skin Ellipse Detection Image_temp2.jpg"], self.child_frame)
        self.child_frame.set_show_image(["../TempInfo/Skin Ellipse Detection Image_temp1.jpg"], self.child_frame)

    def OnAbout(self, event):
        """
        显示关于软件与作者及源码的窗口
        :param event: 事件源
        :return: None
        """
        id = event.GetId()
        # print(id)
        if id == 501:   # 关于软件
            title = "关于软件"
            ico_path = "../images/soft02.ico"
            self.help_control_type = 1
            size = (756, 500)
        elif id ==502:  # 关于作者
            title = "关于作者"
            ico_path = "../images/author02.ico"
            self.help_control_type = 2
            size = (756, 450)
        else:           # 源码
            title = "源码"
            ico_path = "../images/code02.ico"
            self.help_control_type = 3
            size = (356, 250)

        # 开始弹窗，显示
        self.child_frame = controls.ShowImage(title=title, size=size)  # 初始化对话框
        self.child_frame.Show()  # 显示对话框
        self.child_frame.set_icon(ico_path)

        # 建立线程，开始计算肤色似然度，同时生成肤色似然图与二值化图像
        web_thread = threading.Thread(target=self.__about)
        web_thread.start()  # 启动线程

    def __about(self):
        """
        开始显示
        :return: None
        """
        # 开始显示Web文本
        self.child_frame.set_about(self.help_control_type, self.child_frame)

        # 清空关于控件类型
        self.help_control_type = -1

    def OnHSV(self, event):
        """
        HSV范围筛选法
        :param event: 事件源
        :return: None
        """
        # 开始弹窗，检测
        self.child_frame = controls.ShowImage(title="HSV范围筛选法二值化图像", size=(1200, 700))  # 初始化对话框
        self.child_frame.Show()  # 显示对话框
        self.child_frame.set_icon("../images/likelihoodimage01.ico")

        # 建立线程，开始计算肤色似然度，同时生成肤色似然图与二值化图像
        hsv_thread = threading.Thread(target=self.__hsv)
        hsv_thread.start()  # 启动线程

    def __hsv(self):
        process = image.ImageProcess()
        process.hsv_detection(self.image)

        self.child_frame.set_show_image(["../TempInfo/Skin HSV Detection Image_temp1.jpg",
                                         "../TempInfo/Skin HSV Detection Image_temp2.jpg"], self.child_frame)

    def OnCr_Otsu(self, event):
        """
        YCrCb颜色空间的Cr分量+Otsu阈值分割
        :param event: 事件源
        :return: None
        """
        # 开始弹窗，检测
        self.child_frame = controls.ShowImage(title="YCrCb颜色空间的Cr分量+Otsu阈值分割", size=(1200, 700))  # 初始化对话框
        self.child_frame.Show()  # 显示对话框
        self.child_frame.set_icon("../images/likelihoodimage01.ico")

        # 建立线程，开始计算肤色似然度，同时生成肤色似然图与二值化图像
        cr_otsu_thread = threading.Thread(target=self.__cr_otsu)
        cr_otsu_thread.start()  # 启动线程

    def __cr_otsu(self):
        process = image.ImageProcess()
        process.cr_otsu_detection(self.image)

        self.child_frame.set_show_image(["../TempInfo/Skin Cr_Otsu Detection Image_temp1.jpg",
                                         "../TempInfo/Skin Cr_Otsu Detection Image_temp2.jpg"], self.child_frame)

    def OnRGB(self, event):
        """
        RGB颜色空间的肤色检测
        :param event: 事件源
        :return: None
        """
        # 开始弹窗，检测
        self.child_frame = controls.ShowImage(title="RGB色彩空间二值化图像", size=(1200, 700))  # 初始化对话框
        self.child_frame.Show()  # 显示对话框
        self.child_frame.set_icon("../images/likelihoodimage01.ico")

        # 建立线程，开始计算肤色似然度，同时生成肤色似然图与二值化图像
        rgb_thread = threading.Thread(target=self.__rgb)
        rgb_thread.start()  # 启动线程

    def __rgb(self):
        process = image.ImageProcess()
        process.rgb_detection(self.image)

        self.child_frame.set_show_image(["../TempInfo/Skin RGB Detection Image_temp1.jpg",
                                         "../TempInfo/Skin RGB Detection Image_temp2.jpg"], self.child_frame)

    def OnCompare(self, event):
        """
        多图像比较
        :param event: 事件源
        :return: None
        """
        # 开始弹窗，检测
        self.child_frame = controls.ShowImage(title="Cb、Cr筛选+YCbCr椭圆模型+RGB+HSV", size=(1300, 750))  # 初始化对话框
        self.child_frame.Show()  # 显示对话框
        self.child_frame.set_icon("../images/compare02.ico")

        # 建立线程，开始计算肤色似然度，同时生成肤色似然图与二值化图像
        compare_thread = threading.Thread(target=self.__compare)
        compare_thread.start()  # 启动线程

    def __compare(self):
        process = image.ImageProcess()
        process.rgb_detection(self.image)

        # 将Cb、Cr范围筛选二值化图像保存
        img_temp = Image.fromarray(cv2.cvtColor(self.image_list[2], cv2.COLOR_GRAY2RGB))  # 将传参过来的OpenCV图转换成PIL.Image格式
        img_temp.save("../TempInfo/Skin CbCr Detection Image_temp1.jpg")

        self.child_frame.set_show_image(["../TempInfo/Skin CbCr Detection Image_temp1.jpg",
                                         "../TempInfo/Skin Ellipse Detection Image_temp1.jpg",
                                         "../TempInfo/Skin RGB Detection Image_temp1.jpg",
                                         "../TempInfo/Skin HSV Detection Image_temp1.jpg"], self.child_frame)

    def OnEllipseModel(self, event):
        """
        多图像比较
        :param event: 事件源
        :return: None
        """
        # 开始弹窗
        self.child_frame = controls.ShowImage(title="椭圆模型说明", size=(800, 400))  # 初始化对话框
        self.child_frame.Show()  # 显示对话框
        self.child_frame.set_icon("../images/ellipse02.ico")

        self.child_frame.set_show_image(["../TempInfo/ellipse01.jpg",
                                         "../TempInfo/ellipse_temp.jpg"], self.child_frame)

        # 建立线程
        # model_thread = threading.Thread(target=self.__ellipse_model)
        # model_thread.start()  # 启动线程

    def __ellipse_model(self):

        self.child_frame.set_show_image(["../TempInfo/Skin CbCr Detection Image_temp1.jpg",
                                         "../TempInfo/Skin Ellipse Detection Image_temp1.jpg",
                                         "../TempInfo/Skin RGB Detection Image_temp1.jpg",
                                         "../TempInfo/Skin HSV Detection Image_temp1.jpg"], self.child_frame)

    def OnExit(self, event):
        """
        退出整个程序
        :param event: 事件源
        :return: None
        """
        self.Destroy()

    def OnMultiScheme(self, event):
        """
        多方案比较窗口打开
        :param event: 事件源
        :return: None
        """
        multi_frame = controls.MultiSchemeFrame(image=self.image, title='多方案比较')
        multi_frame.Show()

    # def OnStep(self, event):
    #     """
    #     单步图像处理，，当仅仅
    #     :param event:
    #     :return:
    #     """

