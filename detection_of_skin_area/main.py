from detection_of_skin_area import rw  # 文件读写模块
import wx   # wxPython模块
import gui  # GUI模块

# 初始化一个文件读写对象
io = rw.SequenceTXTIO()

# 从TXT文件中读取【控件Label】
label_dict_list = io.read_sequence_from_txt("../config/labeldict.txt")
# print(label_dict_list)

# 从TXT文件中读取【个人设置】
personal_settings_dict = io.read_sequence_from_txt("../config/personalsettings.txt")
# print(personal_settings_dict)

app = wx.App(False)
main_frame = gui.MainFrame(label_dict_list, personal_settings_dict)
app.MainLoop()
