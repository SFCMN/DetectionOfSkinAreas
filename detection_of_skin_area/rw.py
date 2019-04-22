class SequenceTXTIO:
    def __init__(self):
        pass

    def read_sequence_from_txt(self, file_path_name):
        """
        从指定文件中读取数据并转化为序列，并返回
        :param file_path_name:
        :return: 序列
        """
        value_sequence_str = ""
        file = open(file_path_name, "r+", encoding='utf-8')
        sequence = file.read(1024)
        while sequence != "":
            value_sequence_str = value_sequence_str + sequence
            sequence = file.read(1024)
        file.close()
        return eval(value_sequence_str)

    def write_sequence_into_txt(self, file_path_name, value_sequence):
        """
        将指定的序列写入到指定文件中
        :param file_path_name: 指定文件
        :param value_sequence: 指定序列
        :return: None
        """
        file = open(file_path_name, "w+", encoding='utf-8')
        file.write(str(value_sequence))
        file.flush()
        file.seek(0)
        file.close()
