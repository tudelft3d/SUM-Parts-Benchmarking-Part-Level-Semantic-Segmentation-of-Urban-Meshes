# -*- coding: utf-8 -*-
# @Author  : LG
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
from ISAT.ui.info_dock import Ui_Form
import time
import os

class InfoDockWidget(QtWidgets.QWidget, Ui_Form):
    def __init__(self, mainwindow):
        super(InfoDockWidget, self).__init__()
        self.setupUi(self)
        self.mainwindow = mainwindow
        self.image = self.mainwindow.scene.image_data
        self.mask = self.mainwindow.scene.mask_image_data
        self.progressBar.reset()
        self.lineEdit_note.textChanged.connect(self.note_changed)
        self.basic_num = 0
        self.smart_num = 0

    def note_changed(self):
        if self.mainwindow.load_finished:
            self.mainwindow.set_saved_state(False)

    def update_widget(self):
        if self.mainwindow.current_label is not None:
            self.label_width.setText('{}'.format(self.mainwindow.current_label.width))
            self.label_height.setText('{}'.format(self.mainwindow.current_label.height))
            self.label_depth.setText('{}'.format(self.mainwindow.current_label.depth))
            self.lineEdit_note.setText(self.mainwindow.current_label.note)

    def update_progress_bar(self, write_file = True):
        self.image = self.mainwindow.scene.image_data
        self.mask = self.mainwindow.scene.mask_image_data
        self.basic_num = self.mainwindow.scene.basic_operation_num
        self.smart_num = self.mainwindow.scene.smart_operation_num
        if self.mask is not None:
            valid_pixels = np.count_nonzero((self.mask != 0))

            black_pixels = np.all(self.image == [0, 0, 0], axis=-1)
            white_pixels = np.all(self.image == [255, 255, 255], axis=-1)
            total_pixels = np.sum(~(black_pixels | white_pixels))

            if valid_pixels > total_pixels:
                valid_pixels = total_pixels
            if valid_pixels >= 0 and total_pixels > 0:
                progress = valid_pixels / total_pixels if total_pixels else 0
                self.progressBar.setValue(int(progress * 100))
                self.progressBar.setFormat(f'{progress * 100:.1f}%')
                if valid_pixels > 0 and write_file:
                    elapsed_time = time.time() - self.mainwindow.scene.p_start_time
                    self.mainwindow.scene.p_start_time = time.time()
                    filename = self.mainwindow.files_list[self.mainwindow.current_index]
                    filename = filename.split('.')[0]
                    filepath = os.path.join(self.mainwindow.image_root, filename + ".txt")
                    # with open(filepath, "a") as file:
                    #     file.write(f"time {elapsed_time:.2f} : {progress:.3f}\n")
                    #     file.write(f"basic : {self.basic_num}\n")
                    #     file.write(f"smart : {self.smart_num}\n")

                    if os.path.exists(filepath):
                        with open(filepath, "r") as file:
                            lines = file.readlines()
                    else:
                        # 如果文件不存在，初始化为包含basic和smart行的列表
                        lines = ["basic : 0\n", "smart : 0\n"]

                    # 更新basic和smart行
                    lines[0] = f"basic : {self.basic_num}\n"
                    lines[1] = f"smart : {self.smart_num}\n"

                    # 追加新的时间记录
                    new_time_entry = f"time {elapsed_time:.2f} : {progress:.3f}\n"
                    if len(lines) > 2:
                        lines.append(new_time_entry)  # 如果已经存在basic和smart行，追加time数据
                    else:
                        lines += [new_time_entry]  # 如果文件是新创建的，添加time数据

                    # 重写文件
                    with open(filepath, "w") as file:
                        file.writelines(lines)