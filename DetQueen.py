import datetime
import sys
import os
import cv2
from PyQt5.QtCore import Qt, QTimer, QUrl
from PyQt5.QtGui import QImage, QPixmap, QDesktopServices
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, \
    QComboBox, QFileDialog, QTextEdit

from model_run.model_run import run_yolo


class ObjectDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.fixed_width = 640  # 设置显示图像固定的宽度
        self.fixed_height = 640  # 设置显示图像固定的高度
        self.initUI()
        # self.initPlaceholderLabels()  # 初始化占位 QLabel
        self.yoloweightPath = {'yolov8n': ['model_weight', 'yolov8_finetuned_model', 'yolov8n_best.pt'],
                               'yolov8s': ['model_weight', 'yolov8_finetuned_model', 'yolov8s_best.pt'],
                               'yolov8m': ['model_weight', 'yolov8_finetuned_model', 'yolov8m_best.pt'], }
        self.model_selceted_path = self.yoloweightPath['yolov8n']  # 默认相对路径，采取8n模型
        self.img_path = None
        self.video_path = None
        self.store_base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')

    def initUI(self):
        self.setWindowTitle("蜂王目标检测平台")
        self.setGeometry(100, 100, 800, 800)

        # 创建主窗口
        main_widget = QWidget(self)
        self.setCentralWidget(main_widget)

        # 创建布局
        main_layout = QVBoxLayout()
        Operation_layout = QHBoxLayout()
        terminal_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        left_top_layout = QVBoxLayout()
        left_bottom_layout = QVBoxLayout()
        right_layout = QHBoxLayout()

        # 创建按钮和控件
        input_image_button = QPushButton("选择输入图片")
        input_video_button = QPushButton("选择输入视频")
        start_detection_button = QPushButton("开始检测")
        open_store_directory = QPushButton("打开文件存储地址")
        # 模型选择添加选项
        model_combo = QComboBox()
        model_combo.addItem("yolov8n")
        model_combo.addItem("yolov8s")
        model_combo.addItem("yolov8m")

        left_top_layout.addWidget(QLabel("选择模型:"))
        left_top_layout.addWidget(model_combo)

        # 创建标签用于显示检测结果
        num_results_text_label = QLabel("检测数值结果:")
        # num_results_text_label.setStyleSheet("border: 2px solid balck; padding: 4px; background-color: white; color: black;")
        num_results_text_label.setAlignment(Qt.AlignBottom)
        self.result_label = QLabel()
        self.result_label.setStyleSheet("background-color: white")
        # self.result_label.setFixedSize(100,200)
        left_bottom_layout.addWidget(num_results_text_label)
        left_bottom_layout.addWidget(self.result_label)

        # 创建右边两个子区域
        right_child1 = QVBoxLayout()
        right_child2 = QVBoxLayout()
        # 两个子区域的文本label
        input_text_label = QLabel("输入图像或视频")
        # input_text_label.setStyleSheet("border: 2px solid balck; padding: 4px; background-color: white; color: black;")
        input_text_label.setAlignment(Qt.AlignCenter)
        # input_text_label.setMinimumSize(20,40)
        result_text_label = QLabel("检测图像或视频")
        # result_text_label.setStyleSheet("border: 2px solid black; padding: 4px; background-color: white; color: black;")
        result_text_label.setAlignment(Qt.AlignCenter)
        # 两个子区域的显示label
        self.input_image_label = QLabel()
        self.input_image_label.setStyleSheet("background-color: black;")
        self.input_image_label.setFixedSize(self.fixed_height, self.fixed_width)
        self.result_image_label = QLabel()
        self.result_image_label.setStyleSheet("background-color: black;")
        self.result_image_label.setFixedSize(self.fixed_height, self.fixed_width)
        # 将文本和显示添加到对应布局
        right_child1.addWidget(input_text_label)
        right_child1.addWidget(self.input_image_label)
        right_child2.addWidget(result_text_label)
        right_child2.addWidget(self.result_image_label)

        right_layout.addLayout(right_child1)
        right_layout.addLayout(right_child2)
        # 添加到主布局
        left_layout.addWidget(input_image_button)
        left_layout.addWidget(input_video_button)
        left_layout.addWidget(start_detection_button)
        left_layout.addWidget(open_store_directory)

        left_layout.addLayout(left_top_layout)
        left_layout.addLayout(left_bottom_layout)

        Operation_layout.addLayout(left_layout)
        Operation_layout.addLayout(right_layout)
        # 创建一个文本编辑框用于显示终端信息
        self.text_edit = QTextEdit(self)
        self.text_edit.setReadOnly(True)
        terminal_layout.addWidget(self.text_edit)
        self.text_edit.insertPlainText('蜂王目标检测模型默认为: yolov8n \n')

        # container = QWidget()
        # container.setLayout(terminal_layout)
        # self.setCentralWidget(container)

        main_layout.addLayout(Operation_layout)
        main_layout.addLayout(terminal_layout)

        main_widget.setLayout(main_layout)

        # 连接按钮点击事件
        model_combo.currentIndexChanged.connect(self.on_combo_box_item_selected)
        input_image_button.clicked.connect(self.load_input_image)
        input_video_button.clicked.connect(self.load_input_video)
        start_detection_button.clicked.connect(self.start_detection)
        open_store_directory.clicked.connect(self.openFileExplorer)

    def on_combo_box_item_selected(self, index):
        self.model_selceted_path = self.yoloweightPath[list(self.yoloweightPath.keys())[index]]
        self.text_edit.insertPlainText(f'检测模型已更换: {list(self.yoloweightPath.keys())[index]}\n')

    def load_input_image(self):
        # 添加选择图片的逻辑
        self.text_edit.insertPlainText(f'正在选择输入图片\n')
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("Images (*.png *.xpm *.jpg )")
        file_dialog.setViewMode(QFileDialog.List)  # 以纵向列表的方式显示文件
        file_dialog.setFileMode(QFileDialog.ExistingFiles)  # 可以一次性选择多个图片作为目标检测器的输入

        if file_dialog.exec_():
            file_paths = file_dialog.selectedFiles()
            if file_paths:
                image_path = file_paths[0]
                self.img_path = image_path
                input_image = cv2.imread(image_path)
                input_image = cv2.resize(input_image, (self.fixed_width, self.fixed_height))
                self.display_image(input_image, self.input_image_label)
        self.text_edit.insertPlainText(f'已选择输入图片\n'
                                       f'图片输入路径:{self.img_path}\n')
        # 如果输入图片，则清空视频路径
        self.video_path = None

    def display_image(self, image, label):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.fixed_width, self.fixed_height))
        height, width, channel = image.shape
        # 采取640x640为显示框分辨率
        bytes_per_line = 3 * width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        label.setPixmap(pixmap)

    def load_input_video(self):
        # 添加选择视频的逻辑
        self.text_edit.insertPlainText(f'正在选择输入视频\n')
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("Videos (*.mp4 *.api)")
        file_dialog.setViewMode(QFileDialog.List)  # 以纵向列表的方式显示文件
        file_dialog.setFileMode(QFileDialog.ExistingFiles)  # 可以一次性选择多个图片作为目标检测器的输入

        if file_dialog.exec_():
            file_paths = file_dialog.selectedFiles()
            if file_paths:
                video_path = file_paths[0]
                self.video_path = video_path
                self.video_input = cv2.VideoCapture(video_path)
                # 只显示视频的第一帧，暂不进行播放
                self.display_frame(self.video_input, self.input_image_label)
        self.text_edit.insertPlainText(f'已选择输入视频\n'
                                       f'视频输入路径：{self.video_path}\n'
                                       f'视频处理时间较长，时长不宜过长，视频检测处理时间大概为视频时长9倍。\n')
        # 如果输入视频，则清空图片路径
        self.img_path = None

    def display_frame(self, video, image_label):
        ret, frame = video.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.fixed_width, self.fixed_height))
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            image_label.setPixmap(pixmap)

    def time2str(self):
        current_time = datetime.datetime.now()
        formatted_time = current_time.strftime("%Y%m%d%H%M%S")
        return formatted_time

    def abs_dir(self, file_relative_dir):
        current_script_path = os.path.abspath(__file__)
        parent_dir = os.path.dirname(current_script_path)
        file_abs_dir = os.path.join(parent_dir, *file_relative_dir)
        return file_abs_dir

    def start_detection(self):
        # 添加目标检测逻辑
        # 模型的绝对路径
        self.absolute_model_dir = self.abs_dir(self.model_selceted_path)

        if self.img_path and not self.video_path:
            self.store_path = os.path.join(self.store_base_path, f'img_{self.time2str()}')
            self.text_edit.insertPlainText(f'图片的存储地址为：{self.store_path}\n'
                                           f'正在检测图片中······\n')
            os.makedirs(self.store_path, exist_ok=True)
            img_result_dict = run_yolo(self.absolute_model_dir, self.img_path, None, self.store_path)
            # 检测文本结果
            self.result_label.setText(f"蜂后：{img_result_dict['queen']}\n工蜂：{img_result_dict['worker']}")
            # 检测图像结果
            result_img_path = os.path.join(self.store_path, os.path.basename(self.img_path))
            result_img = cv2.imread(result_img_path)
            self.display_image(result_img, self.result_image_label)
            self.text_edit.insertPlainText(f'图片检测完成。\n')
        elif not self.img_path and self.video_path:
            self.store_path = os.path.join(self.store_base_path, f'video_{self.time2str()}')
            self.text_edit.insertPlainText(f'视频的存储地址为：{os.path.join(self.store_path, "predict")}\n')
            os.makedirs(self.store_path, exist_ok=True)
            self.video_result_dict = run_yolo(self.absolute_model_dir, None, self.video_path, self.store_path)
            video_result_name = os.path.basename(self.video_path).split('.')[0]
            # 检测完成后，开始同步播放输入视频、检测视频、检测数值结果（逐帧播放）
            self.video_output = cv2.VideoCapture(
                os.path.join(self.store_path, 'predict', f'{video_result_name}.avi'))  # yolov8模型将结果自动保存在predict文件夹中

            # 创建一个变量跟踪帧数
            self.frame_count = 0
            # 创建一个定时器来更新视频帧
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.update_frames)
            self.timer.start(30)  # 更新频率，单位毫秒
            # 结束
            self.text_edit.insertPlainText(f'已完成对视频的检测\n')
        else:
            self.text_edit.insertPlainText('错误路径：图片和视频路径都不存在。\n')

    def update_frames(self):
        ret1, frame1 = self.video_input.read()
        ret2, frame2 = self.video_output.read()
        # 首先显示文本
        if self.frame_count >= len(self.video_result_dict):
            self.text_edit.insertPlainText(f'视频帧率提取设置出错')
        else:
            current_frame_cls_dict = self.video_result_dict[self.frame_count]
            self.result_label.setText(f"蜂后：{current_frame_cls_dict['queen']}\n工蜂：{current_frame_cls_dict['worker']}")
            if ret1:
                self.frame_count += 1
                frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
                frame1 = cv2.resize(frame1, (self.fixed_width, self.fixed_height))
                image1 = QImage(frame1, frame1.shape[1], frame1.shape[0], QImage.Format_RGB888)
                pixmap1 = QPixmap.fromImage(image1)
                self.input_image_label.setPixmap(pixmap1)
                self.input_image_label.setAlignment(Qt.AlignCenter)

            if ret2:
                frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
                frame2 = cv2.resize(frame2, (self.fixed_width, self.fixed_height))
                image2 = QImage(frame2, frame2.shape[1], frame2.shape[0], QImage.Format_RGB888)
                pixmap2 = QPixmap.fromImage(image2)
                self.result_image_label.setPixmap(pixmap2)
                self.result_image_label.setAlignment(Qt.AlignCenter)

    def openFileExplorer(self):
        # 指定要打开的文件夹路径
        if self.img_path and not self.video_path:
            # 使用QDesktopServices打开文件资源管理器并显示指定文件夹
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.store_path))
        elif not self.img_path and self.video_path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.join(self.store_path, 'predict')))
        else:
            self.text_edit.insertPlainText(f'还没有检测\n')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ObjectDetectionApp()
    window.show()
    sys.exit(app.exec_())
