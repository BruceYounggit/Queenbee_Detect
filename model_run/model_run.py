import os
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

def show_box(yolo_img_result, img_path, store_path):
    boxes = yolo_img_result[0].boxes.boxes  # [n,6]
    cls = yolo_img_result[0].boxes.cls
    original_image = yolo_img_result[0].orig_img
    detected_image = original_image
    img_name = os.path.basename(img_path)
    final_dir = os.path.join(store_path, img_name)
    for i in range(cls.shape[0]):
        item = cls[i].item()
        x1, y1, x2, y2 =list(map(lambda x:int(x), boxes[i][:4]))
        # 计算边框的左上角和右下角坐标
        # x1, y1 = int(x - w / 2), int(y - h / 2)
        # x2, y2 = int(x + w / 2), int(y + h / 2)
        thickness = 2  # 边框线宽度
        # queen
        if item == 0:
            color = (0, 0, 255)  # 红色 (BGR 格式)
        else:
            color = (255, 0, 0)  # 蓝色
        detected_image = cv2.rectangle(detected_image, (x1, y1), (x2, y2), color, thickness)
    cv2.imwrite(final_dir, detected_image)


def run_yolo(model_path, img_path, video_path, store_path):
    # Load a model
    model = YOLO(model_path)
    # 测试单张图片
    if img_path and not video_path:
        result = model(img_path)
        # 检测的图像保存
        show_box(result, img_path, store_path)
        # 检测的文本信息，如各个类别的数量
        cls_dict = {'queen': 0, 'worker': 0}
        cls_dict['worker'] += int(result[0].boxes.cls.sum().item())
        cls_dict['queen'] += result[0].boxes.cls.shape[0] - cls_dict['worker']
        return cls_dict
    # 测试视频
    elif video_path and not img_path:
        results = model(video_path, save=True, project=store_path)
        videoframes_list = []
        for result in results:
            cls_dict = {'queen': 0, 'worker': 0}
            cls_dict['worker'] += int(result.boxes.cls.sum().item())
            cls_dict['queen'] += result.boxes.cls.shape[0] - cls_dict['worker']
            videoframes_list.append(cls_dict)
        return videoframes_list
    else:
        raise Exception('同一时间只能输入图片或视频之一，不能同时输入。')


def vis(x):
    plt.imshow(x)
    plt.show()


if __name__ == '__main__':
    model_path = r'../model_weight/yolov8_finetuned_model/yolov8n_best.pt'
    img_path = r'../imgs/img_1.png'
    video_path = r'../videos/small_queen.mp4'
    store_path = r'../results'
    run_yolo(model_path, img_path=img_path, video_path=None, store_path=store_path)
