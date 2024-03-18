import cv2
import os
from PIL import Image, ImageFilter
import shutil
from tqdm import tqdm

ROOT = "/home/ywn/reID/YOLOv5-DeepSort_Pytorch/"
plane_dir = "mot_dataset/plane"

output_path_images = "mot_yolo_all/images"
output_path_labels = "mot_yolo_all/labels"
output_path_cropped = "mot_yolo_all/cropped/train"
os.makedirs(output_path_images, exist_ok=True)
os.makedirs(output_path_labels, exist_ok=True)
os.makedirs(output_path_cropped, exist_ok=True)

def write_labels_images(gt_image_dir):
    gt_path = os.path.join(gt_image_dir, "gt/gt.txt")
    with open(gt_path, "r") as f:
        lines = f.readlines()
    with tqdm(total = len(lines), desc="Transforming", unit="img") as pbar:
        for line in lines:
            parts = list(map(int, line.split()[0:7]))

            x1, y1, x2, y2 = parts[2], parts[3], parts[4], parts[5]
            img = f"{str(parts[0]).zfill(6)}.jpg"
            image_name = os.path.join(gt_image_dir, f"img/{img}")
            image = Image.open(image_name)
            #得到图像的长和宽
            (width, height) = image.size
            # 计算得到归一化的中心点坐标
            x0, y0 = (x1 + x2) / (2 * width), (y1 + y2) / (2 * height)
            #计算得到目标的归一化宽和长
            w0, h0 = (x2 - x1) / width, (y2 - y1) / height
            # 得到每个类别下的数字 和 类别名称
            number, mot_class= gt_image_dir.split("/")[-1], gt_image_dir.split("/")[-2]
            
            # 设置保存的label文件名和images文件名
            label_save_name = os.path.join(output_path_labels, f"{mot_class}{number}_{img.split('.')[0]}.txt")
            image_save_name = os.path.join(output_path_images, f"{mot_class}{number}_{img}")
            # 保存得到的坐标信息
            with open(label_save_name, "a+") as f:
                f.write(" ".join([str(num) for num in [parts[1], x0, y0, w0, h0]]))
                f.write("\n")
            # 将图片换个名字保存，以便训练
            shutil.copy(image_name, image_save_name)
            # 看每个类别数, 如果该类别下的文件夹不存在，则创建
            output_path_cropped_classes = os.path.join(output_path_cropped, f"{mot_class}{parts[1]}")
            if not os.path.exists(output_path_cropped_classes):
                os.mkdir(output_path_cropped_classes)

            image_cropped_name = os.path.join(output_path_cropped, f"{mot_class}{parts[1]}/{number}_{img.split('.')[0]}_{parts[1]}_cropped.jpg")
            

            image_cropped = image.crop((x1, y1, x2, y2))
            # w_crop, h_crop = image_cropped.size
            # image_5times = image_cropped.resize((w_crop * 5, h_crop * 5))
            image_cropped.save(image_cropped_name)
            pbar.update(1)

plane_dir = os.path.join(ROOT, "mot_dataset/plane")
plane_numbers = os.listdir(plane_dir)
for numbers in sorted(plane_numbers):
    gt_image_dir = os.path.join(plane_dir, numbers)
    write_labels_images(gt_image_dir=gt_image_dir)
    break