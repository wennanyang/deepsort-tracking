import cv2
import os

IMAGE_DIR = "/home/ywn/reID/YOLOv5-DeepSort_Pytorch/mot_dataset/plane/039/img"
OUTPUT_DIR = "/home/ywn/reID/YOLOv5-DeepSort_Pytorch/video/plane39.mp4"

images = [img for img in sorted(os.listdir(IMAGE_DIR)) if img.endswith(".jpg")]

img = cv2.imread(os.path.join(IMAGE_DIR, images[0]))

height, width, layers = img.shape

video = cv2.VideoWriter(OUTPUT_DIR, cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))
for image in images:
    img = cv2.imread(os.path.join(IMAGE_DIR, image))
    video.write(img)

video.release()
