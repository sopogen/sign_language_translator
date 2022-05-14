import cv2
from img2vec_pytorch import Img2Vec
import os
from PIL import Image
import torch

folder_path = "/mnt/d/data/KETI/36878~40027/"
file_list = os.listdir(folder_path)

def video_to_image(file_path):
    file_name = file_path.split('/')[-1].split('.')[0]
    video_capture = cv2.VideoCapture(file_path)
    video_capture.set(cv2.CAP_PROP_FPS, 25)

    saved_frame_idx = 0
    while video_capture.isOpened():
        frame_is_read, frame = video_capture.read()

        if frame_is_read:
            cv2.imwrite(f"{file_name}_frame_{str(saved_frame_idx)}.jpg", frame)
            saved_frame_idx += 1

        else:
            print("Could not read the frame.")

def image_to_vector(file_path):
    img2vec = Img2Vec(cuda=False, model="efficientnet_b0")
    img = Image.open(file_path)
    vec = img2vec.get_vec(img, tensor=True)
    pooling = torch.nn.AdaptiveAvgPool2d(output_size=1)
    vec = pooling(vec).squeeze(3).squeeze(2)
    return vec