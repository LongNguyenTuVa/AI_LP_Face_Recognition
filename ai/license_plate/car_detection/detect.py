import glob
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from ai.license_plate.models.experimental import attempt_load
import os
from ai.license_plate.utils.general import  non_max_suppression,  \
    scale_coords, xyxy2xywh,  increment_path

img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

class CarDetection:  # for inference
    __shared_instance = None
    @staticmethod
    def getInstance(): 
        """Static Access Method"""
        if CarDetection.__shared_instance == None: 
            CarDetection() 
        return CarDetection.__shared_instance
    def __init__(self):
        self.device = 'cpu'
        self.img_size = 640
        if CarDetection.__shared_instance != None: 
            raise Exception ("This class is a singleton class !") 
        else: 
            # Singleton Pattern Design only instantiate the model once
            import sys
            sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
            self.model = attempt_load('yolov5s.pt', map_location=self.device)   
            CarDetection.__shared_instance = self
        
    def car_detect(self, img0):
        stride = int(self.model.stride.max())  # model stride
        # dataset = LoadImages(self.image_path, img_size=640, stride=stride)
        # img0 = cv2.imread(image_path)  # BGR
        # Padded resize
        img = letterbox(img0, self.img_size, stride=stride)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred = self.model(img, augment=True)[0]
        pred = non_max_suppression(pred, 0.2, 0.45, classes=[2, 5, 7], agnostic=True)
        xy = []
        for i, det in enumerate(pred):  # detections per image
            #frame = getattr(dataset, 'frame', 0)
            gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                    w = xywh[2] * img0.shape[1]
                    h = xywh[3] * img0.shape[0]
                    xy.append(w * h)

                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()

                    x = xywh[0] * img0.shape[1]
                    y = xywh[1] * img0.shape[0]
                    w = xywh[2] * img0.shape[1]
                    h = xywh[3] * img0.shape[0]
                    if w * h == max(xy):
                        x1 = int(x - w / 2)
                        x2 = int(x + w / 2)
                        y1 = int(y)
                        y2 = int(y + h / 2)
                        img0 = img0[y1:y2, x1:x2]
            else:
                img0 = img0[int(img0.shape[0] / 2):, :]
        return img0

