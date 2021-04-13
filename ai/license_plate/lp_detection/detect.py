import cv2
import numpy as np
import matplotlib.pyplot as plt
from .local_utils import detect_lp
import os
from os.path import splitext,basename
from keras.models import model_from_json
from tensorflow import keras
import glob

def load_model(path):
    try:
        path = splitext(path)[0]
        with open('%s.json' % path, 'r') as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json, custom_objects={})
        model.load_weights('%s.h5' % path)
        print("Loading model successfully...")
        return model
    except Exception as e:
        print(e)

def preprocess_image(image, resize=False):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = img / 255
    if resize:
        img = cv2.resize(img, (224,224))
    return img

def get_plate(image, wpod_net, Dmax=608, Dmin=256):
    vehicle = preprocess_image(image)
    ratio = float(max(vehicle.shape[:2])) / min(vehicle.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)
    _ , plate_image, _, coordinate = detect_lp(wpod_net, vehicle, bound_dim, lp_threshold=0.5, wh_threshold = 1)
    return plate_image, coordinate

class LP_Detect:
    def __init__(self, model_path = "ai/models/wpod-net.json"):
        self.wpod_net = load_model(model_path)

    def detect(self, image):
        plate_image, coordinate = get_plate(image, self.wpod_net)
        return plate_image


