import cv2
import numpy as np
import matplotlib.pyplot as plt
from .local_utils import detect_lp
import os
from os.path import splitext,basename
from keras.models import model_from_json
from tensorflow import keras
import glob
import tensorflow as tf

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

def preprocess_image(image,resize=False):
    img = image
    # img = img[int(img.shape[0]/5*3):,100:-100]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255
    if resize:
        img = cv2.resize(img, (224,224))
    return img

def get_plate(image, wpod_net, Dmax=608, Dmin=256):
    vehicle = preprocess_image(image)
    ratio = float(max(vehicle.shape[:2])) / min(vehicle.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)
    _ , plate_image, _, coordinate, prob = detect_lp(wpod_net, vehicle, bound_dim, lp_threshold=0.5, wh_threshold = 1.3)
    return plate_image, coordinate, prob

class LP_Detect:
    __shared_instance = None
    @staticmethod
    def getInstance(): 
        """Static Access Method"""
        if LP_Detect.__shared_instance == None: 
            LP_Detect() 
        return LP_Detect.__shared_instance
    def __init__(self,):
        self.coordinate = None
        self.plate_type = None
        if LP_Detect.__shared_instance != None: 
            raise Exception ("This class is a singleton class !") 
        else: 
            # Singleton Pattern Design only instantiate the model once
            self.model = tf.keras.models.load_model('ai/license_plate/lp_detection/Classification_license_plate.h5')
            self.wpod_net = load_model("ai/license_plate/models/wpod-net.json") 
            LP_Detect.__shared_instance = self

    def detect(self, image, classify=False):
        
        plate_image, self.coordinate, self.prob = get_plate(image, self.wpod_net)
        plate_image = (255*plate_image[0]).astype(np.uint8)
        if classify:
            plate = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
            plate = cv2.resize(plate, (64, 64))
            plate = np.expand_dims(plate, axis=0)
            result=self.model.predict(plate).squeeze()
            # print(result)

            if result[0] > result[1]:
                self.plate_type = 1
                plate_image = cv2.resize(plate_image, (470, 110))
            else:
                self.plate_type = 2
                plate_image = cv2.resize(plate_image, (280, 200))
                plt.imshow(plate_image)
                plt.show()

        return plate_image, self.prob[0], self.plate_type

    def get_coord(self):
        return self.coordinate

    def get_prob(self):
        return self.prob[0]

    def get_plate_type(self):
        return self.plate_type