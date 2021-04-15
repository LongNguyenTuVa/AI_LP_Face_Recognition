
import os, sys
import cv2

from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '../../ai')))

from license_plate.lp_detection.detect import LP_Detect
from license_plate.lp_recognition.recognize import LP_Recognize
from utils import generate_image_file_name

class LPRecognition:

    def __init__(self, data_dir):
        self.lp_detection = LP_Detect()
        self.lp_recognition = LP_Recognize()
        self.lp_dir = os.path.join(data_dir, 'license_plates')

        # create license plate folder
        os.makedirs(self.lp_dir, exist_ok=True)

    def recognize(self, image):
        # Detect license plate first
        lp_image = self.lp_detection.detect(image)
        lp_text = self.lp_recognition.rec(lp_image)
        
        # Save image
        if lp_image is not None:
            image_name = generate_image_file_name()
            lp_image_path = os.path.join(self.lp_dir, image_name)
            cv2.imwrite(lp_image_path, lp_image[0]*255)

        return lp_image_path, lp_text
