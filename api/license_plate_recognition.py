
import os, sys
import cv2

from ai.license_plate.lp_detection.detect import LP_Detect
from ai.license_plate.lp_recognition.recognize import LP_Recognize
from ai.license_plate.car_detection.detect import CarDetection
from api.utils import generate_image_file_name

class LPRecognition:

    def __init__(self, data_dir):
        self.lp_detection = LP_Detect()
        self.lp_recognition = LP_Recognize()
        self.car_detection = CarDetection()

        self.lp_dir = os.path.join(data_dir, 'license_plates')

        # create license plate folder
        os.makedirs(self.lp_dir, exist_ok=True)

    def recognize(self, image):
        # Detect license plate first
        car_image = self.car_detection.car_detect(image)
        lp_image = self.lp_detection.detect(car_image, classify=True)
        detect_prob = self.lp_detection.get_prob()
        plate_type = self.lp_detection.get_plate_type()
        lp_text = self.lp_recognition.rec(lp_image, mode=plate_type)
        
        # Save image
        if lp_image is not None:
            image_name = generate_image_file_name()
            lp_image_path = os.path.join(self.lp_dir, image_name)
            cv2.imwrite(lp_image_path, lp_image)

        return lp_image_path, lp_text, detect_prob
