
import os, sys
import cv2
import logging
import time
import imutils

from ai.license_plate.lp_detection.detect import LP_Detect
# from ai.license_plate.lp_recognition.recognize import LP_Recognize
from ai.license_plate.lp_recognition.recognize_crnn import LPRCNN
from ai.license_plate.car_detection.detect import CarDetection
from api.utils import generate_image_file_name
from api.exceptions import ErrorResponse

class LPRecognition:

    def __init__(self, data_dir):
        self.lp_detection = LP_Detect()
        # self.lp_recognition = LP_Recognize()
        self.lp_recognition = LPRCNN()
        self.car_detection = CarDetection()

        self.lp_dir = os.path.join(data_dir, 'license_plates')

        # create license plate folder
        os.makedirs(self.lp_dir, exist_ok=True)

    def recognize(self, image):
        image_name, suffix_name = generate_image_file_name('lp')
        image_path = os.path.join(self.lp_dir, image_name)
        cv2.imwrite(image_path, image)
        logging.info(f'save image: {image_path}')

        # Detect license plate first
        detection_conf = 0
    
        start = time.time()
        car_image, car_type = self.car_detection.car_detect(image)

        try:
            start = time.time()
            car_image_resized = imutils.resize(car_image[0], width=1000)

            lp_image, detection_conf, plate_type = self.lp_detection.detect(car_image_resized, classify=True, car_type=car_type)

            logging.info(f'detect license plate: {time.time() - start}s')
            logging.info(f'image: {image_path} license plate detection confident: {detection_conf}')
        except:
            try:
                if car_image[1].size != 0:
                    start = time.time()
                    car_image_resized = imutils.resize(car_image[1], width=1000)
                    lp_image, detection_conf, plate_type = self.lp_detection.detect(car_image_resized, classify=True, car_type=car_type)
                    logging.info(f'detect license plate: {time.time() - start}s')
                    logging.info(f'image: {image_path} license plate detection confident: {detection_conf}')
            except:
                raise ErrorResponse(406)

        recognition_conf = 0
        try:
            start = time.time()
            lp_text, recognition_conf = self.lp_recognition.rec(lp_image, mode=plate_type)
            logging.info(f'recognize license plate: {time.time() - start}s')

            logging.info(f'image: {image_path} license plate recognition confident: {recognition_conf}')
        except:
            raise ErrorResponse(407)
        
        detection_conf = int(round(detection_conf * 100))
        recognition_conf = int(round(recognition_conf * 100))

        lp_image_path = ''
        
        # Save image
        if lp_image is not None:
            lp_image_path = os.path.join(self.lp_dir, suffix_name)
            cv2.imwrite(os.path.join(self.lp_dir, suffix_name), lp_image)

        return  {
                    'text': lp_text,
                    'detection_conf' : str(detection_conf),
                    'recognition_conf' : str(recognition_conf),
                    'image_path' : lp_image_path
                }