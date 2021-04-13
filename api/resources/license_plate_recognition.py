
import os, sys

from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '../../ai')))

from license_plate.lp_detection.detect import LP_Detect
from license_plate.lp_recognition.recognize import LP_Recognize

class LPRecognition:

    def __init__(self):
        self.lp_detection = LP_Detect()
        self.lp_recognition = LP_Recognize()

    def recognize(self, image):
        # Detect license plate first
        lp_image = self.lp_detection.detect(image)
        lp_text = self.lp_recognition.rec(lp_image)
        return lp_image, lp_text