from lp_detection.detect import *
from lp_recognition.recognize import *
import matplotlib.pyplot as plt
import numpy as np

def main():
    data_path = "data"
    # Detect for template
    detect = LP_Detect()
    recognize = LP_Recognize()

    for p in os.listdir(data_path):
        try:
            plate_image = detect.detect(image_path = data_path + "/" + p)
        except Exception as e:
            print(e)
            print("Failed Detection")
            continue
        try:
            text = recognize.rec(plate_image)
        except Exception as e:
            print(e)
            print("Failed Recognition")
            continue
        try:
            print("Detect %i plate(s) in"%len(plate_image), p)
            print(text)
            # print("Coordinate of plate(s) in image: \n", coordinate)

            # Visualize our result
            plt.figure(figsize=(12,5))
            plt.subplot(1,2,1)
            plt.axis(False)
            plt.imshow(preprocess_image(data_path + "/" + p))
            plt.subplot(1,2,2)
            # plt.axis(False)
            plt.imshow(plate_image[0])
            # cv2.imwrite("output/Wild/" + str(count).zfill(4) + ".jpg", plate_image[0]*255)
            plt.show()
        except Exception as e:
            print(e)
            print("Failed Visualization")
            continue

if __name__ == "__main__":
    main()