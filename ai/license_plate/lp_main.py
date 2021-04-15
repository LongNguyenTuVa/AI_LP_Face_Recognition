from lp_detection.detect import *
from lp_recognition.recognize import *
from car_detection.detect import *
import matplotlib.pyplot as plt
import numpy as np

def main():
    
    data_path = "data97"
    # Detect for template
    lp_detect = LP_Detect()
    car_detect = CarDetection()
    recognize = LP_Recognize()

    correct = 0
    total = 0
    ground_truth = []
    predict = []
    for p in os.listdir(data_path):
        true = p.split(".")[0].split(" ")[0].split("_")[-1]
        original_image = cv2.imread(data_path + "/" + p)
        try:
            image = car_detect.car_detect(image_path = data_path + "/" + p)
        except Exception as e:
            print("Failed Car Detection")
            continue
        try:
            plate_image = lp_detect.detect(image = image, classify=True)
            #print(coord)
            print("--------------True", true, "--------------")
            cv2.imwrite("output" + "/" + p, plate_image)
            plate_type = lp_detect.get_plate_type()
            # if plate_image.shape[0] == 200:
            #     plate_type = 2
            # else:
            #     plate_type = 1
        except Exception as e:
            print("Failed LP Detection")
            continue
        try:
            text = recognize.rec(plate_image, mode = plate_type)
            # true = p.split(".")[0].split(" ")[0]
            while len(true) < 8:
                true += "*"
            ground_truth.append(true)
            predict.append(text)
        except Exception as e:
            print("Failed Recognition")
            continue
        try:
            text2 = text.replace("*", "")
            text2 = text2[:3] + '-' + text2[3:]
            if len(text2) == 9:
                text2 = text2[:7] + '.' + text2[7:]
            (label_width, label_height), baseline = cv2.getTextSize(text2, cv2.FONT_HERSHEY_PLAIN, 1.5, 3)
            original_image = cv2.rectangle(original_image, (int(original_image.shape[1]/2 - 120), original_image.shape[0] - 40), (int(original_image.shape[1]/2 - 120)+label_width*2+10, original_image.shape[0]), (255,255,255), -1)
            original_image = cv2.putText(original_image, text2, (int(original_image.shape[1]/2 - 110), original_image.shape[0]-2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA)
            cv2.imwrite("output" + "/car_" + p, original_image)
            # print("Detect %i plate(s) in"%len(plate_image), p)
            print("--------------Pred", text2, "--------------")
            print("------------------------------------------")
            total+=1
            if true == text:
                correct += 1
            # # print("Coordinate of plate(s) in image: \n", coordinate)

            # # Visualize our result
            # plt.figure(figsize=(12,5))
            # plt.subplot(1,2,1)
            # plt.axis(False)
            # plt.imshow(preprocess_image(cv2.imread(data_path + "/" + p)))
            # plt.subplot(1,2,2)
            # # plt.axis(False)
            # plt.imshow(plate_image)
            # plt.show()
        except Exception as e:
            print("Failed Visualization")
            continue
    with open("true.txt", "w") as f:
        for item in ground_truth:
            f.write("%s\n" % item)
    with open("pred.txt", "w") as f:
        for item in predict:
            f.write("%s\n" % item)
    print("TOTAL ACCURACY: ", str(int(correct / total*100)) + "%")

if __name__ == "__main__":
    main()