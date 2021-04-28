import os, glob, shutil, time
import cv2
import imutils

from api.license_plate_recognition import LPRecognition

DATA_DIR='static/images'
lp_recognition = LPRecognition(DATA_DIR)

# test_dir = 'test_data/license_plates/20210423_Royal_TestData'
test_dir = 'test_data/license_plates/errors'
# test_dir = 'test_data/license_plates/OneDrive_1_4-27-2021/Data11'
# test_dir = 'test_data/license_plates/from_video'

test_output_dir = test_dir + '_Ouput'
test_error_dir = test_output_dir + '/error'
test_success_dir = test_output_dir + '/success'

if os.path.exists(test_output_dir):
    shutil.rmtree(test_output_dir)

os.makedirs(test_output_dir, exist_ok=True)
os.makedirs(test_error_dir, exist_ok=True)
os.makedirs(test_success_dir, exist_ok=True)

total_img = 0
success_img = 0
    
for image_path in glob.glob(test_dir + '/*.*'):
    total_img = total_img + 1
    image_name = os.path.split(image_path)[-1].replace('.jpg', '').replace('.JPG', '')

    image = cv2.imread(image_path)

    lp_text = ''

    try:
        start = time.time()

        result = lp_recognition.recognize(image)

        end = time.time()

        lp_image_path = result['image_path']
        lp_text = result['text']
        detection_conf = result['detection_conf']
        recognition_conf = result['recognition_conf']

        lp_image_name = image_name + '_' + lp_text + '.jpg'
        # shutil.copyfile(lp_image_path, os.path.join(test_output_dir, lp_image_name))

        pre_text = lp_text.replace('-', '').replace('.', '')
        gt_text = image_name.split('_')[-1]

        print('detection_conf', detection_conf, 'recognition_conf', recognition_conf, 'time: ', (time.time() - start))

        if pre_text.lower() == gt_text.lower():
            print(image_path, 'predict success: ', gt_text, pre_text)
            success_img = success_img + 1
            shutil.copyfile(lp_image_path, os.path.join(test_success_dir, lp_image_name))
        else:
            print(image_path, 'predict error: ', gt_text, pre_text)
            shutil.copyfile(lp_image_path, os.path.join(test_error_dir, lp_image_name))

    except Exception as e:
        print('cannot detect lp from image', image_path, e)
        shutil.copyfile(image_path, os.path.join(test_error_dir, image_name + '.jpg'))

    if image.shape[1] > image.shape[0]:
        image_debug = imutils.resize(image, width=1000)
    else:
        image_debug = imutils.resize(image, height=1000)

    image_h, image_w, _ = image_debug.shape
    center_x = int(image_w / 2)
    box_w = 200
    box_h = 30

    image_debug = cv2.rectangle(image_debug, (center_x - int(box_w/2), (image_h - box_h)), (center_x + int(box_w/2), image_h), (0, 0, 255), -1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    textsize = cv2.getTextSize(lp_text, font, 0.9, 2)[0]

    textX =int((center_x - textsize[0] / 2))
    textY = int((image_h - box_h + textsize[1] + 5))
    cv2.putText(image_debug, lp_text, (textX, textY), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    image_debug_path = os.path.join(test_output_dir, image_name + '.jpg')
    cv2.imwrite(image_debug_path, image_debug)

print('predict', success_img, '/', total_img)