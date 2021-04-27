import os, glob, shutil, time
import cv2

from api.license_plate_recognition import LPRecognition

DATA_DIR='static/images'
lp_recognition = LPRecognition(DATA_DIR)

test_dir = 'test_data/license_plates/20210412_LP_TestData'
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
for image_path in glob.glob(test_dir + '/*.jpg'):
    total_img = total_img + 1
    image_name = os.path.split(image_path)[-1].replace('.jpg', '')

    image = cv2.imread(image_path)

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

print('predict', success_img, '/', total_img)