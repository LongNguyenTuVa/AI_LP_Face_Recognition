import os, sys, glob
# import cv2
# import logging 
# import time
# import shutil
 
import requests

base_url = 'http://127.0.0.1:5000'
test_dir = 'data/face_images'
detect = True
recognize = True

sucess_register = 0
total_register = 0
if detect:
    detect_url = base_url + "/api/face/register"
    for path in glob.glob(os.path.join(test_dir, "*")):
        user_id = path.split("\\")[-1].split("/")[-1]

        for i, image_path in enumerate(glob.glob(os.path.join(path, "*"))):
            if i != (len(glob.glob(os.path.join(path, "*"))) - 1):

                image_name = image_path.split("\\")[-1].split("/")[-1]

                detect_payload={'user_id': user_id}
                detect_files=[
                  ('image',(image_name, open(image_path,'rb'),'application/octet-stream'))
                ]
                headers = {}

                response = requests.request("POST", detect_url, headers=headers, data=detect_payload, files=detect_files)

                print(response.text)
                if "200" in response.text:
                    sucess_register += 1
                total_register += 1
    print("Sucessfully register:", sucess_register, "=", "{0:.00%}".format(sucess_register/total_register))

sucess_recognize = 0
total_recognize = 0
if recognize:
    rec_url = base_url + "/api/face/recognize"
    for path in glob.glob(os.path.join(test_dir, "*")):
        user_id = path.split("\\")[-1].split("/")[-1]

        for i, image_path in enumerate(glob.glob(os.path.join(path, "*"))):
            if i == (len(glob.glob(os.path.join(path, "*"))) - 1):

                image_name = image_path.split("\\")[-1].split("/")[-1]

                rec_payload={}
                rec_files=[
                  ('image',(image_name,open(image_path,'rb'),'application/octet-stream'))
                ]
                headers = {}

                response = requests.request("POST", rec_url, headers=headers, data=rec_payload, files=rec_files)

                print(user_id, "is predicted as", response.text.split("user_id\":\"")[-1].split("\"},\"message")[0])

                if response.text.split("user_id\":\"")[-1].split("\"},\"message")[0] == user_id:
                    sucess_recognize += 1
                total_recognize += 1

    print("Sucessfully recognize:", sucess_recognize, "=", "{0:.00%}".format(sucess_recognize/total_recognize))


