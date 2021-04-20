import logging, logging.config, yaml
import os
import markdown
import time

from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

from api.face_recognition import FaceRecognition, db
from api.license_plate_recognition import LPRecognition
from api.utils import *
from api.exceptions import InvalidUsage

os.makedirs('logs', exist_ok=True)
logging.config.dictConfig(yaml.load(open('config/logging.conf'), Loader=yaml.FullLoader))

app = Flask(__name__)

import tensorflow as tf
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#   tf.config.experimental.set_memory_growth(gpu, True)
# Assume that you have 12GB of GPU memory and want to allocate ~4GB:
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*4)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

app.config.update(
    DATA_DIR='static/images',
    DB_DIR='data/database',
    SQLALCHEMY_TRACK_MODIFICATIONS=False,
    FACE_SIMILARITY_THRESHOLD=0.7,
    MIN_IMAGE=3,
    UPLOAD_FOLDER='static/'
)

data_dir = app.config['DATA_DIR']
db_dir = app.config['DB_DIR']
if not os.path.exists(db_dir):
    os.makedirs(db_dir)

app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_dir}/flp.sqlite3'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'static/'

logging.info(f'Server start with data_dir: {data_dir}, db_dir: {db_dir}')

db.init_app(app)
db.app = app
db.create_all()

global lp_recognition, face_recognition
lp_recognition = LPRecognition(data_dir)
face_recognition = FaceRecognition(data_dir, app.config['FACE_SIMILARITY_THRESHOLD'], app.config['MIN_IMAGE'])

if __name__ == '__main__':
    app.run()

@app.route('/api/docs')
def api_doc():
    return app.send_static_file('api_doc.html')

@app.route('/api/face/recognize', methods=['POST'])
def recognize_face():
    start = time.time()
    error = validate_request_with_image(request)
    if error:
        raise InvalidUsage(error, 400)

    try:
        image = get_image_from_request(request)
    except:
        raise InvalidUsage('read image error', 400)

    face_image_path, user_id, detection_conf, recognition_distance = face_recognition.recognize(image)
    end = time.time()
    logging.info(f'request processed with time: {end - start}')
    return jsonify(
        user_id=user_id,
        detection_conf=detection_conf,
        recognition_distance=recognition_distance,
        image_path=face_image_path
    )

@app.route('/api/face/register', methods=['POST'])
def register_face():
    start = time.time()
    user_id = request.form.get('user_id')

    error = validate_request_with_image(request)
    if error:
        raise InvalidUsage(error, 400)

    try:
        image = get_image_from_request(request)
    except:
        raise InvalidUsage('read image error', 400)

    user_id = face_recognition.register_face(user_id, image)

    end = time.time()
    logging.info(f'request processed with time: {end - start}')

    return jsonify(
        user_id=str(user_id)
    )

@app.route('/api/license_plate/recognize', methods=['POST'])
def recognize_lp():
    start = time.time()
    error = validate_request_with_image(request)
    if error:
        raise InvalidUsage(error, 400)
    try:
        image = get_image_from_request(request)
    except:
        raise InvalidUsage('read image error', 400)

    lp_image_path, lp_text, detection_conf, recognition_conf = lp_recognition.recognize(image)
    end = time.time()
    logging.info(f'request processed with time: {end - start}')
    return jsonify(
        text=lp_text,
        detection_conf=detection_conf,
        recognition_conf=recognition_conf,
        image_path=lp_image_path
    )

@app.errorhandler(InvalidUsage)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response
