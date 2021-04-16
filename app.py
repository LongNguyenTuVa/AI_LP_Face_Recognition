import logging, logging.config, yaml
import os

from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

from api.face_recognition import FaceRecognition, db
from api.license_plate_recognition import LPRecognition
from api.utils import *
from api.exceptions import InvalidUsage

os.makedirs('logs', exist_ok=True)
logging.config.dictConfig(yaml.load(open('config/logging.conf'), Loader=yaml.FullLoader))

# app = Flask(__name__, static_url_path='/data')
app = Flask(__name__)

# app.config.from_envvar('APP_CONFIG_FILE')

app.config.update(
    DATA_DIR='data/images',
    DB_DIR='data/database',
    SQLALCHEMY_TRACK_MODIFICATIONS=False,
    FACE_THRESHOLD=0.8,
    MIN_IMAGE=3,
    UPLOAD_FOLDER='data/'
)

data_dir = app.config['DATA_DIR']
db_dir = app.config['DB_DIR']
if not os.path.exists(db_dir):
    os.makedirs(db_dir)

app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_dir}/flp.sqlite3'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'data/images'

logging.info(f'Server start with data_dir: {data_dir}, db_dir: {db_dir}')

db.init_app(app)
db.app = app
db.create_all()

global lp_recognition, face_recognition
lp_recognition = LPRecognition(data_dir)
face_recognition = FaceRecognition(data_dir, app.config['FACE_THRESHOLD'])

if __name__ == '__main__':
    app.run(debug=True)

# @app.before_request
# def log_request_info():
#     app.logger.debug('Headers: %s', request.headers)
#     # app.logger.debug('Body: %s', request.get_data())

@app.route('/')
def hello():
    return 'OK'

@app.route('/api/face/recognize', methods=['POST'])
def recognize_face():
    error = validate_request_with_image(request)
    if error:
        raise InvalidUsage(error, 400)

    try:
        image = get_image_from_request(request)
    except:
        raise InvalidUsage('read image error', 400)

    face_image_path, user_id, detection_conf, recognition_distance = face_recognition.recognize(image)
    return jsonify(
        user_id=user_id,
        detection_conf=detection_conf,
        recognition_distance=recognition_distance,
        image_path=face_image_path
    )

@app.route('/api/face/register', methods=['POST'])
def register_face():
    error = validate_request_with_image_list(request, app.config['MIN_IMAGE'])
    if error:
        raise InvalidUsage(error, 400)

    try:
        images = get_image_list_from_request(request)
    except:
        raise InvalidUsage('read image error', 400)

    user_id = request.form.get('user_id')
    user_id = face_recognition.register_face(user_id, images)

    return str(user_id)

@app.route('/api/license_plate/recognize', methods=['POST'])
def recognize_lp():
    error = validate_request_with_image(request)
    if error:
        raise InvalidUsage(error, 400)
    try:
        image = get_image_from_request(request)
    except:
        raise InvalidUsage('read image error', 400)

    lp_image_path, lp_text, detection_conf = lp_recognition.recognize(image)
    return jsonify(
        text=lp_text,
        detection_conf=detection_conf,
        recognition_conf=0,
        image_path=lp_image_path
    )

@app.errorhandler(InvalidUsage)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response