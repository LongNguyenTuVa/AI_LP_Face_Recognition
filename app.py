import logging, logging.config, yaml
import os

from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

from api.face_recognition import FaceRecognition, db
from api.license_plate_recognition import LPRecognition
from api.utils import get_image_from_request
from api.exceptions import InvalidUsage

os.makedirs('logs', exist_ok=True)
logging.config.dictConfig(yaml.load(open('config/logging.conf'), Loader=yaml.FullLoader))

app = Flask(__name__)

app.config.from_envvar('APP_CONFIG_FILE')
data_dir = app.config['DATA_DIR']
db_dir = app.config['DB_DIR']
if not os.path.exists(db_dir):
    os.makedirs(db_dir)
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_dir}/flp.sqlite3'

app.logger.info(f'Server start with data_dir: {data_dir}, db_dir: {db_dir}')

db.init_app(app)
db.app = app
db.create_all()

global lp_recognition, face_recognition
lp_recognition = LPRecognition(data_dir)
face_recognition = FaceRecognition(db_dir)

if __name__ == '__main__':
    app.run(debug=True)

@app.route('/api/face/recognize', methods=['POST'])
def recognize_face():
    return 'recognize_face'

@app.route('/api/face/register', methods=['POST'])
def register_face():
    return 'register_face'

@app.route('/api/license_plate/recognize', methods=['POST'])
def recognize_lp():
    image = get_image_from_request(request)
    if image is None:
        raise InvalidUsage('Invalid Request', status_code=400)
    lp_image_path, lp_text = lp_recognition.recognize(image)
    return jsonify(
        text=lp_text,
        detection_conf=0,
        recognition_conf=0,
        image_path=lp_image_path
    )
    return "OK"

@app.errorhandler(InvalidUsage)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response