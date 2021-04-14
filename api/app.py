import logging, logging.config, yaml
import os

from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

from resources.face_recognition import FaceRecognition
from resources.license_plate_recognition import LPRecognition
from utils import get_image_from_request
from exceptions import InvalidUsage

os.makedirs('logs', exist_ok=True)
logging.config.dictConfig(yaml.load(open('api/config/logging.conf'), Loader=yaml.FullLoader))

app = Flask(__name__)

db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String, unique=True, nullable=False)
    email = db.Column(db.String, unique=True, nullable=False)

db.session.add(User(name="Flask", email="example@example.com"))
db.session.commit()

users = User.query.all()
print(users)
app.logger.info(users)

app.config.from_envvar('APP_CONFIG_FILE')
data_dir = app.config['DATA_DIR']
app.logger.info(f'Server start with data_dir: {data_dir}, db_dir: {app.config["SQLALCHEMY_DATABASE_URI"]}')

global lp_recognition
lp_recognition = LPRecognition(data_dir)

if __name__ == "__main__":
    app.run(host="0.0.0.0")

@app.route('/api/face/recognize', methods=['POST'])
def recognize_face():
    return 'recognize_face'

@app.route('/api/face/register', methods=['POST'])
def register_face():
    return 'register_face'

@app.route('/api/license_plate/recognize', methods=['POST'])
def recognize_lp():
    # raise InvalidUsage('Not implemented', status_code=404)
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

@app.errorhandler(InvalidUsage)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response