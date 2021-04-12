from flask import Flask

from resources.face_recognition import FaceRecognition
from resources.license_plate_recognition import LPRecognition

app = Flask(__name__)

if __name__ == '__main__':
    app.run(debug=True)

@app.route('/api/face/recognize')
def recognize_face():
    return 'recognize_face'

@app.route('/api/face/register')
def register_face():
    return 'register_face'

@app.route('/api/license_plate/recognize')
def recognize_lp():
    return 'recognize_lp'
