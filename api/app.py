from flask import Flask, request, jsonify
from PIL import Image

import io
import numpy as np

# from resources.face_recognition import FaceRecognition
from resources.license_plate_recognition import LPRecognition
from exceptions import InvalidUsage

global lp_recognition
lp_recognition = LPRecognition()

app = Flask(__name__)

if __name__ == "__main__":
    app.run(debug=True)
    lp_recognition = LPRecognition()

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
    lp_image, lp_text = lp_recognition.recognize(image)
    return lp_text

def get_image_from_request(request):
    if request.files.get('image'):
        image_data = request.files['image'].read()
        image = Image.open(io.BytesIO(image_data))
        return np.asarray(image)
    return None

@app.errorhandler(InvalidUsage)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response
