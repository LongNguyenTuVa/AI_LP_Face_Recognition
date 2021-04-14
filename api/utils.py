from PIL import Image
from datetime import datetime

import string
import random
import io
import numpy as np

def get_image_from_request(request):
    if request.files.get('image'):
        image_data = request.files['image'].read()
        image = Image.open(io.BytesIO(image_data))
        return np.asarray(image)
    return None

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

def generate_image_file_name():
    time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f'{time_str}_{id_generator()}.jpg'