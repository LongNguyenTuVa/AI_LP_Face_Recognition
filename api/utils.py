from PIL import Image
from datetime import datetime

import string
import random
import io
import cv2
import numpy as np

from PIL import Image
from datetime import datetime

IMAGE_FILE_EXT = ('.png', '.jpg', '.jpeg')

def convert_embedding_vector_to_string(embedding_vector):
    return ' '.join([str(item) for item in embedding_vector])

def convert_embedding_string_to_vector(embedding_str):
    return np.fromstring(embedding_str, dtype=float, sep=' ')

def convert_tensor_to_image(tensor_image):
    image = tensor_image.permute(1, 2, 0).numpy() * 255
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def get_image_from_request(request):
    if request.files.get('image'):
        image_data = request.files['image'].read()
        image = Image.open(io.BytesIO(image_data))
        return np.array(image)
    return None

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

def generate_image_file_name(suffix):
    time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    id_str = id_generator()
    return f'{time_str}_{id_str}.jpg', f'{time_str}_{id_str}_{suffix}.jpg'
    
def validate_request_with_image(request):
    if not request.files.get('image'):
        return '[image] field can not empty'
    
    filename = request.files['image'].filename
    if not filename.lower().endswith(IMAGE_FILE_EXT):
        return f'invalid image file: {filename}, only accept {IMAGE_FILE_EXT} file'

    return None
