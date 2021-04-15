import string
import random
import io
import cv2
import numpy as np

from PIL import Image
from datetime import datetime

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

def get_image_list_from_request(request):
    images = []
    if request.files.get('images'):
        image_list = request.files.getlist('images')
        for template in image_list:
            image_data = template.read()
            image = Image.open(io.BytesIO(image_data))
            images.append((template.filename, np.array(image)))
    return images

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

def generate_image_file_name(suffix):
    time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    id_str = id_generator()
    return f'{time_str}_{id_str}.jpg', f'{time_str}_{id_str}_{suffix}.jpg', 