import os
import cv2
import logging
import numpy as np

from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

from ai.face.face_detection.detect import Detect
from api.utils import *
from api.exceptions import InvalidUsage

db = SQLAlchemy()

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer)
    embedding_str = db.Column(db.String)
    face_image_path = db.Column(db.String)
    registered_date = db.Column(db.DateTime)

    def __init__(self, user_id, embedding_str, face_image_path, registered_date):
        self.user_id = user_id
        self.embedding_str = embedding_str
        self.face_image_path = face_image_path
        self.registered_date = registered_date

class FaceRecognition:
    def __init__(self, data_dir, threshold):
        self.face_detection = Detect()

        self.face_dir = os.path.join(data_dir, 'faces')
        self.threshold = threshold

        # create license plate folder
        os.makedirs(self.face_dir, exist_ok=True)

        import shutil
        shutil.rmtree(self.face_dir)

    def recognize(self, image):
        embedding_vectors = self.get_embedding_vectors()
        embedding_result = self.calc_embedding(image)
        if not embedding_result:
            raise InvalidUsage('can not detect face from image', 400)
        (face_image_path, embedding_vector, detection_conf) = embedding_result
        logging.info(f'detect face with face_image_path: {face_image_path}, detection_conf: {detection_conf}')

        users = []
        for item in embedding_vectors:
            user_id = item[0]
            e = item[1]
            recognition_distance = self.cosine_similarity(embedding_vector, e)
            if recognition_distance >= self.threshold:
                users.append((user_id, recognition_distance))

        if len(users) != 0: 
            matched_user = max(users, key=lambda item:item[1])
            detection_conf = int(round(detection_conf * 100))
            return face_image_path, matched_user[0], f'{detection_conf}%', "{:.2f}".format(recognition_distance)
        return None, None, None, None

    def register_face(self, user_id, images):
        users = []
        if not user_id:
            user_id = self.get_user_id()
            logging.info(f'Register a new user')
        else:
            logging.info(f'Register new images for user: {user_id}')
        
        registered_date = datetime.now()

        for item in images:
            name = item[0]
            image = item[1]
            face_image_path, embedding_vector, detection_conf = self.calc_embedding(image)
            if face_image_path is not None and embedding_vector is not None and detection_conf >= 0.7:
                users.append(User(user_id, convert_embedding_vector_to_string(embedding_vector), face_image_path, registered_date))

        db.session.bulk_save_objects(users)
        db.session.commit()

        return user_id

    def get_user_id(self):
        user_ids = [user.user_id for user in User.query.all()]
        return max(user_ids) + 1 if len(user_ids) != 0 else 1

    def get_embedding_vectors(self):
        embedding_vectors = []
        for user in User.query.all():
            embedding_vector = convert_embedding_string_to_vector(user.embedding_str)
            embedding_vectors.append((user.user_id, embedding_vector))
        return embedding_vectors
                
    def calc_embedding(self, image):
        image_name, suffix_name = generate_image_file_name('face')
        image_path = os.path.join(self.face_dir, image_name)
        cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        face_image, detection_conf = self.face_detection.detect(image)

        if face_image is not None and detection_conf > 0.8:
            face_image_path = os.path.join(self.face_dir, suffix_name)
            cv2.imwrite(face_image_path, convert_tensor_to_image(face_image))

            # Calculate embedding vector
            embedding_vector = self.face_detection.calc_embedding(face_image)
            return (face_image_path, embedding_vector[0], detection_conf)
        return None

    def cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        
