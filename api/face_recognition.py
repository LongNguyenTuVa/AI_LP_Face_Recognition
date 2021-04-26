import os
import cv2
import logging
import time
import numpy as np

from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

from ai.face.face_detection.detect import Detect
from api.utils import *
from api.exceptions import InvalidUsage

db = SQLAlchemy()

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String)
    embedding_str = db.Column(db.String)
    face_image_path = db.Column(db.String)
    registered_date = db.Column(db.DateTime)

    def __init__(self, user_id, embedding_str, face_image_path, registered_date):
        self.user_id = user_id
        self.embedding_str = embedding_str
        self.face_image_path = face_image_path
        self.registered_date = registered_date

    def __repr__(self):
        return f'{self.user_id}, {self.registered_date}'

class FaceRecognition:
    def __init__(self, data_dir, threshold):
        self.face_detection = Detect()

        self.face_dir = os.path.join(data_dir, 'faces')
        self.similarity_threshold = threshold

        # create license plate folder
        os.makedirs(self.face_dir, exist_ok=True)

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
            if recognition_distance >= self.similarity_threshold:
                users.append((user_id, recognition_distance))

        if len(users) != 0: 
            matched_user = max(users, key=lambda item:item[1])
            recognition_distance = int(round(matched_user[1] * 100) )
            detection_conf = int(round(detection_conf * 100))
            return face_image_path, matched_user[0], str(detection_conf), str(recognition_distance)
        else:
            raise InvalidUsage('user not found', 400)

    def register_face(self, user_id, image):
        logging.info(f'Register a new image for user_id: {user_id}')

        registered_date = datetime.now()
        embedding_result = self.calc_embedding(image)
        if not embedding_result:
            raise InvalidUsage('can not detect face from image', 400)
        (face_image_path, embedding_vector, detection_conf) = embedding_result

        user = User(user_id, convert_embedding_vector_to_string(embedding_vector), face_image_path, registered_date)
        
        self.save_user_to_database(user_id, user)

        return user_id

    def save_user_to_database(self, user_id, user):
        current_users = User.query.filter(User.user_id==user_id).order_by(User.registered_date).all()

        if len(current_users) >= 10:
            oldest_user = current_users[0]
            db.session.delete(oldest_user)
      
        db.session.add(user)
        db.session.commit()

    def get_embedding_vectors(self):
        embedding_vectors = []
        for user in User.query.all():
            embedding_vector = convert_embedding_string_to_vector(user.embedding_str)
            embedding_vectors.append((user.user_id, embedding_vector))
        return embedding_vectors
                
    def calc_embedding(self, image):
        image_name, suffix_name = generate_image_file_name('face')
        image_path = os.path.join(self.face_dir, image_name)
        cv2.imwrite(image_path, image)
        logging.info(f'save image: {image_path}')

        start = time.time()
        face_result = self.face_detection.detect(image)
        logging.info(f'face detection: {time.time() - start}s')

        if face_result:
            (face_image, original_face_image, detection_conf) = face_result
            logging.info(f'image: {image_path} face detection confident: {detection_conf}')

            face_image_path = os.path.join(self.face_dir, suffix_name)
            logging.info(f'save face image: {face_image_path}')
            cv2.imwrite(face_image_path, original_face_image)

            # Calculate embedding vector
            start = time.time()
            embedding_vector = self.face_detection.calc_embedding(face_image)
            logging.info(f'calculate embedding vector: {time.time() - start}s')
            
            return (face_image_path, embedding_vector[0], detection_conf)
        else:
            logging.info(f'image: {image_path} can not detect face')
            return None

    def cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))