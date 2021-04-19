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
    def __init__(self, data_dir, threshold, min_image):
        self.face_detection = Detect()

        self.face_dir = os.path.join(data_dir, 'faces')
        self.similarity_threshold = threshold
        self.min_image = min_image

        # create license plate folder
        os.makedirs(self.face_dir, exist_ok=True)

        # import shutil
        # shutil.rmtree(self.face_dir)

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
            detection_conf = int(round(detection_conf * 100))
            return face_image_path, matched_user[0], f'{detection_conf}%', "{:.2f}".format(matched_user[1])
        else:
            raise InvalidUsage('user not found', 400)

    def register_face(self, user_id, images):
        error_images = []

        register_new = False

        logging.info(f'register new images for user: {user_id}')
        users = []
        if not user_id:
            user_id = self.get_user_id()
            register_new = True
            logging.info(f'Register a new user')
        else:
            logging.info(f'Register new images for existed user: {user_id}')
        
        registered_date = datetime.now()

        for item in images:
            name = item[0]
            image = item[1]

            embedding_result = self.calc_embedding(image)
            if not embedding_result:
                error_images.append(name)
            else:
                (face_image_path, embedding_vector, _) = embedding_result
                users.append(User(user_id, convert_embedding_vector_to_string(embedding_vector), face_image_path, registered_date))

        error_message = ', '.join(error_images)
        if not register_new and len(users) == 0:
            raise InvalidUsage(f"registering new image for existed user with user_id: {user_id} requires at least 1 image containing user's face. Only {len(users)} images contain the user's face, please re-register image {error_message} with another image.")
            
        if register_new and len(users) < self.min_image:
            raise InvalidUsage(f"registering new user requires at least {self.min_image} images containing user's face. Only {len(users)} images contain the user's face, please re-register image {error_message} with another image.")

        self.save_user_to_database(users)
        return user_id

    def save_user_to_database(self, users):
        # # Get all users from database
        # old_users = User.query.all()

        # # Only keep 10 images
        # if len(old_users) + len(users) > 10:
        #     # Delete old user
        #     index = 10 - len(users)
        #     users.extend(old_users[index:])

        db.session.bulk_save_objects(users)
        db.session.commit()

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
        logging.info(f'save image: {image_path}')

        face_result = self.face_detection.detect(image)
        if face_result:
            (face_image, original_face_image, detection_conf) = face_result
            logging.info(f'image: {image_path} face detection confident: {detection_conf}')

            face_image_path = os.path.join(self.face_dir, suffix_name)
            logging.info(f'save face image: {face_image_path}')
            cv2.imwrite(face_image_path, convert_tensor_to_image(original_face_image))

            # Calculate embedding vector
            embedding_vector = self.face_detection.calc_embedding(face_image)
            return (face_image_path, embedding_vector[0], detection_conf)
        else:
            logging.info(f'image: {image_path} can not detect face')
            return None

    def cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        
