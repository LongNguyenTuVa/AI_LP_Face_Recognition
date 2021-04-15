from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Face(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    face_id = db.Column(db.Integer)
    embedding = db.Column(db.String)
    image_path = db.Column(db.String)
    registered_date = db.Column(db.DateTime)

class FaceRecognition:
    def __init__(self, db_dir):
        print('init face recognition')
        # self.user_manager = UserManager()
        # self.database = Database(db_dir)