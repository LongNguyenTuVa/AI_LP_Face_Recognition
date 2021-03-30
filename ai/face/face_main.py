from face_detection.detect import *
from face_recognition.recognize import *
import matplotlib.pyplot as plt
import numpy as np

def main():
    embedding = False
    template_path = "C:/Users/LongNguyenThanh/Desktop/Python Files/Face_Recognition/FaceNet/Data/Template"
    target_path = "C:/Users/LongNguyenThanh/Desktop/Python Files/Face_Recognition/FaceNet/Data/Target"

    # Detect for template
    detect = Detect()
    if embedding == True:
        face_template, names_template, embeddings_template = detect.face_detect(template_path, save=True)
    else:
        npy_file = np.load(template_path + "/" + "data.npy", allow_pickle=True)
        names_template = npy_file.item().get("Names")
        embeddings_template = torch.Tensor(npy_file.item().get("Embeddings"))
        face_template = torch.Tensor(npy_file.item().get("Faces"))

    # Detect for Target
    print("----------------------------------------------------------------")
    face_target, names_target, embeddings_target = detect.face_detect(target_path)
    print("----------------------------------------------------------------")

    # Recognize + distance compare
    recognize = Recognize()
    faces, face_name, dist_10 = recognize.face_recognize(face_template, embeddings_template, embeddings_target, names_template, names_target)
    
    # Print distance + face
    for i in range(len(dist_10)):
        print(face_name[i], "{:.2f}".format(dist_10[i]))
        plt.imshow(faces[i].permute(1, 2, 0).numpy())
        plt.show()

if __name__ == "__main__":
    main()