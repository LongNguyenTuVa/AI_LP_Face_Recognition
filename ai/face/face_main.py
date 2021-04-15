from face_detection.detect import *
from face_recognition.recognize import *
import matplotlib.pyplot as plt
import numpy as np

def main():
    embedding = False
    top_10 = False
    template_path = "C:/Users/LongNguyenThanh/Desktop/Python Files/Face_Recognition/FaceNet/Data/Template"
    target_path = "C:/Users/LongNguyenThanh/Desktop/Python Files/Face_Recognition/FaceNet/Data/Target"

    try:
        # Detect for template
        detect = Detect()
        if embedding == True:
            face_template, names_template, embeddings_template, _ = detect.face_detect(template_path, save=True)
        else:
            npy_file = np.load(template_path + "/" + "data.npy", allow_pickle=True)
            names_template = npy_file.item().get("Names")
            embeddings_template = torch.Tensor(npy_file.item().get("Embeddings"))
            face_template = torch.Tensor(npy_file.item().get("Faces"))
    except:
        print("Template Reading Failed")

    try:
        # Detect for Target
        print("----------------------------------------------------------------")
        face_target, names_target, embeddings_target, probability = detect.face_detect(target_path)
        print("----------------------------------------------------------------")
    except:
        print("Detection Failed")

    try:
        # Recognize + distance compare
        recognize = Recognize(0.8)
        faces, face_name, dist_10 = recognize.face_recognize(face_template, embeddings_template, embeddings_target, names_template, names_target)
    except:
        print("Embedding Failed")

    try:
        if top_10 == True:
            # Print distance + face
            for i in range(len(dist_10)):
                print("Identity Found:  " + face_name[i] + " with distance: " "{:.2f}".format(dist_10[i]))
                print(faces[i].permute(1, 2, 0).numpy().shape)
                plt.imshow(faces[i].permute(1, 2, 0).numpy())
                plt.show()
        else:
            dist = sorted(dist_10)[0]
            name = face_name[dist_10.index(dist)]
            face = faces[dist_10.index(dist)]

            print("Identity Found:  " + name + " with distance: " "{:.2f}".format(dist))
            plt.imshow((face.permute(1, 2, 0).numpy()))
            plt.show()
    except:
        print("Visualization Failed")

if __name__ == "__main__":
    main()