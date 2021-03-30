from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import pandas as pd
import os, glob

workers = 0 if os.name == 'nt' else 4
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def collate_fn(x):
    return x[0]

def save_npy(names, embeddings, face_alligned, path):
    data = {"Names": names,
            "Embeddings": embeddings.tolist(),
            "Faces": face_alligned.tolist()}
    np.save(path + "/" + 'data.npy', data)


class Detect:
    def __init__(self):
        self.detection_model = MTCNN(
                image_size=160, margin=0, min_face_size=20, keep_all=False,
                thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
                device=device
            )
        self.embedding_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    def face_detect(self, path, save=False):
        # Load Folder as Dataset
        dataset = datasets.ImageFolder(path)
        dataset.idx_to_class = {number:name for name, number in dataset.class_to_idx.items()}
        loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)

        # Detect and Allign Faces
        face_alligned = []
        names = []
        for image, number in loader:
            image_alligned, probability = self.detection_model(image, return_prob=True)
            if image_alligned is not None and probability > 0.9:
                print(dataset.idx_to_class[number] + ' - Detection Probability: {0:.1%}'.format(probability))
                face_alligned.append(image_alligned)
                names.append(dataset.idx_to_class[number])
            else:
                print(dataset.idx_to_class[number] + " - Fail face detection")

        # Embedding Faces
        face_alligned = torch.stack(face_alligned).to(device)
        embeddings = self.embedding_model(face_alligned).detach().cpu()
        # Save the embeddings
        if save == True:
            save_npy(names, embeddings, face_alligned, path)


        return face_alligned, names, embeddings

    
        