from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import os, glob, logging

workers = 0 if os.name == 'nt' else 4
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def collate_fn(x):
    return x[0]
def max_area(box):
    return (box[2]-box[0]) * (box[3]-box[1])

def save_npy(names, embeddings, face_alligned, path):
    data = {"Names": names,
            "Embeddings": embeddings.tolist(),
            "Faces": face_alligned.tolist()}
    np.save(path + "/" + 'data.npy', data)

class Detect:
    def __init__(self):
        logging.info('PyTorch - Load face detection model')
        self.detection_model = MTCNN(
                image_size=160, margin=0, min_face_size=20, keep_all=False,
                thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
                select_largest=True, device=device
            )
        logging.info('PyTorch - Load face embedding model')
        self.embedding_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    '''
    Detect face from image
    '''
    def detect(self, image):
        image_alligned, prob =  self.detection_model(image, return_prob=True)

        if image_alligned is not None:
            boxes, probs = self.detection_model.detect(image)
            boxes = boxes.squeeze()

            # Draw boxes and save faces
            orginal_image = np.asarray(image)
            if type(boxes[0]) is np.ndarray:
                box_sort = sorted(boxes, key=max_area)
                boxes = box_sort[-1]
            face_image = orginal_image[int(boxes[1]):int(boxes[3]),int(boxes[0]):int(boxes[2])]
            return (image_alligned, face_image, prob)
            
        return None

    def calc_embedding(self, face_image):
        return self.embedding_model(face_image.unsqueeze(0)).to(device).detach().numpy()

    def face_detect(self, path, save=False):
        # Load Folder as Dataset
        dataset = datasets.ImageFolder(path)
        dataset.idx_to_class = {number:name for name, number in dataset.class_to_idx.items()}
        loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)

        # Detect and Allign Faces
        face_alligned = []
        names = []
        probability_list = []
        face_original = []
        for image, number in loader:
            image_alligned, probability = self.detection_model(image, return_prob=True)

            if image_alligned is not None and probability > 0.9:
                boxes, probs = self.detection_model.detect(image)
                boxes = boxes.squeeze()
                # Draw boxes and save faces
                orginal_image = np.asarray(image)
                if type(boxes[0]) is np.ndarray:
                    box_sort = sorted(boxes, key=max_area)
                    boxes = box_sort[0]      
                face_image = orginal_image[boxes[1]:boxes[3],boxes[0]:boxes[2]]

                print(dataset.idx_to_class[number] + ' - Detection Probability: {0:.1%}'.format(probability))
                face_alligned.append(image_alligned)
                names.append(dataset.idx_to_class[number])
                probability_list.append(probability)
                face_original.append(face_image)
            else:
                print(dataset.idx_to_class[number] + " - Fail face detection")

        # Embedding Faces
        face_alligned = torch.stack(face_alligned).to(device)
        embeddings = self.embedding_model(face_alligned).detach().cpu()
        # Save the embeddings
        if save == True:
            save_npy(names, embeddings, face_alligned, path)


        return face_alligned, names, embeddings, probability_list, face_original

    
        
