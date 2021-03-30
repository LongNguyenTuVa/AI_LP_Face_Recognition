class Recognize:
    def __init__(self, TRHESHOLD = 1.1):
        self.TRHESHOLD = TRHESHOLD

    def face_recognize(self, face_template, embeddings_template, embeddings_target, names_template, names_target):
        # Calculate distance of all template vs target
        dists = [[(e1 - e2).norm().item() for e2 in embeddings_template] for e1 in embeddings_target]
        
        distance = []
        faces = []
        face_name = []

        # Keep only the ones with small distance + top 10
        for dist in dists:
            if min(dist) < self.TRHESHOLD:
                min_distance = sorted(dist)
                dist_10 = min_distance[:10]
                for i in range(len(dist_10)):
                    if dist_10[i] < self.TRHESHOLD:
                        distance.append(dist_10[i])
                        face_name.append(names_template[dist.index(dist_10[i])])
                        faces.append(face_template[dist.index(dist_10[i])])
                        
        return faces, face_name, distance