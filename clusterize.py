import gensim
import numpy as np
import sklearn.preprocessing
import sklearn.manifold
import sklearn.cluster
import pandas as pd

word2vec_model_location = r"data/input/GoogleNews-vectors-negative300.bin"
input_location = r'data/output/deteccion_preprocessed.txt'
output_location = r'data/output/images_labeled.csv'
clusterization_algorithm = sklearn.cluster.DBSCAN(eps=3)

news = gensim.models.KeyedVectors.load_word2vec_format(
    word2vec_model_location, binary=True
    ) 

def to_vector(words,model):
    vec = np.zeros(300)
    for word in words:
        if (word in model):
            vec += model[word]
    return vec / np.linalg.norm(vec)

images = []
objects = []
with open(input_location) as file:
    for line in file:
        line = line.rstrip().split('!#!')
        image = line[0]
        detected = line[1].replace(' ', '_').split(',')
        if len(set(detected)) > 3:
            images.append(image)
            objects.append(detected)

label_encoder_images = sklearn.preprocessing.LabelEncoder()
images_encoded = label_encoder_images.fit_transform(images)
images_encoded = np.reshape(images_encoded, (-1, 1))

label_encoder_objects = sklearn.preprocessing.LabelEncoder()
objects_encoded = label_encoder_objects.fit_transform([str(current_objects) for current_objects in objects])
objects_encoded = np.reshape(objects_encoded, (-1, 1))

objects_matrix = np.zeros((len(objects), len(news.wv['hello'])))

for i,current_objects in enumerate(objects):
    vector = to_vector(current_objects,news)
    objects_matrix[i,] = vector

data_matrix = np.concatenate([objects_matrix,images_encoded,objects_encoded], axis=1)

data_matrix_without_nan = data_matrix[~np.isnan(data_matrix).any(axis=1)]

# tisni = sklearn.manifold.TSNE(verbose=True)

# tisni.fit(data_matrix_without_nan[:,:300])

model = clusterization_algorithm
model.fit(data_matrix_without_nan[:,:300])

labeled_images = np.concatenate([data_matrix_without_nan[:,300:],
                np.reshape(model.labels_, (-1, 1)),
               ],axis=1).astype(int)
labeled_images = pd.DataFrame(labeled_images, columns = ['image','objects','label'])
labeled_images['image'] = label_encoder_images.inverse_transform(labeled_images.image)
labeled_images['objects'] = label_encoder_objects.inverse_transform(labeled_images.objects)

labeled_images.sort_values(by='label').to_csv(output_location)