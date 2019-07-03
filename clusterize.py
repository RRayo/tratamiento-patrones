import gensim
import numpy as np
import sklearn.preprocessing
import sklearn.manifold
import sklearn.cluster
import pandas as pd
import matplotlib.pyplot as plt

#defining script constants

word2vec_model_location = r"data/input/GoogleNews-vectors-negative300.bin" #any word2vec binary can be used
input_location = r'data/output/deteccion_preprocessed.txt' #file containing each line as an image and comma-separated objects
output_location = r'data/output/images_labeled.csv' #output clusterized file
cluster_plot_location = r'data/output/cluster.pdf' #2D plot of the clusters
min_objects = 3 #minimum number of unique objects per image
clusterization_algorithm = sklearn.cluster.OPTICS(min_samples=50) #any sklearn clusterization Class

print('loading: ' + word2vec_model_location)

#loading the word2vec model
news = gensim.models.KeyedVectors.load_word2vec_format(
    word2vec_model_location, binary=True
    ) 

#function to transform a list of objects into a unique vector, each vector of each object is added and the final vector is normalized
def to_vector(words,model):
    vec = np.zeros(300)
    for word in words:
        if (word in model):
            vec += model[word]
    return vec / np.linalg.norm(vec)

#saving each image along with their objects into lists

images = []
objects = []
with open(input_location) as file:
    for line in file:
        line = line.rstrip().split('!#!')
        image = line[0]
        detected = line[1].replace(' ', '_').split(',')
        if len(set(detected)) > min_objects:
            images.append(image)
            objects.append(detected)

print('using %s images' % str(len(images)))

#encoding each name of images and objects to then insert them into a numpy matrix

label_encoder_images = sklearn.preprocessing.LabelEncoder()
images_encoded = label_encoder_images.fit_transform(images)
images_encoded = np.reshape(images_encoded, (-1, 1))

label_encoder_objects = sklearn.preprocessing.LabelEncoder()
objects_encoded = label_encoder_objects.fit_transform([str(current_objects) for current_objects in objects])
objects_encoded = np.reshape(objects_encoded, (-1, 1))

#creating an empty matrix to store the vector of each image

objects_matrix = np.zeros((len(objects), len(news.wv['hello'])))

#populating the matrix

for i,current_objects in enumerate(objects):
    vector = to_vector(current_objects,news)
    objects_matrix[i,] = vector

data_matrix = np.concatenate([objects_matrix,images_encoded,objects_encoded], axis=1)

#dropping NAs

data_matrix_without_nan = data_matrix[~np.isnan(data_matrix).any(axis=1)]

print('reducing dimensionality')

#using t-SNE to reduce the dimensionality of the data matrix

tisni = sklearn.manifold.TSNE(verbose=True)
tisni.fit(data_matrix_without_nan[:,:300])

print('clusterizing')

#fitting the clustering model

clusterization_algorithm.fit(tisni.embedding_)

print('%s clusters found' % str(np.unique(clusterization_algorithm.labels_).shape[0]))

#creating a dataframe to save the labels of each image

labeled_images = np.concatenate([data_matrix_without_nan[:,300:],
                np.reshape(clusterization_algorithm.labels_, (-1, 1)),
               ],axis=1).astype(int)
labeled_images = pd.DataFrame(labeled_images, columns = ['image','objects','label'])
labeled_images['image'] = label_encoder_images.inverse_transform(labeled_images.image)
labeled_images['objects'] = label_encoder_objects.inverse_transform(labeled_images.objects)

#saving the dataframe into a csv

labeled_images[['image','label','objects']].sort_values(by='label').to_csv(output_location, index=False)

#saving the clusterized plot

plt.scatter(
    tisni.embedding_[:,0],
    tisni.embedding_[:,1],
    c = clusterization_algorithm.labels_,
    s = 3
           )
plt.savefig(cluster_plot_location)