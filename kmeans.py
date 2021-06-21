import cv2
import os
import numpy as np

from sklearn.cluster import KMeans

# from keras.models import Model
# from keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input


def get_files(path_to_files_, size):
    fn_images = []
    for file in os.listdir(path_to_files_):

        with open(os.path.join(path_to_files_, file), 'rb') as f:
            check_chars = f.read()[-2:]
        if check_chars != b'\xff\xd9':
            print('Not complete image')
        else:

            img = cv2.imread(os.path.join(path_to_files_, file))
            img = cv2.resize(img, size)
            fn_images.append([file, img])

    return dict(fn_images)


def get_model(layer='fc2'):
    base_model = VGG16(weights='imagenet', include_top=True)
    model_ = Model(inputs=base_model.input,
                  outputs=base_model.get_layer(layer).output)
    return model_


def feature_vector(img_arr,model_):

 img_arr = np.expand_dims(img_arr, axis=0)
 arr4d_pp = preprocess_input(img_arr)
 return model_.predict(arr4d_pp)[0, :]


def feature_vectors(img_dir, model_):
    f_vector = {}
    for fn, img in img_dir.items():
        f_vector[fn] = feature_vector(img, model_)
    return f_vector


n_clusters = 150
path_to_files = "/home/kow/CutOutWiz/Dataset/RAW"
path_to_cluster = "/home/kow/CutOutWiz/Dataset/CLUSTERS"
img_directory = get_files(path_to_files, size=(224, 224))

# Create Keras NN model.
model = get_model()

# # Feed images through the model and extract feature vectors.
img_feature_vector = feature_vectors(img_directory, model)
images = list(img_feature_vector.values())
fns = list(img_feature_vector.keys())
# File name necessary to write image
file_names = list(img_directory.keys())

# K-Means clustering
k_means = KMeans(n_clusters=n_clusters, init='random', algorithm='auto', random_state=100)
y_fit = k_means.fit(images)
y_k_means = k_means.predict(images)

# Create folder for cluster savings
root = os.walk(path_to_cluster)
for c in range(0, n_clusters):
    if not os.path.exists(path_to_cluster + '/' + 'cluster_' + str(c)):
        os.mkdir(path_to_cluster + '/' + 'cluster_' + str(c))

# Write Image on folder
for fn, cluster in zip(file_names, y_k_means):
    image = cv2.imread(os.path.join(path_to_files, fn))
    cv2.imwrite(path_to_cluster + '/' + 'cluster_' + str(cluster) + '/' + fn, image)
