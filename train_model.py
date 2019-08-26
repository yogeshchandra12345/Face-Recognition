import cv2
import numpy as np
import os
from os.path import join

data_path = './faces/'

images_ = [_img for _img in os.listdir(data_path) if '.jpg' in join(data_path, _img)]

training_data, labels = [], []

for index, file_ in enumerate(images_):
    image_path = data_path + file_
    image_ = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    training_data.append(np.asarray(image_, dtype=np.uint8))
    labels.append(index)

labels = np.asarray(labels, dtype=np.int32)

model = cv2.face.LBPHFaceRecognizer_create()
model.train(np.asarray(training_data), np.asarray(labels))
# save the model to disk
model.save('model.xml')

print(" Model Training Complete !!!!! ")


