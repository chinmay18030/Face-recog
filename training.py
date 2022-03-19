import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout,Flatten
from keras.layers.convolutional import Conv2D,MaxPooling2D
import pickle
from sklearn.preprocessing import LabelEncoder

images = np.load("/content/images.npy")
classNo = np.load("/content/classes.npy")

def preProcessing(img):
  img = cv2.equalizeHist(img)
  img = img/255
  return img


for i in range(len(classNo)):
  if classNo[i]== "main":
    classNo[i] = 1
  if classNo[i]== "nothing":
    classNo[i] = 0

x_train, x_test, y_train, y_test = train_test_split(images, classNo, test_size=0.2)

print(classNo)

X_train= np.array(list(map(preProcessing,x_train)))
X_test= np.array(list(map(preProcessing,x_test)))


X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)

#### IMAGE AUGMENTATION
dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             rotation_range=10)
dataGen.fit(X_train)
noOfClasses = 2
#### ONE HOT ENCODING OF MATRICES
y_train = to_categorical(y_train,2)
y_test = to_categorical(y_test,2)


def myModel():
    noOfFilters = 60
    sizeOfFilter1 = (5,5)
    sizeOfFilter2 = (3, 3)
    sizeOfPool = (2,2)
    noOfNodes= 500
    imageDimensions= (32,32,3)

    model = Sequential()
    model.add((Conv2D(noOfFilters,sizeOfFilter1,input_shape=(imageDimensions[0],
                      imageDimensions[1],1),activation='relu')))
    model.add((Conv2D(noOfFilters, sizeOfFilter1, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add((Conv2D(noOfFilters//2, sizeOfFilter2, activation='relu')))
    model.add((Conv2D(noOfFilters//2, sizeOfFilter2, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(noOfNodes,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noOfClasses, activation='softmax'))

    model.compile(optimizer="adam",loss='categorical_crossentropy',metrics=['accuracy'])
    return model

model = myModel()
model.fit(X_train, y_train, epochs=5, batch_size=45)

model.evaluate(X_test,y_test)
