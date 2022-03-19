import numpy
import os
import cv2


def load_dataset(location):
    images = []  # LIST CONTAINING ALL THE IMAGES
    classNo = []  # LIST CONTAINING ALL THE CORRESPONDING CLASS ID OF IMAGES
    myList = os.listdir(location)
    print("Total Classes Detected:", len(myList))
    print(myList)

    print("Importing Classes .......")
    for x in myList:
        myPicList = os.listdir(location + "/" + str(x))
        for y in myPicList:
            curImg = cv2.imread(location + "/" + str(x) + "/" + y)
            curImg = cv2.cvtColor(curImg, cv2.COLOR_BGR2GRAY)
            curImg = cv2.resize(curImg, (32, 32))
            images.append(curImg)
            classNo.append(x)
        print(x, end=" ")
    print(" ")
    print("Total Images in Images List = ", len(images))
    print("Total IDS in classNo List= ", len(classNo))
    return images, classNo

images, classes = load_dataset("image_Data")

numpy.save("images", images)
numpy.save("classes", classes)