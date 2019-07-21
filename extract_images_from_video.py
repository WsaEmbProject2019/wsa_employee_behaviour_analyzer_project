import cv2
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from skimage.transform import resize
import os
from os import listdir

file1 = "Office_Etiquette.mp4"
file2 = "Office_Etiquette_Final.mp4"
data_path = "data/"
n_classes=13

def extract_frames_from_video_train():
    count = 0
    video_file = data_path + file1
    cap = cv2.VideoCapture(video_file)
    frameRate = cap.get(5)
    while(cap.isOpened()):
        frameId = cap.get(1)
        ret, frame = cap.read()
        if(ret!=True):
            break
        if(frameId%math.floor(frameRate)==0):
            filename = "train%d.jpg"%count
            count+=1
            print(filename)
            cv2.imwrite("generated_frames_train/" + filename, frame)
    cap.release()
    print("Done!!!")

def extract_frames_from_video_test():
    count = 0
    video_file = data_path + file2
    cap = cv2.VideoCapture(video_file)
    frameRate = cap.get(5)
    while(cap.isOpened()):
        frameId = cap.get(1)
        ret, frame = cap.read()
        if(ret!=True):
            break
        if(frameId%math.floor(frameRate)==0):
            filename = "test%d.jpg"%count
            count+=1
            print(filename)
            cv2.imwrite("generated_frames_test/" + filename, frame)
    cap.release()
    print("Done!!!")

def extract_frames_from_video_valid(file3):
    count = 0
    video_file = data_path + file3
    cap = cv2.VideoCapture(video_file)
    frameRate = cap.get(5)
    while(cap.isOpened()):
        frameId = cap.get(1)
        ret, frame = cap.read()
        if(ret!=True):
            break
        if(frameId%math.floor(frameRate)==0):
            filename = "valid%d.jpg"%count
            count+=1
            print(filename)
            cv2.imwrite("generated_frames_valid/" + filename, frame)
    cap.release()
    print("Done!!!")

def visualize_frame_from_video_train():
    img = plt.imread("generated_frames_train/train0.jpg")
    plt.imshow(img)
    plt.show()

def visualize_frame_from_video_test():
    img = plt.imread("generated_frames_test/test0.jpg")
    plt.imshow(img)
    plt.show()

def visualize_frame_from_video_valid():
    img = plt.imread("generated_frames_test/valid0.jpg")
    plt.imshow(img)
    plt.show()


def load_data_train():
    X_train = []
    X_image = []
    data_train = pd.read_csv(data_path + "train.csv")
    print(data_train.head())
    #image train
    for img_name in data_train.Image_ID :
        image = plt.imread("generated_frames_train/" + img_name)
        X_train.append(image)
    X_train = np.array(X_train)
    for i in range(0, X_train.shape[0]):
        img = resize(X_train[i], preserve_range=True, output_shape=(224, 224)).astype(int)
        X_image.append(img)
    X_image = np.array(X_image)
    #label train
    Y_train = data_train.Class
    Y_train = to_categorical(Y_train, num_classes=n_classes)
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X_image, Y_train, random_state=0, test_size=0.2)
    return Xtrain, Xtest, Ytrain, Ytest

def load_data_test():
    X_test = []
    X_test_im = []
    data_test = pd.read_csv(data_path + "test.csv")
    print(data_test.head())
    #image train
    for img_name in data_test.Image_ID:
        img = plt.imread("generated_frames_test/" + img_name)
        X_test.append(img)
    X_test = np.array(X_test)
    for i in range(0, X_test.shape[0]):
        img = resize(X_test[i], preserve_range=True, output_shape=(224, 224)).astype(int)
        X_test_im.append(img)
    X_test_im = np.array(X_test_im)
    #label train
    Y_test = data_test.Class
    Y_test = to_categorical(Y_test, num_classes=n_classes)
    return X_test_im, Y_test

def load_data_valid():
    X_valid = []
    X_valid_im = []
    #image train
    for img_name in listdir("generated_frames_valid/"):
        img = plt.imread("generated_frames_valid/" + img_name)
        X_valid.append(img)
    X_valid = np.array(X_valid)
    for i in range(0, X_valid.shape[0]):
        img = resize(X_valid[i], preserve_range=True, output_shape=(224, 224)).astype(int)
        X_valid_im.append(img)
    X_valid_im = np.array(X_valid_im)
    return X_valid_im

if __name__=="__main__":
    #extract_frames_from_video_train()
    #visualize_frame_from_video_train()
    #extract_frames_from_video_test()
    #visualize_frame_from_video_test()
    Xtrain, Xtest1, Ytrain, Ytest1 = load_data_train()
    Xtest, Ytest = load_data_test()
    print("Xtrain shape : {}, Ytrain shape : {}".format(Xtrain.shape, Ytrain.shape))
    print("Xtest1 shape : {}, Ytest1 shape : {}".format(Xtest1.shape, Ytest1.shape))
    print("Xtest shape : {}, Ytest shape : {}".format(Xtest.shape, Ytest.shape))