from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from extract_images_from_video import load_data_valid,load_data_test, extract_frames_from_video_valid
from keras.utils import to_categorical
from model import video_classifier_model
import numpy as np
from os import listdir
import os, sys
from keras.preprocessing.image import load_img, img_to_array

listInt = []
def getIndex(YList):
    index = 0
    #print("YList : ", YList)
    for data in YList:
        #print("data : ", data)
        if data == 1:
            retour = index
        else:
            index += 1
    listInt.append(retour)
    return retour

def analyze_behaviour():
    # define categorical label
    label_dict = {
        0: 'Nothing in the current frame',
        1: 'Integrity and Duty',
        2: 'Phone at work',
        2: 'Phone at work',
        3: 'Harassment',
        4: 'Beauty and make-up',
        5: 'Eating at work',
        6: 'Phone Ring',
        7: 'Debat and Discussion',
        8: 'Play at work',
        9: 'Bothering',
        10: 'Selfishness',
        11: 'Impoliteness',
        12: 'Stealing tools',
    }
    #load data
    X_test, Y_test = load_data_test()
    print('X_test shape : {}'.format(X_test.shape))
    print('Y_test shape : {}'.format(Y_test.shape))

    #file2 = "Office_Etiquette_Final.mp4"
    #extract_frames_from_video_valid(file2)
    video = load_data_valid()
    print('input image shape : {}'.format(video.shape))

    #call model
    video_classifier = load_model('model/emb_epoch10.h5py')
    #print(classifierCNN.summary())

    # predict label category rely for the input image
    predict = video_classifier.predict(video)
    print("predicted : ", predict)
    predict = predict.astype('int')
    #predict = np.argmax(np.round(predict), axis=1)
    print("predicted : ", predict)
    #predict = to_categorical(predict, num_classes=13)
    #print("predicted : ", predict)
    print("predicted[0] : ", predict[0])
    print("predicted[1] : ", predict[1])
    print('predict class shape : {}'.format(predict.shape))
    print('right class shape : {}'.format(Y_test.shape))

    # show result
    i = 0
    for frame in listdir("generated_frames_valid/"):
        res = label_dict[getIndex(predict[i])]
        print("frame : {}, behaviour found : {}".format(frame, res))
        #img = plt.imread("generated_frames_valid/valid%d.jpg"%i)
        #plt.imshow(img)
        #plt.show()
        i+=1

if __name__=="__main__":
    analyze_behaviour()
    #print("listInt", listInt)