from docutils.nodes import version
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, InputLayer, Dropout
from extract_images_from_video import load_data_train,load_data_test
from model import video_classifier_model
import matplotlib.pyplot as plt
from keras.models import load_model

def test():
    #base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    X_test, Y_test = load_data_test()
    #X_train, X_valid, Y_train, Y_valid = load_data_train()
    print('X_test shape : {}'.format(X_test.shape))
    print('Y_test shape : {}'.format(Y_test.shape))
    #X_test = base_model.predict(X_test)
    #X_test = X_test.reshape(186, 224*224*3)
    #X_test = X_test/X_test.max()

    video_classifier = load_model('model/emb_epoch10.h5py')
    predictions = video_classifier.predict_classes(X_test)
    print("the screen time of seeing none action is ", predictions[predictions==0].shape[0], "sec")
    print("the screen time of seeing Integrity persone is ", predictions[predictions==1].shape[0], "sec")
    print("the screen time of Persone whoe phones at work is ", predictions[predictions==2].shape[0], "sec")
    print("the screen time of seeing Person who makes adultery is ", predictions[predictions==3].shape[0], "sec")
    print("the screen time of seeing Person who makes make-up and beauty is ", predictions[predictions==4].shape[0], "sec")
    print("the screen time of seeing Person who eats at work is ", predictions[predictions==5].shape[0], "sec")
    print("the screen time of seeing phone ring is ", predictions[predictions==6].shape[0], "sec")
    print("the screen time of seeing Person who make discussion is ", predictions[predictions==7].shape[0], "sec")
    print("the screen time of seeing Person who plays at work is ", predictions[predictions==8].shape[0], "sec")
    print("the screen time of seeing Person who makes bothering is ", predictions[predictions==9].shape[0], "sec")
    print("the screen time of seeing Person who is selfish is ", predictions[predictions==10].shape[0], "sec")
    print("the screen time of seeing Person who is not polite is ", predictions[predictions==11].shape[0], "sec")
    print("the screen time of seeing Person who is stealer is ", predictions[predictions==12].shape[0], "sec")

    #evaluation
    evaluation = video_classifier.evaluate(X_test, Y_test, verbose=1)
    print('\n loss : {} %'.format(evaluation[0]*100))
    print('accuracy : {} %'.format(evaluation[1]*100))
if __name__=="__main__":
    test()