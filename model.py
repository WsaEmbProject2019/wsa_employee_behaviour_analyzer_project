from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dropout, Input, Dense, Flatten, UpSampling2D
from keras.layers import LeakyReLU
import keras

num_classes = 13

def video_classifier_model():
    model = Sequential()
    model.add(Conv2D(filters=32, input_shape=(224, 224, 3), kernel_size=(3, 3), activation='relu', padding='same'))
    #model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
    #model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
    #model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    #model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.3))

    model.add(Dense(num_classes, activation='sigmoid'))

    #compile model
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    print (model.summary())
    #plot_model(model, to_file='model/mnist_model_rochel.png')
    return model


if __name__=="__main__":
    video_classifier_model()