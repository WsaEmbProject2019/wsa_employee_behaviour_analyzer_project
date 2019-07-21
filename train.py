from keras.applications.vgg16 import VGG16
from keras.layers import Dense, InputLayer, Dropout
from extract_images_from_video import load_data_train
from model import video_classifier_model
import matplotlib.pyplot as plt

def train():
    #base_model = VGG16(weights='imagenet', include_top='False', input_shape=(224, 224, 3))
    X_train, X_valid, Y_train, Y_valid = load_data_train()
    #X_train = base_model.predict(X_train)
    #X_valid = base_model.predict(X_valid)
    print('X_train shape : {}, X_test shape : {}'.format(X_train.shape, X_valid.shape))
    #X_train = X_train.reshape(208, 224*224*3)
    #X_valid = X_valid.reshape(90, 224*224*3)
    #train = X_train/X_train.max()
    #X_valid = X_valid/X_train.max()
    #print('train shape : {}, X_test shape : {}'.format(train.shape, X_valid.shape))

    video_classifier = video_classifier_model()
    video_classifier_train = video_classifier.fit(X_train, Y_train, epochs=10, batch_size=64, validation_data=(X_valid, Y_valid), verbose=1)
    video_classifier.save('model/emb_epoch10.h5py')
    accuracy = video_classifier_train.history['acc']
    accuracy_eval = video_classifier_train.history['val_acc']
    loss = video_classifier_train.history['loss']
    loss_eval = video_classifier_train.history['val_loss']

    epochs = range(len(accuracy))
    plt.figure(figsize=[5, 5])
    plt.subplot(121)
    plt.plot(epochs, accuracy, 'bo', label="Training accuracy")
    plt.plot(epochs, accuracy_eval, 'b', label="Validation Accuracy")
    plt.title("Training and validation accuracy")
    plt.legend()
    plt.subplot(122)
    plt.plot(loss, 'bo', label="Training loss")
    plt.plot(epochs, loss_eval, 'b', label="Validation loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.show()

if __name__=="__main__":
    train()