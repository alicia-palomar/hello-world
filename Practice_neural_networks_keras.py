import numpy as np
import matplotlib.pyplot as plt

from keras import layers
from keras import models
import tensorflow as tf

##MNIST DATASET
model = models.Sequential()
model.add(layers.Conv2D(32, (5, 5), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D(2, 2))

#with the summary() method, we can see the information about the parameters of each layer and shape of
#output tensors of each layer:
model.summary()

#Simple deep neural network
model2 = models.Sequential()
model2.add(layers.Conv2D(32,(5, 5), activation = 'relu', input_shape=(28, 28, 1)))
model2.add(layers.MaxPooling2D((2, 2)))
model2.add(layers.Conv2D(64, (5, 5), activation='relu'))
model2.add(layers.MaxPooling2D((2, 2)))
model2.summary()
 #adding a densely connected layer, which will serve to feed a final layer of softmax. We have to first flatten the 3D tensor to one of 1D
 #in order to adjust the tensors to the input of the dense layers like the softmax, wwhich is a 1D tensor, while the output of the
 #previous one is a 3D tensor.

model2.add(layers.Flatten())
model2.add(layers.Dense(10, activation='softmax'))
model2.summary()

#Training and evaluation of the model (https://github.com/jorditorresBCN/DEEP-LEARNING-practical-introduction-with-Keras/blob/master/Densely-connected-networks.ipynb)

from keras.datasets import mnist
from keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.ndim)  # 3
print(x_train.shape)  # (60000,28,28)
print(x_test.shape)  # (10000, 28, 28)
print(x_train.dtype)  # uint8

plt.figure()
plt.imshow(x_train[8], cmap=plt.cm.binary)
print(y_train[8])


x_train = x_train.reshape((60000, 28, 28, 1))
x_test = x_test.reshape((10000, 28, 28, 1))
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

model2.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model2.fit(x_train, y_train, batch_size=100, epochs=5, verbose=1)

#The batch size defines the number of samples that will be propagated through the network. For instance, let's say you
# have 1050 training samples and you want to set up a batch_size equal to 100. The algorithm takes the first 100 samples
# (from 1st to 100th) from the training dataset and trains the network. Next, it takes the second 100 samples (from
# 101st to 200th) and trains the network again. We can keep doing this procedure until we have propagated all samples
# through of the network.

test_loss, test_acc = model2.evaluate(x_test, y_test)

print('Test loss:', test_loss)
print('Test accuracy:', test_acc)

y_pred=model2.predict(x_test) # For each case, an array of probabilities to be from each class
y_pred_classes=np.argmax(y_pred, axis=1)

y_test_classes=np.argmax(y_test, axis=1)

## Confusion matrix (sklearn website)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = numpy.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Actual class')
    plt.xlabel('Predicted class')

from sklearn.metrics import confusion_matrix
import itertools

confusion_mtx = confusion_matrix(y_test_classes, y_pred_classes)
plot_confusion_matrix(confusion_mtx, classes=range(10))