import numpy as np
import matplotlib.pyplot as plt

from keras import layers
from keras import models
import tensorflow as tf

# MNIST DATASET
model = models.Sequential()
model.add(layers.Conv2D(32, (5, 5), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D(2, 2))

# with the summary() method, we can see the information about the parameters of each layer and shape of
# output tensors of each layer:
model.summary()

# Simple deep neural network
model2 = models.Sequential() # the Sequential class, which allows the creation of a basic neural network
model2.add(layers.Conv2D(32,(5, 5), activation = 'relu', input_shape=(28, 28, 1)))
model2.add(layers.MaxPooling2D((2, 2)))
model2.add(layers.Conv2D(64, (5, 5), activation='relu'))
model2.add(layers.MaxPooling2D((2, 2)))
model2.summary()
 # adding a densely connected layer, which will serve to feed a final layer of softmax. We have to first flatten the 3D tensor to one of 1D
 # in order to adjust the tensors to the input of the dense layers like the softmax, wwhich is a 1D tensor, while the output of the
 # previous one is a 3D tensor.

model2.add(layers.Flatten())
model2.add(layers.Dense(10, activation='softmax'))
model2.summary()

# Training and evaluation of the model (https://github.com/jorditorresBCN/DEEP-LEARNING-practical-introduction-with-Keras/blob/master/Densely-connected-networks.ipynb)

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
model2.fit(x_train, y_train, batch_size=100, epochs=5, verbose=1) # verbose=1 to see the fitting process

# The batch size defines the number of samples that will be propagated through the network. For instance, let's say you
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

# Confusion matrix (sklearn website)

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

# Densely connected layers ( all the neurons in
# each layer are connected to all the neurons in the next layer)

x_train = x_train.reshape((60000, 784))
x_test = x_test.reshape((10000, 784))

# we transformed the tensor (image) from 2 dimensions (28x28) to a 1D vector in order to facilitate the entry
# of data (This is not necessary for convolutionals)

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Now, we represent the labels with a vector of 10 positions, where the position corresponding to the digit that
# represents the image contains a 1 and the remaining positions contain the value 0. In this case, one-hot encoding is
# used, which transforms the labels (0 .. 9) into a vector of as many zeros as the number of different possible
# labels and containing the value of 1 in the index that corresponds to the value of the label. (Using to_categorical
# from keras.utils)

y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

from keras.models import Sequential

#  the model in Keras is considered as a sequence of layers and each
# of them gradually "distills" the input data to obtain the desired output. In
# Keras we can find all the required types of layers that can be easily added to
# the model through the add() method.

model_dense = Sequential()
model_dense.add(layers.core.Dense(10, activation='sigmoid', input_shape=(784,)))
# input data of the first layer is a tensor of 784 feauures
model_dense.add(layers.core.Dense(10, activation='softmax'))
# softmax layer of 10 neurons, which will return a matrix off 10 probability values representing the 10 possible
# digits. Each value will be the probability that the image of the current digit belongs to each one of them.

model_dense.summary()
# 7850 parameters for the first layer ( 784
# parameters for the weights wij and therefore 10×784 parameters to store the weights of the 10 neurons.
# In addition to the 10 additional parameters for the 10 bj biases corresponding to each one of them.

# In the second layer, being a softmax function, it is required to connect all 10 neurons with the 10 neurons
# of the previous layer. Therefore 10x10 wi parameters are required and in addition 10 bj biases corresponding to each
# node.

# Once we have our model defined, we can configure how its learning process will be with the compile() method,
# with which we can specify some properties through method arguments. The first of these arguments is the loss function
# that we will use to evaluate the degree of error between calculated outputs and the desired outputs of the training
# data. On the other hand, we specify an optimizer that, as we will see, is the way we have to specify the optimization
# algorithm that allows the neural network to calculate the weights of the parameters from the input data and the
# defined loss function.

model_dense.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
# sgd= stocastic gradient descent

model_dense.fit(x_train, y_train, batch_size=100, epochs=5)
test_loss, test_acc = model_dense.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)

predictions = model_dense.predict(x_test)
# The predict() method return a vector with the predictions for the whole dataset elements. We can know which class
# gives the most probability of belonging by means of the argmax function of Numpy, which returns the index of the
# position that contains the highest value of the vector.

np.argmax(predictions[11])
print(predictions[11])


