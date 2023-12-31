from layer_def import Dense, Tanh, mse, mse_prime
import numpy as np
from network import train, predict

from keras.datasets import mnist
from keras.utils import to_categorical

from colorama import Fore, Back, Style


#create network
network = [
    Dense(28 * 28, 40), 
    Tanh(), 
    Dense(40, 20), 
    Tanh(),
    Dense(20, 10)
]

def preprocess_data(x, y, limit):
    # reshape and normalize input data
    x = x.reshape(x.shape[0], 28 * 28, 1)
    x = x.astype("float32") / 255
    # encode output which is a number in range [0,9] into a vector of size 10
    # e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    y = to_categorical(y)
    y = y.reshape(y.shape[0], 10, 1)
    return x[:limit], y[:limit]

# train model on MNIST number data
def trainMnist(network, numOfEpochs):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, y_train = preprocess_data(x_train, y_train, 1000)
    x_test, y_test = preprocess_data(x_test, y_test, 20)
    
    #train
    train(network, mse, mse_prime, x_train, y_train, epochs=numOfEpochs, learning_rate=0.1)

    #--------------------------------------------------------------------------------

    print(Fore.LIGHTBLACK_EX + "---------Sample---------")

    for x, y in zip(x_test, y_test):
        output = predict(network, x)
        if(np.argmax(output) == np.argmax(y)):
            print(Fore.GREEN)
        else:
            print(Fore.RED)
        print('pred:', np.argmax(output), '\ttrue:', np.argmax(y))

