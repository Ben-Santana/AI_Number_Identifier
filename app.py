from cmath import cos, sin
import math
import numpy as np
import pygame
import random
import mnist as mn
import layer_def as ld
import network as net
from colorama import Fore, Back, Style

from keras.datasets import mnist

pygame.init()

#screen settings
screen = pygame.display.set_mode((700, 561))
backgroundColor = [55, 55, 55]
screen.fill(backgroundColor)

appRunning = True
delta_time = 0.0
clock = pygame.time.Clock()

#load neural network from JSON file
ld.load(mn.network)

# create input interface array
inputArray = np.zeros((28, 28))

#check for interval to predict value
prediction_counter = 0

#-----FOR TESTING PURPOSES------
trainingSetCounter = 300
#-------------------------------



def initMnist(epochs):
    global inputArray

    #load from JSON save file
    ld.load(mn.network)

    #train network
    mn.trainMnist(mn.network, epochs)

    #save newly trained model
    ld.save(mn.network)


#draws window to write number
def drawInputArray():

    #loop through input array
    for x in range(28):
        for y in range(28):
            #set color according to array value
            color = (
                min(255, int(220 * inputArray[x][y] + 35)),
                min(255, int(220 * inputArray[x][y] + 35)),
                min(255, int(220 * inputArray[x][y] + 35))
            )

            #draw corresponding rectangle
            pygame.draw.rect(screen, color, pygame.Rect((y*20), (x*20), 19, 19))

def predictRepeat():
    global prediction_counter
    if(prediction_counter%144 == 0): 
        mnistPredict()
        prediction_counter = 1
    else: prediction_counter += 1


#input users number into neural network, then output and return prediction
def mnistPredict():
    global inputArray
    
    # Ensure inputArray is a NumPy array and has the correct shape
    inputArray = inputArray.reshape(784, 1)
    inputArray = inputArray.astype("float32")

    #get prediction array, most probable number and certainty from neural network
    prediction = net.predict(mn.network, inputArray)
    number = np.argmax(prediction)

    #reset to original shape
    inputArray = inputArray.reshape(28, 28)

    #print prediction
    print(Fore.GREEN + str(number))

    #return prediction
    return number


def setArray(index):
    global inputArray

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, y_train = mn.preprocess_data(x_train, y_train, 1000)
    x_test, y_test = mn.preprocess_data(x_test, y_test, 20)

    inputArray = inputArray.reshape(784, 1)
    inputArray = inputArray.astype("float32")

    inputArray = x_train[index]

    inputArray = inputArray.reshape(28, 28)




def mousePressed(button):
    for x in range(28):
        for y in range(28):
            mousex, mousey = pygame.mouse.get_pos()
            if (mousex > (x*20)) and (mousex < (x*20) + 19) and (mousey > (y*20)) and (mousey < (y*20) + 19):

                #check for left click
                if button == 0:
                    inputArray[y][x] = 1.0
                    try:
                        #draw on array
                        inputArray[y+1][x+1] = 1.0
                        inputArray[y+1][x] = 1.0
                        inputArray[y][x+1] = 1.0
                    except:
                        pass

                #check for left click
                if button == 1:
                    inputArray[y][x] = 0.0
                    try:
                        #erase from array
                        inputArray[y+1][x+1] = 0.0
                        inputArray[y+1][x] = 0.0
                        inputArray[y][x+1] = 0.0
                    except:
                        pass
    pass

def update():
    predictRepeat()
    pass


def draw():
    drawInputArray()
    pass



while appRunning:
    if pygame.mouse.get_pressed()[0]: mousePressed(0)
    if pygame.mouse.get_pressed()[2]: mousePressed(1)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            appRunning = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                #reset input values
                inputArray = np.zeros((28, 28))
            if event.key == pygame.K_ESCAPE:
                appRunning = False
                print(Fore.RED + "EXIT")
            if event.key == pygame.K_RETURN:
                initMnist(int(input("Epochs: ")))
            if event.key == pygame.K_0:
                setArray(trainingSetCounter)
                trainingSetCounter += 3
                
        

    update()
    draw()

    pygame.display.flip()

    delta_time = 0.001 * clock.tick(144)


pygame.quit()