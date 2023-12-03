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

#pygame settings
screen = pygame.display.set_mode((650, 561))
white = [55, 55, 55]
screen.fill(white)
appRunning = True
delta_time = 0.0
clock = pygame.time.Clock()

#load neural network from JSON file
ld.load(mn.network)

# create input interface array
inputArray = np.zeros((28, 28))

#check for interval to predict value
prediction_counter = 0

#ignore printing 0 when input array is empty
zeroIgnore = 0

#ignroe repeated predictions
last_predict = 0

#how certain the model has to be in order to print
certainty_threshold = 0.4

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
    if(prediction_counter%60 == 0): 
        mnistPredict()
        prediction_counter = 1
    else: prediction_counter += 1


#input users number into neural network, then output and return prediction
def mnistPredict():
    global inputArray
    global zeroIgnore
    global last_predict
    
    # Ensure inputArray is a NumPy array and has the correct shape
    inputArray = inputArray.reshape(784, 1)
    inputArray = inputArray.astype("float32")

    #get prediction array, most probable number and certainty from neural network
    prediction = net.predict(mn.network, inputArray)
    number = np.argmax(prediction)
    certainty = prediction[number]

    #set zeroInore to certainty to cancel zeros when input array is empty
    if zeroIgnore == 0:
        zeroIgnore = certainty

    #reset to original shape
    inputArray = inputArray.reshape(28, 28)

    #return prediction
    if certainty > certainty_threshold and certainty != zeroIgnore and number != last_predict:
        #print prediction
        print(Fore.GREEN + str(number) + Fore.CYAN + ", " + str(certainty))

        #store certainy to avoid printing again
        last_predict = number

        return number
    
    #return -1 in case of no valid prediction
    return -1




def mousePressed(button):
    for x in range(28):
        for y in range(28):
            mousex, mousey = pygame.mouse.get_pos()
            if (mousex > (x*20)) and (mousex < (x*20) + 19) and (mousey > (y*20)) and (mousey < (y*20) + 19):
                if button == 0:
                    inputArray[y][x] = 1.0
                    try:
                        inputArray[y+1][x+1] = 1.0
                        inputArray[y+1][x] = 1.0
                        inputArray[y][x+1] = 1.0
                    except:
                        pass
                if button == 1:
                    inputArray[y][x] = 0.0
                    try:
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
                
        

    update()
    draw()

    pygame.display.flip()

    delta_time = 0.001 * clock.tick(144)


pygame.quit()