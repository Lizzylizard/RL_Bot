#!/usr/bin/env python

import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as mpimg

# show image with matplotlib
def show(middle, title):
    plt.title('image_' + str(title))
    imgplot = plt.imshow(img)
    plt.show()
    
# count background pixel
def count_pixel(img):
    cnt = 0
    for i in range(len(img[0])):
      if (img[0, i] > 50):
        cnt += 1
      else:
        break
    return cnt

# calculate state (how close to the middle of the line is the robot)
def get_state(img):
    left = count_pixel(img)
    right = count_pixel(np.flip(img, 1))
    width = np.size(img[0])
    abs_right = (width-right)
    middle = float(left + abs_right) / 2.0
    opt_middle = float(width / 2.0)

    dist = opt_middle - middle
    # if positive => line is left
    # if negative => line is right
    if(middle >= (width * (48.0/100.0)) and middle <= (width *(52.0/100.0))):
      # line is in the middle
      dist = 0
    if (left >= (width * (99.0 / 100.0)) or right >= (
      width * (99.0 / 100.0))):
      # line is lost
      # just define that if line is ALMOST lost, it is completely
      # lost, so terminal state gets reached
      dist = 25
      

    state = dist
    return state, middle


def get_reward(state):
    abs_state = abs(state)
    if(abs_state == 0):
      reward = 1
    else:
      reward = float(abs_state / 25.0) * (-1)
    return reward

if __name__ == '__main__':

    path = '/home/elisabeth/catkin_ws/src/DeepNeuralNetwork/dqn_neu_2/test_images'
    
    for i in range(1000, 1010):
        image_path = path + '/image_' + str(i) + '.png'
        img = mpimg.imread(image_path)
        state, middle = get_state(img)
        reward = get_reward(state)
        show(middle, str(i) + ", state = " + str(state) + ", reward = " + str(reward))
