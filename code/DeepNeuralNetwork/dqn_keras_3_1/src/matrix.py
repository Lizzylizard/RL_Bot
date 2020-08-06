#!/usr/bin/env python

# import own scripts
import main as rm
import image as mi

# import numpy
import numpy as np
from numpy import random

# Import OpenCV libraries and tools
import cv2 as cv
from cv_bridge import CvBridge, CvBridgeError

# ROS
import rospy
import rospkg
from std_msgs.msg import String, Float32, Int32
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import Twist
from gazebo_msgs.srv import GetModelState
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState

# other
import math
import time


class Bot:

  # constructor
  def __init__(self):
    # action space
    self.actions = np.arange(7)
    '''
    0 = sharp left, 1 = left, 2 = slightly left, 3 = forward, 
    4 = slightly right, 5 = right, 6 = sharp right, (not in array: 7 = stop)
    '''
    self.stop_action = 7

    # state space
    self.states = np.arange(8)
    '''
    0 = line is far left, 1 = line is left, 2 = line is slightly left, 3 = line is in the middle, 
    4 = line is slightly right, 5 = line is right, 6 = line is far right, 7 = line is lost 
    '''
    self.lost_line = 7

    # q-matrix (empty in the beginning)
    self.Q = np.zeros(shape=[len(self.states), len(self.actions)])

    # image helper
    self.img_helper = mi.MyImage()

  # returns the reward for a given state
  def calculate_reward(self, curr_state):
    if (curr_state == 3):
      # best case: middle
      reward = 0
    elif (curr_state == 2):
      # second best case: slightly left
      reward = -2
    elif (curr_state == 1):
      # bad case: left
      reward = -3
    elif (curr_state == 0):
      # worse case: far left
      reward = -4
    elif (curr_state == 4):
      # second best case: slightly right
      reward = -2
    elif (curr_state == 5):
      # bad case: right
      reward = -3
    elif (curr_state == 6):
      # worse case: far right
      reward = -4
    else:
      # worst case: line is lost
      reward = (-1000)

    return reward

  # check where the line is --> check current state of the robot
  def get_state(self, img):
    line_state = self.img_helper.get_line_state(img)
    return line_state

  # explore by choosing a random action
  def explore(self, img):
    # choose one random action
    action_arr = np.random.choice(self.actions, 1)
    action = action_arr[0]
    return action

  # choose best action by getting max value out of the q-matrix
  def exploit(self, img, state):
    action = np.argmax(self.Q[state, :])
    return action

  # fill q-matrix -> Bellman equation
  def update_q_table(self, curr_state, action, alpha, reward, gamma,
                     next_state):
    # update q-matrix
    self.Q[curr_state, action] = (1 - alpha) * self.Q[
      curr_state, action] + alpha * (reward + gamma * np.max(
        self.Q[next_state, :]))

  # use filled q-matrix to simply drive
  def drive(self, img):
    state = self.get_state(img)
    action = np.argmax(self.Q[state, :])
    if (state == self.lost_line):
      # stop robot if line is lost
      action = self.stop_action
    return action

    # save q-matrix as a .txt-file

  def save_q_matrix(self, end, total_learning_time, episodes,
                    minutes_learning, total, minutes, distance,
                    speed):
    try:
      # open correct file
      f = open(
        "/home/elisabeth/catkin_ws/src/Q-Learning/rl_matrix/src"
        "/Q_Matrix"
        "/Code/Learn_Simple_3/Q-Matrix-Records.txt",
        "a")
      # f = open("../Q_Matrix/Q-Matrix-Records.txt", "a")

      # pretty print matrix
      string = self.printMatrix(end)

      # add statistics
      string += ("\nLearning time = " + str(
        total_learning_time) + " seconds = " \
                 + str(minutes_learning) + " minutes")
      string += ("\nNumber Episodes = " + str(episodes))
      string += ("\nTotal time = " + str(total) + " seconds = " + str(
        minutes) + " minutes")
      string += ("\nDistance = " + str(distance) + " meters")
      string += ("\nSpeed = " + str(speed) + " m/s)")

      # write into file
      f.write(string)

      # close file
      f.close()
    except Exception as e:
      print(str(e) + "\nFile not written")

  # pretty print matrix
  def printMatrix(self, end):
    readable_time = time.ctime(end)
    string = "\n\nICH\n"
    string += (str(readable_time) + ")\n[")
    for i in range(len(self.Q)):
      string += " ["
      row_max = np.argmax(self.Q[i, :])
      number_arr = np.round(self.Q[i], 3)
      for j in range(len(self.Q[i])):
        number = number_arr[j]
        if (j == row_max):
          number_str = "**{:.3f}**,".format(number)
          number_str = number_str.center(14)
          string += number_str
        else:
          number_str = "{:.3f},".format(number)
          number_str = number_str.center(14)
          string += number_str
      string += "]\n"
    string += "]"
    print(string + "\n")
    return string

  # use pre defined q matrix to drive, to see whether driving works
  # or not (not in use anymore)
  def own_q_matrix(self, img):
    q = np.zeros(shape=[len(self.states), len(self.actions)])
    # line = far left, action = sharp left
    q[0] = [1, 0, 0, 0, 0, 0, 0]
    # line = left, action = left
    q[1] = [0, 1, 0, 0, 0, 0, 0]
    # line = slightly left, action = slightly left
    q[2] = [0, 0, 1, 0, 0, 0, 0]
    # line = middle, action = forward
    q[3] = [0, 0, 0, 1, 0, 0, 0]
    # line = slightly right, action = slightly right
    q[4] = [0, 0, 0, 0, 1, 0, 0]
    # line = right, action = right
    q[5] = [0, 0, 0, 0, 0, 1, 0]
    # line = far right, action = sharp right
    q[6] = [0, 0, 0, 0, 0, 0, 1]
    # line = lost, action = stop
    q[7] = [0, 0, 0, 0, 0, 0, 0]

    state = self.get_state(img)
    if (state == self.lost_line):
      action = self.stop_action
    else:
      action = np.argmax(q[state, :])
    return action
