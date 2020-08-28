#!/usr/bin/env python

# own scripts
import main
import Network

# numpy
import numpy as np

# ROS
import rospy
import rospkg
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from gazebo_msgs.srv import GetModelState
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState

# global variables
global robot

# shutdown
def shutdown():
  print("Shutdown")
  global robot
  robot.stop()
  robot.set_position()

# main
if __name__ == '__main__':
  # robot instance
  global robot
  robot = main.Robot()

  # user input
  net = input("Network number: ")
  val = input("Number of episodes to test on: ")
  speed = input("Speed of robot: ")

  # file paths
  net_number = net
  training_net_path = \
    "/home/elisabeth/catkin_ws/src/DeepNeuralNetwork/dqn_neu_1" \
    "/Training/Training_Network_" + str(net_number) + ".h5"

  # hyperparameters
  image_length = robot.image_length
  nr_of_images = robot.images_in_one_input
  net_input_size = nr_of_images * image_length
  #net_input_size = 1
  number_episodes = val

  # networks
  training_network = Network.Network(net_input_size)
  training_network = training_network.load_model(training_net_path)

  # visualization
  robot.speed = speed
  robot.instantiate_node()
  robot.publish_action(7)
  rospy.on_shutdown(shutdown)

  # one image
  current_image, next_index = robot.get_image()
  last_image = current_image.copy()

  # multiple images
  last_index = next_index
  stack = robot.get_multiple_images(last_index, next_index)
  mult_images = robot.get_correct_nr_of_images(stack)
  last_mult_images = mult_images.copy()

  # starting state
  state = robot.get_state(current_image)
  last_state = state

  # main program
  try:
    i = 0
    while not rospy.is_shutdown():
      # save last image(s)
      last_image = current_image.copy()
      last_mult_images = mult_images.copy()
      # get current image and its' index
      last_index = next_index
      current_image, next_index = robot.get_image()
      # get all the images that were received during the last step
      stack = robot.get_multiple_images(last_index, next_index)
      mult_images = robot.get_correct_nr_of_images(stack)

      if(i <= number_episodes):
        # select action
        state = np.array([state])
        # action_values = training_network.predict(state)
        action_values = training_network.predict(mult_images)
        action = np.argmax(action_values)
        # execute action
        robot.publish_action(action)
        # save last state
        last_state = state
        # get resulting state
        state = robot.get_state(current_image)
        # check if episode done
        done = False
        if(state == 7):
          robot.stop()
          robot.set_position()
          print("Stopped")
        # print
        print("Last state \t = " + str(last_state))
        print("Action \t = " + str(action))
        print("Resulting state = " + str(state))
        print("Episode \t = " + str(i))
        print("-"*60)
        i += 1
      # continue if user wishes to
      else:
        cont = input("Continue?: ")
        if(cont):
          val = input("Number of episodes to test on: ")
          i = 0
          number_episodes = val
        else:
          break

  except rospy.ROSInterruptException:
    print("Exception")
    robot.stop()