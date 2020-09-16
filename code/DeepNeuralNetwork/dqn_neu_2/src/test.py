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
    "/home/elisabeth/catkin_ws/src/DeepNeuralNetwork/dqn_neu_2" \
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

  # stop counter
  stop_counter = 0

  # skip images until stack is full
  for i in range(nr_of_images):
    robot.get_image()

  # one image
  current_image = robot.get_image()
  last_image = current_image.copy()

  # multiple images
  stack = robot.numpy_image_buffer.copy()
  stack = stack.flatten()
  stack = np.array([stack])

  # starting state
  state = robot.get_state(current_image)
  last_state = state

  # main program
  try:
    i = 0
    while not rospy.is_shutdown():
      # save last image(s)
      last_image = current_image.copy()
      last_stack = stack.copy()
      # get current image
      current_image = robot.get_image()
      # get the last couple of images
      stack = robot.numpy_image_buffer.copy()
      stack = stack.flatten()
      stack = np.array([stack])

      if(i <= number_episodes):
        # select action
        state = np.array([state])
        # action_values = training_network.predict(state)
        action_values = training_network.predict(stack)
        action = np.argmax(action_values)
        # execute action
        robot.publish_action(action)
        # save last state
        last_state = state
        # get resulting state
        state = robot.get_state(current_image)
        # check if episode done
        done = False
        if(state == robot.lost_state):
          robot.stop()
          robot.set_position()
          stop_counter += 1
          print("Stopped")
        # print
        print("Last state \t = " + str(last_state))
        print("Action \t = " + str(action))
        print("Resulting state = " + str(state))
        print("Episode \t = " + str(i))
        print("Stops until now = " + str(stop_counter/2))
        print("-"*60)
        i += 1
      # continue if user wishes to
      else:
        # stop robot
        robot.publish_action(7)
        # ask whether to continue or not
        cont = input("Continue?: ")
        if(cont):
          val = input("Number of episodes to test on: ")
          robot.speed = input("Speed of robot: ")
          i = 0
          number_episodes = val
        else:
          break

  except rospy.ROSInterruptException:
    print("Exception")
    robot.stop()