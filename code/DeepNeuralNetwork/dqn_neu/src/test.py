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
  # user input
  net = input("Network number: ")
  val = input("Number of episodes to test on: ")
  speed = input("Speed of robot: ")

  # file paths
  net_number = net
  training_net_path = \
    "/home/elisabeth/catkin_ws/src/DeepNeuralNetwork/dqn_neu" \
    "/Training/Training_Network_" + str(net_number) + ".h5"

  # hyperparameters
  image_length = 50
  net_input_size = 1 * image_length
  #net_input_size = 1
  number_episodes = val

  # networks
  training_network = Network.Network(net_input_size)
  training_network = training_network.load_model(training_net_path)

  # visualization
  global robot
  robot = main.Robot()
  robot.speed = speed
  robot.instantiate_node()
  robot.publish_action(7)
  rospy.on_shutdown(shutdown)

  current_image, first_index = robot.get_image()
  state = robot.get_state(current_image)
  last_state = state

  # main program
  try:
    i = 0
    while not rospy.is_shutdown():
      # get current image and its' index
      current_image, first_index = robot.get_image()

      if(i <= number_episodes):
        # select action
        state = np.array([state])
        # action_values = training_network.predict(state)
        action_values = training_network.predict(current_image)
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