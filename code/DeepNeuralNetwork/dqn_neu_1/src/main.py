#!/usr/bin/env python

# import own scripts
import Memory
import Network

# import numpy
import numpy as np
from numpy import random

# tensorflow
import numpy as np
import tensorflow as tf

# Import OpenCV libraries and tools
import cv2 as cv
from cv_bridge import CvBridge, CvBridgeError

# Matplotlib
from matplotlib import pyplot as plt
from matplotlib.colors import hsv_to_rgb

# ROS
import rospy
import rospkg
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from gazebo_msgs.srv import GetModelState
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState

# other
import random
import os
import time

class Robot:
  '''---------------------- Constructor   -------------------------'''
  def __init__(self):
    # saving variables
    self.image_path = \
      "/home/elisabeth/catkin_ws/src/DeepNeuralNetwork/dqn_neu_1" \
      "/images/Image"
    self.net_number = 0
    self.training_net_path = \
      "/home/elisabeth/catkin_ws/src/DeepNeuralNetwork/dqn_neu_1" \
      "/Training/Training_Network_" + str(self.net_number) + ".h5"
    self.target_net_path = \
      "/home/elisabeth/catkin_ws/src/DeepNeuralNetwork/dqn_neu_1" \
      "/Target/Target_Network_" + str(self.net_number) + ".h5"

    # hyperparameters
    self.speed = 10.0
    self.epsilon = 1.0
    self.max_episodes = 2000
    self.max_steps_per_episode = 400
    self.memory_size = 10000
    self.batch_size = 100
    self.image_length = 50
    self.images_in_one_input = 2
    self.net_input_size = self.images_in_one_input * self.image_length
    # self.net_input_size = 1
    self.update_rate = 5

    #other
    self.image = np.zeros(shape=[1, 50])
    self.image_buffer = np.zeros(shape=[100, 50])
    self.image_index = 0
    self.image_cnt = 0

    self.episode_counter = 0
    self.start_decaying = self.max_episodes / 5
    self.min_exploration_rate = 0.01
    self.decay_per_episode = self.calc_decay_per_episode()
    self.steps_in_episode = 0

    self.memory = Memory.Memory(self.memory_size)   # replay-buffer
    self.training_network = Network.Network(self.net_input_size)
    self.target_network = Network.Network(self.net_input_size)

    self.img_in_loop = 0

  '''------------------------- ROS Node ---------------------------'''
  def instantiate_node(self):
    # ROS variables
    # publisher to publish on topic /cmd_vel
    self.velocity_publisher = rospy.Publisher('/cmd_vel', Twist,
                                              queue_size=100)
    # initializing ROS-node
    rospy.init_node('reinf_dqn_driving', anonymous=True)
    # subscriber for topic '/camera/image_raw'
    self.sub = rospy.Subscriber('/camera/image_raw', Image,
                                self.cam_im_raw_callback)

  '''---------------------- Image methods  ------------------------'''
  # receive images
  def cam_im_raw_callback(self, msg):
    # convert to cv
    img = self.ros_to_cv_image(msg)
    self.image = np.copy(img)

    # store image in image buffer
    self.image_index = self.image_cnt % len(self.image_buffer)
    self.fill_image_buffer(self.image, self.image_index)

    '''
    # get state
    state = self.get_state(self.image)

    # save image
    title = "Image " + str(self.image_cnt) + ", State = " + str(
      state)
    self.save_image(self.image, title, self.image_cnt)
    '''


    # increment image counter
    self.image_cnt += 1
    self.img_in_loop += 1

  # convert ros image to open cv image
  def ros_to_cv_image(self, ros_img):
    bridge = CvBridge()
    try:
      cv_image = bridge.imgmsg_to_cv2(ros_img, "passthrough")
    except CvBridgeError as e:
      rospy.logerr("CvBridge Error: {0}".format(e))
    return cv_image

  # save image on file system
  def save_image(self, img, title, path_fragment):
    path = self.image_path
    path += "_" + str(path_fragment)
    plt.imshow(img)
    plt.title(title)
    plt.savefig(path)

  # return the last received image and it's image buffer index
  def get_image(self):
    nr_images = self.image_cnt
    while(self.image_cnt <= nr_images):
      pass
    return self.image, self.image_index

  # get consecutive images
  def get_multiple_images(self, first_index, second_index):
    if(first_index <= second_index):
      stack = self.image_buffer[first_index:second_index+1]
      print("Number of images = " + str(len(stack)))
    else:
      stack = self.image_buffer[0:second_index+1]
      print("Stack full")
      print("Number of images = " + str(len(stack)))
    return stack

  # get as many images as needed (flattened)
  def get_correct_nr_of_images(self, stack):
    arr = np.zeros(shape=[self.images_in_one_input,
                          self.image_length])
    if (len(stack) >= self.images_in_one_input):
      arr = stack[0:self.images_in_one_input]
    else:
      diff = self.images_in_one_input - len(stack)
      for i in range(diff):
        arr[i] = self.image_buffer[-1]
    # shape correctly
    arr = arr.flatten()
    arr_2 = np.zeros(shape=[1, len(arr)])
    arr_2[0] = arr
    return arr_2

  # buffer received images to be able to get them later on
  def fill_image_buffer(self, img, index):
    self.image_buffer[index] = img[0]
    if(index == 0):
      self.image_buffer[1:len(self.image_buffer)].fill(0.0)

  '''---------------------- State methods -------------------------'''
  # count background pixel
  def count_pixel(self, img):
    cnt = 0
    for i in range(len(img[0])):
      if (img[0, i] > 50):
        cnt += 1
      else:
        break
    return cnt

  # calculate state (how close to the middle of the line is the robot)
  def get_state(self, img):
    left = self.count_pixel(img)
    right = self.count_pixel(np.flip(img, 1))
    width = np.size(img[0])
    abs_right = (width-right)
    middle = float(left + abs_right) / 2.0

    if (left >= (width * (99.0 / 100.0)) or right >= (
      width * (99.0 / 100.0))):
      # line is lost
      # just define that if line is ALMOST lost, it is completely
      # lost, so terminal state gets reached
      state = 7
    elif (middle >= (width * (0.0 / 100.0)) and middle <= (
      width * (2.5 / 100.0))):
      # line is far left
      state = 0
    elif (middle > (width * (2.5 / 100.0)) and middle <= (
      width * (21.5 / 100.0))):
      # line is left
      state = 1
    elif (middle > (width * (21.5 / 100.0)) and middle <= (
      width * (40.5 / 100.0))):
      # line is slightly left
      state = 2
    elif (middle > (width * (40.5 / 100.0)) and middle <= (
      width * (59.5 / 100.0))):
      # line is in the middle
      state = 3
    elif (middle > (width * (59.5 / 100.0)) and middle <= (
      width * (78.5 / 100.0))):
      # line is slightly right
      state = 4
    elif (middle > (width * (78.5 / 100.0)) and middle <= (
      width * (97.5 / 100.0))):
      # line is right
      state = 5
    elif (middle * (97.5 / 100.0)) and middle <= (
      width * (100.0 / 100.0)):
      # line is far right
      state = 6
    else:
      # line is lost
      state = 7

    return state

  '''------------------------- Reward -----------------------------'''
  def get_reward(self, state):
    if (state == 0):
      # far left
      return -0.8
    elif(state == 1):
      # left
      return -0.2
    elif(state == 2):
      # slightly left
      return 0
    elif(state == 3):
      # middle
      return 1
    elif(state == 4):
      # slightly right
      return 0
    elif(state == 5):
      # right
      return -0.2
    elif(state == 6):
      # far right
      return -0.8
    else:
      # lost
      return -1

  '''---------------------- Action methods ------------------------'''
  # publish given action on /cmd_vel topic
  def publish_action(self, action):
    # deviation from speed to turn the robot to the left or right
    # sharp curve => big difference
    sharp = self.speed * (1.0 / 7.0)
    # middle curve => middle difference
    middle = self.speed * (1.0 / 8.5)
    # slight curve => slight difference
    slightly = self.speed * (1.0 / 10.0)

    vel = Twist()
    vel.linear.z  = 0
    vel.angular.x = 0
    vel.angular.y = 0
    vel.angular.z = 0

    if (action == 0):
      vel.linear.x = self.speed + sharp
      vel.linear.y = self.speed - sharp
      #print("Sharp left")
    elif(action == 1):
      vel.linear.x = self.speed + middle
      vel.linear.y = self.speed - middle
      #print("Left")
    elif(action == 2):
      vel.linear.x = self.speed + slightly
      vel.linear.y = self.speed - slightly
      #print("Slightly left")
    elif(action == 3):
      vel.linear.x = self.speed
      vel.linear.y = self.speed
      #print("Forward")
    elif(action == 4):
      vel.linear.x = self.speed - slightly
      vel.linear.y = self.speed + slightly
      #print("Slightly right")
    elif(action == 5):
      vel.linear.x = self.speed - middle
      vel.linear.y = self.speed + middle
      #print("Right")
    elif(action == 6):
      vel.linear.x = self.speed - sharp
      vel.linear.y = self.speed + sharp
      #print("Sharp right")
    else:
      vel.linear.x = 0
      vel.linear.y = 0
      #print("Stop")

    self.velocity_publisher.publish(vel)

  # publish stopping action
  def stop(self):
    self.publish_action(7)

  '''------------- Exploration-Exploitation Trade-Off -------------'''
  # take random action when exploring
  # take best action else
  def epsilon_greedy(self, state):
    rand = random.uniform(0, 1)
    if(rand <= self.epsilon):
      # print("Exploring")
      action_arr = np.random.randint(low=0, high=7, size=10)
      np.random.shuffle(action_arr)
      action = action_arr[0]
      return action, rand
    else:
      # state_arr = np.array([state])
      action_values = self.training_network.predict_q_values(
        state)
      action = np.argmax(action_values)
      return action, rand

  # decay exploring probability
  def decay_epsilon(self):
    if(self.episode_counter > self.start_decaying):
      self.epsilon -= self.decay_per_episode

  # calculate the value that has to be subtracted from the
  # exploration probability each episode
  def calc_decay_per_episode(self):
    total_decay_episodes = self.max_episodes - self.start_decaying
    total_decay_steps = self.epsilon - self.min_exploration_rate
    decay_per_episode = float(total_decay_steps) / \
                        float(total_decay_episodes)
    return decay_per_episode

  '''-------------------------- Learning --------------------------'''
  # train the training network
  def learn(self):
    # get training data out of memory
    training_data = self.memory.get_memory_batch(self.batch_size)
    # resulting_states = np.zeros(shape=[self.batch_size,
    # self.images_in_one_input])
    resulting_states = np.zeros(shape=[self.batch_size,
                                       self.images_in_one_input *
                                       self.image_length])
    last_states = np.zeros(shape=[self.batch_size,
                                  self.images_in_one_input *
                                  self.image_length])
    actions = np.zeros(shape=[self.batch_size, 1])
    rewards = np.zeros(shape=[self.batch_size, 1])
    done_array = np.zeros(shape=[self.batch_size, 1])
    for i in range(len(training_data)):
      #resulting_states[i, 0] = training_data[i].get(
      # "resulting_state")
      resulting_states[i] = training_data[i].get("resulting_state")
      last_states[i] = training_data[i].get("last_state")
      actions[i] = training_data[i].get("action")
      rewards[i] = training_data[i].get("reward")
      done_array[i] = training_data[i].get("done")

    print("Input shape = " + str(last_states.shape))
    # print("Last states = \n" + str(last_states))

    # predict current q values
    current_q_values = self.training_network.predict_q_values(
      last_states)
    # predict next q values
    next_q_values = self.target_network.predict_q_values(
      resulting_states)
    # calculate the "real" q values
    expected_q_values = self.bellman(current_q_values,
                                     next_q_values, actions,
                                     rewards, done_array)
    # calculate the loss
    loss = self.training_network.update_weights(state=last_states, \
                                          targets=expected_q_values,
                                          batch_size=self.batch_size)
    return loss

  # bellman equation for double dqn
  def bellman(self, curr_Q, next_Q, action, reward, done):
    expected_Q = np.copy(curr_Q)
    for i in range(len(curr_Q)):
      max_Q = np.max(next_Q[i])
      index = int(action[i])
      if(done[i]):
        expected_Q[i, index] = reward[i]
      else:
        expected_Q[i, index] = reward[i] + 0.95 * max_Q
    return expected_Q

  '''---------------------- Episode methods -----------------------'''
  # increase episode counter and set robot back to starting position
  def end_episode(self):
    self.episode_counter += 1
    self.steps_in_episode = 0
    self.set_position()
    # skip next image
    self.get_image()
    # decay epsilon
    robot.decay_epsilon()

  # all print statements
  def print_debug_info(self, rand, last_state,
                       action, state, reward,
                       loss, copied, done):
    if(done):
      print("END OF EPISODE " + str(self.episode_counter))

    print("Episode \t = " + str(self.episode_counter))
    print("Step \t\t = " + str(self.steps_in_episode))
    print("Expl. prob.\t = " + str(self.epsilon))
    print("Random nr. \t = " + str(rand))
    if(rand <= self.epsilon):
      print("Exploring")
    else:
      print("Exploiting")
    print("Last state \t = " + str(last_state))
    print("Action \t\t = " + str(action))
    print("State \t\t = " + str(state))
    print("Reward \t\t = " + str(reward))
    print("Loss \t\t = " + str(loss))
    print("Copied \t\t = " + str(copied))
    print("Done \t\t = " + str(done))
    print("-"*60)

  '''--------------------- Position methods -----------------------'''
  # set position of robot
  def set_position(self):
    # straight line from far left going into right curve
    x = -2
    y = 4.56540826549
    z = -0.0298790967155

    '''
    # choose random number between 0 and 1
    rand = random.uniform(0, 1)
    #print("rand = " + str(rand))
    if(rand <= (1.0/5.0)):
      #initial starting position
      x = -3.4032014349
      y = -6.22487658223
      z = -0.0298790967155
      #print("case 0")
    elif (rand > (1.0/5.0) and rand <= (2.0 / 5.0)):
      # straight line (long) going into left curve
      x = -0.9032014349
      y = -6.22487658223
      z = -0.0298790967155
      #print("case 1")
    elif (rand > (2.0 / 5.0) and rand <= (3.0 / 5.0)):
      # sharp left curve
      x = 0.930205421421
      y = -5.77364575559
      z = -0.0301045554742
      #print("case 2")
    elif (rand > (3.0 / 5.0) and rand <= (4.0 / 5.0)):
      # sharp right curve
      x = 1.1291257432
      y = -3.37940826549
      z = -0.0298815752691
      #print("case 3")
    elif:
      # straight line going into right curve
      x = 2.4132014349
      y = 4.56540826549
      z = -0.0298790967155
      #print("case 4")
    else: 
      # straight line from far left going into right curve
      x = -2
      y = 4.56540826549
      z = -0.0298790967155
      '''

    state_msg = ModelState()
    state_msg.model_name = 'three_pi'
    state_msg.pose.position.x = x
    state_msg.pose.position.y = y
    state_msg.pose.position.z = z
    state_msg.pose.orientation.x = 0
    state_msg.pose.orientation.y = 0
    state_msg.pose.orientation.z = 0
    state_msg.pose.orientation.w = 0

    rospy.wait_for_service('/gazebo/set_model_state')
    try:
      set_state = rospy.ServiceProxy('/gazebo/set_model_state',
                                     SetModelState)
      resp = set_state(state_msg)

    except rospy.ServiceException as e:
      print("Service call failed: %s" % e)

  '''---------------------- Ending methods ------------------------'''
  # stop and reset robot
  def shutdown(self):
    print("Shutdown")
    self.stop()
    self.set_position()
    self.training_network.save_model(self.training_net_path)
    self.target_network.save_model(self.target_net_path)

'''-------------------------------------------------------------------
----------------------------main program------------------------------
-------------------------------------------------------------------'''
if __name__ == '__main__':
  robot = Robot()
  robot.instantiate_node()
  robot.publish_action(7)
  rospy.on_shutdown(robot.shutdown)

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

  try:
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

      # save images with title = state to check if correct
      #robot.save_image(last_mult_images, "Last images " + str(
        #robot.episode_counter), robot.image_cnt)

      #robot.save_image(mult_images, "Current images " + str(
        #robot.episode_counter), robot.image_cnt)

      if(robot.episode_counter < robot.max_episodes):
        # select action
        action, rand = robot.epsilon_greedy(
          mult_images)
        # execute action
        robot.publish_action(action)
        # save last state
        last_state = state
        # get resulting state
        state = robot.get_state(current_image)
        # save image with title = state to check if correct
        # robot.save_image(current_image, "State = " + str(state),
                         # robot.image_cnt)
        # get reward
        reward = robot.get_reward(state)
        # check if episode done
        done = False
        if(state == 7 or robot.steps_in_episode >=
          robot.max_steps_per_episode):
          robot.end_episode()
          done = True
        else:
          robot.steps_in_episode += 1
        # store experience in memory
        robot.memory.store_experience(resulting_state=mult_images,
          last_state=last_mult_images, reward=reward, action=action,
          done=done)

        # learn
        loss = robot.learn()

        # update target network if necessary
        copied = False
        if(robot.episode_counter % robot.update_rate == 0):
          if(robot.steps_in_episode == 0):
            robot.target_network = robot.training_network.copy(
              robot.target_network)
            copied = True

        # print debugging info
        robot.print_debug_info(rand, last_state,
                               action, state, reward,
                               loss, copied, done)
      else:
        print("Finished")
        robot.training_network.save_model(robot.training_net_path)
        robot.target_network.save_model(robot.target_net_path)
        break


      print("Images in one loop = " + str(robot.img_in_loop))
      robot.img_in_loop = 0

  except rospy.ROSInterruptException:
    print("Exception")
    robot.stop()
    robot.training_network.save_model(robot.training_net_path)
    robot.target_network.save_model(robot.target_net_path)

'''
print("State before = " + str(state))
robot.publish_action(0)
#time.sleep(3)
next_image, second_index = robot.get_image()
state = robot.get_state(current_image)
print("State after = " + str(state))

first_image = [robot.image_buffer[first_index]]
#print("First image = \n" + str(first_image))
robot.save_image(first_image, "First image", "test_1")

#print("Current image = \n" + str(current_image))
robot.save_image(current_image, "Current image", "test_1_1")

second_image = [robot.image_buffer[second_index]]
robot.save_image(second_image, "Second image", "test_2")
#print("Second image = \n" + str(second_image))

#print("Next image = \n" + str(next_image))
robot.save_image(next_image, "Next image", "test_2_1")
'''