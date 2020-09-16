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
      "/home/elisabeth/catkin_ws/src/DeepNeuralNetwork/dqn_neu_2" \
      "/images/Image"
    self.net_number = 1
    self.training_net_path = \
      "/home/elisabeth/catkin_ws/src/DeepNeuralNetwork/dqn_neu_2" \
      "/Training/Training_Network_" + str(self.net_number) + ".h5"
    self.target_net_path = \
      "/home/elisabeth/catkin_ws/src/DeepNeuralNetwork/dqn_neu_2" \
      "/Target/Target_Network_" + str(self.net_number) + ".h5"

    # hyperparameters
    self.speed = 15.0
    self.epsilon = 1.0
    self.max_episodes = 1000
    self.max_steps_per_episode = 400
    self.memory_size = 10000
    self.batch_size = 100
    self.image_length = 50
    self.action_repititions = 6
    self.images_in_one_input = 5
    self.net_input_size = self.images_in_one_input * self.image_length
    # self.net_input_size = 1
    self.update_rate = 5

    #other
    self.image = np.zeros(shape=[1, 50])
    self.image_buffer = []
    self.numpy_image_buffer = np.array(self.image_buffer)
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
    self.lost_state = 25.0
    self.use_image = True

    # close to 1 --> long lasting consequences
    # low --> immediate rewards are preferred
    # or should this actually be alpha???!
    self.gamma = 0.95

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

    '''
    # save incoming images on file system
    path = r'/home/elisabeth/catkin_ws/src/DeepNeuralNetwork/dqn_neu_2/test_images'
    os.chdir(path)
    name = "image_" + str(self.image_cnt) + ".png"
    cv.imwrite(name, self.image)
    '''

    # store image in image buffer
    if(self.use_image):
      self.fill_image_buffer(self.image)
    else:
      pass

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

  # return the last received image
  def get_image(self):
    nr_images = self.image_cnt
    while(self.image_cnt <= nr_images):
      pass
    return self.image

  # buffer received images to be able to get them later on
  def fill_image_buffer(self, img):
    self.image_buffer.append(img)
    if(len(self.image_buffer) > self.images_in_one_input):
      self.image_buffer.pop(0)
    self.numpy_image_buffer = np.array(self.image_buffer)

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
      dist = self.lost_state

    state = dist
    return state

  '''------------------------- Reward -----------------------------'''
  def get_reward(self, last_state, state):
    '''
    if (state == 0):
      # far left
      return -0.9
    elif(state == 1):
      # left
      return -0.7
    elif(state == 2):
      # slightly left
      return -0.5
    elif(state == 3):
      # middle
      return 1
    elif(state == 4):
      # slightly right
      return -0.5
    elif(state == 5):
      # right
      return -0.7
    elif(state == 6):
      # far right
      return -0.9
    else:
      # lost
      return -1
    '''
    '''
    if(last_state-last_action == 0):
      reward = 1
    elif(last_state-last_action == -1 or last_state-last_action == 1):
      reward = -0.2
    elif(last_state-last_action == -2 or last_state-last_action == 2):
      reward = -0.6
    else:
      reward = -1
    return reward
    '''
    '''
    reward = 1 - (float(1)/float(3)) * abs(last_state - last_action)
    # 1 - float(1/3) * 0 = 1          => best case (state == action)
    # 1 - float(1/3) * 1 = float(2/3) => (state +-1 = action)
    # 1 - float(1/3) * 2 = float(1/3) => (state +-2 = action)
    # 1 - float(1/3) * 3 = 0          => (state +-3 = action)
    # 1 - float(1/3) * 4 = float(-1/3)=> (state +-4 = action)
    # 1 - float(1/3) * 5 = float(-2/3)=> (state +-5 = action)
    # 1 - float(1/3) * 6 = -1         => (state +-6 = action)
    # 1 - float(1/3) * 7 = float(-4/3)=> worst case (state == 7, lost)
    return reward
    '''
    '''
    abs_state = abs(state)
    if(abs_state == 0):
      reward = 1
    else:
      reward = float(abs_state / self.lost_state) * (-1)
    return reward
    '''
    abs_state = abs(state)
    abs_last_state = abs(last_state)
    if(abs_state == 0):
      reward = 5
    else:
      if(abs_state < abs_last_state):
        reward = 1
      elif(abs_state == abs_last_state):
        reward = 0
      else:
        reward = -1
    return reward

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
    state = np.array([state])
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
    # if (loss < 10):
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
      resulting_states[i] = training_data[i].get(
        "resulting_state").flatten()

      last_states[i] = training_data[i].get("last_state").flatten()

      actions[i] = training_data[i].get("action")
      rewards[i] = training_data[i].get("reward")
      done_array[i] = training_data[i].get("done")

    #print("Input shape = " + str(last_states.shape))
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
        expected_Q[i, index] = reward[i] + self.gamma * max_Q
    return expected_Q

  '''---------------------- Episode methods -----------------------'''
  # increase episode counter and set robot back to starting position
  def end_episode(self):
    self.episode_counter += 1
    self.steps_in_episode = 0
    self.set_position()
    # skip next images
    for i in range(self.action_repititions):
      self.get_image()
    # decay epsilon
    robot.decay_epsilon()

  # translate state number into human language
  def state_to_words(self, state):
    state_str = str(state)
    if(state > 0 and not(abs(state) == self.lost_state)):
      state_str += ", \tleft"
    elif(state < 0 and not(abs(state) == self.lost_state)):
      state_str += ", \tright"
    elif(state == 0):
      state_str += ", \tmiddle"
    else:
      state_str += ", \tlost"
    return state_str

  # translate action number into human language
  def action_to_words(self, action):
    action_str = str(action)
    if(action == 0):
      action_str += ", \t\tgoing far left"
    elif(action == 1):
      action_str += ", \t\tgoing left"
    elif(action == 2):
      action_str += ", \t\tgoing slightly left"
    elif(action == 3):
      action_str += ", \t\tgoing forward"
    elif(action == 4):
      action_str += ", \t\tgoing slightly right"
    elif(action == 5):
      action_str += ", \t\tgoing right"
    elif(action == 6):
      action_str += ", \t\tgoing far right"
    return action_str

  # all print statements
  def print_debug_info(self, rand, last_state,
                       action, state, reward,
                       loss, copied, done):
    print("Episode \t = " + str(self.episode_counter))
    print("Step \t\t = " + str(self.steps_in_episode))
    print("Expl. prob.\t = " + str(self.epsilon))
    print("Random nr. \t = " + str(rand))
    if(rand <= self.epsilon):
      print("Exploring")
    else:
      print("Exploiting")
    print("Last state \t = " + str(self.state_to_words(last_state)))
    print("Action \t\t = " + str(self.action_to_words(action)))
    print("State \t\t = " + str(self.state_to_words(state)))
    print("Reward \t\t = " + str(reward))
    print("Loss \t\t = " + str(loss))
    print("Copied \t\t = " + str(copied))
    print("Done \t\t = " + str(done))
    print("-"*60)
    if(done):
      print("NEW EPISODE " + str(self.episode_counter+1))

  '''--------------------- Position methods -----------------------'''
  # set position of robot
  def set_position(self):
    # straight line from far left going into right curve
    #x = -2
    x = 2
    #y = 4.56540826549
    y = 4.62540826549
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

  # skip until stack is full for the first time
  for i in range(robot.images_in_one_input):
    robot.get_image()

  # get one image
  current_image = robot.numpy_image_buffer[-1]

  # multiple images
  stack = robot.numpy_image_buffer.flatten()
  last_stack = stack.copy()

  # starting state
  state = robot.get_state(current_image)
  last_state = state

  try:
    while not rospy.is_shutdown():
      if(robot.episode_counter < robot.max_episodes):
        # select action
        action, rand = robot.epsilon_greedy(stack)

        # execute action a couple of times to see difference
        for i in range(robot.action_repititions):
          robot.publish_action(action)
          robot.get_image()

        # before executing action, stop movement of robot
        robot.use_image = False
        for i in range(robot.action_repititions):
          robot.stop()
          robot.get_image()
        robot.use_image = True

        # save last image stack
        last_stack = stack.copy()
        # get the last couple of images
        stack = robot.numpy_image_buffer.flatten()

        # get resulting image
        #current_image = robot.get_image()
        current_image = robot.numpy_image_buffer[-1]

        # save last state
        last_state = state

        # get resulting state
        state = robot.get_state(current_image)

        # get reward
        reward = robot.get_reward(last_state, state)

        # check if episode done
        done = False
        if(state == robot.lost_state or robot.steps_in_episode >=
          robot.max_steps_per_episode):
          done = True
        else:
          robot.steps_in_episode += 1

        # store experience in memory
        robot.memory.store_experience(resulting_state=stack,
          last_state=last_stack, reward=reward, action=action,
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

        # end episode if done
        if(done):
          # robot.end_episode(loss[0])
          robot.end_episode()
          # reinitialize state
          current_image = robot.numpy_image_buffer[-1]
          state = robot.get_state(current_image)
          last_state = state
          # do I have to reinitialize the image stack?!

      else:
        print("Finished")
        robot.training_network.save_model(robot.training_net_path)
        robot.target_network.save_model(robot.target_net_path)
        break


      #print("Images in one loop = " + str(robot.img_in_loop))
      robot.img_in_loop = 0

  except rospy.ROSInterruptException:
    print("Exception")
    robot.stop()
    robot.training_network.save_model(robot.training_net_path)
    robot.target_network.save_model(robot.target_net_path)