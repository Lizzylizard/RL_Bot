#!/usr/bin/env python

# import own scripts
import matrix as bt
import image as mi
import Memory
import Network
import sound

# import numpy
import numpy as np
from numpy import random

# tensorflow
import numpy as np
import tensorflow as tf
from numpy_ringbuffer import RingBuffer

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


class Node:
  '''-------------------------Constructor--------------------------'''
  def __init__(self):
    '''--------------Adjust before running program-----------------'''
    # increment if you wish to save a new version of the network model
    # or set to specific model version if you wish to use an existing
    # model
    self.path_nr = 0
    # set to False if you wish to run program with existing model
    self.learn = True

    '''---------------------Hyperparameters------------------------'''
    # hyperparameters to experiment with
    # number of learning episodes
    # self.max_episodes = 475
    self.max_episodes = 1000
    self.max_steps_per_episode = 400
    # speed of the robot's wheels
    self.speed = 12.0
    # replay buffer capacity
    self.rb_capacity = 10000
    # number of examples that will be extracted at once from
    # the memory
    self.batch_size = 100
    # number of memory samples that will be processed together in
    # one execution of the neural network
    self.mini_batch_size = 1
    # variables for Bellman equation
    self.gamma = 0.95
    self.alpha = 0.95
    # update rate for target network
    self.update_r_targets = 5
    # integer variable after how many episodes exploiting is possible
    self.start_decaying = (self.max_episodes / 5)
    # self.start_decaying = 0

    '''------------------------Class objects-----------------------'''
    # helper classes
    self.bot = bt.Bot()  # reward + q-matrix
    self.imgHelper = mi.MyImage()  # image processing
    self.memory = Memory.Memory(self.rb_capacity) # replay buffer

    '''--------------------------Images----------------------------'''
    # current image
    self.my_img = np.zeros(50)
    # image stack
    self.image_stack = np.zeros(shape=[self.mini_batch_size, 50])
    self.image_stack_help = RingBuffer(self.mini_batch_size,
                                       dtype=(np.uint8, (1, 50)))
    # flattened current multiple images
    self.my_mult_img = np.zeros(shape=[1, self.mini_batch_size*50])
    # last images
    self.last_imgs = np.copy(self.my_mult_img)
    # number of images received in total
    self.img_cnt = 0

    '''-------------------------Episodes---------------------------'''
    # episodes
    self.explorationMode = False  # exploring or exploiting?
    self.episode_counter = 0  # number of done episodes
    self.step_counter = 0  # counts every while iteration
    self.steps_in_episode = 0  # counts steps inside current episode
    # variables deciding whether to explore or to exploit
    # old --> these are currently used
    self.exploration_prob = 0.99
    # self.exploration_prob = 0.8
    self.decay_rate = 0.01
    self.min_exploration_rate = 0.01
    self.max_exploration_rate = 1
    self.decay_per_episode = self.calc_decay_per_episode()
    # new
    self.epsilon = 1
    self.epsilon_min = 0.001
    self.epsilon_decay = 0.999

    '''-------------------------Strings----------------------------'''
    # strings to display actions
    self.action_strings = {
      0: "sharp left",
      1: "left",
      2: "slightly left",
      3: "forward",
      4: "slightly right",
      5: "right",
      6: "sharp right",
      7: "stop"
    }

    # strings to display states
    self.state_strings = {
      0: "far left",
      1: "left",
      2: "slightly left",
      3: "middle",
      4: "slightly right",
      5: "right",
      6: "far right",
      7: "lost"
    }

    '''----------------------Neural network------------------------'''
    # neural network
    # tensorflow session object
    self.sess = tf.compat.v1.Session()
    # policy network
    input_shape = np.shape(self.my_mult_img)
    self.policy_net = Network.Network(mini_batch_size=self.mini_batch_size)
    # target array (for simple way)
    self.targets = np.zeros(shape=[1, (len(self.action_strings) - 1)])
    # target network to calculate 'optimal' q-values
    self.target_net = Network.Network(mini_batch_size=self.mini_batch_size)
    # copy weights and layers from the policy net into the target net
    self.target_net = self.policy_net.copy(self.target_net)

    '''--------------------------Driving---------------------------'''
    # terminal states
    self.lost_line = 7
    self.stop_action = 7

    # current state and action (initially negative)
    self.curr_state = -1
    self.curr_action = -1
    self.last_state = self.curr_state

    # starting coordinates of the robot
    self.x_position = -0.9032014349
    self.y_position = -6.22487658223
    self.z_position = -0.0298790967155

    # deviation from speed to turn the robot to the left or right
    # sharp curve => big difference
    self.sharp = self.speed * (1.0 / 7.0)
    # middle curve => middle difference
    self.middle = self.speed * (1.0 / 8.5)
    # slight curve => slight difference
    self.slightly = self.speed * (1.0 / 10.0)

    # flag to indicate end of learning
    self.first_time = True

    '''----------------------------ROS-----------------------------'''
    # message to post on topic /cmd_vel
    self.vel_msg = Twist()

    # ROS variables
    # publisher to publish on topic /cmd_vel
    self.velocity_publisher = rospy.Publisher('/cmd_vel', Twist,
                                              queue_size=100)
    # initializing ROS-node
    rospy.init_node('reinf_matrix_driving', anonymous=True)
    # subscriber for topic '/camera/image_raw'
    self.sub = rospy.Subscriber('/camera/image_raw', Image,
                                self.cam_im_raw_callback)

    '''------------------------statistics--------------------------'''
    self.step_array = np.zeros(shape=[self.max_episodes+1])
    self.actions = 0
    self.pics_since_action = 0

  '''-----------------------Image methods--------------------------'''
  # callback; copies the received image into a global numpy-array
  def cam_im_raw_callback(self, msg):
    # convert ROS image to cv image, copy it and save it as a global
    # numpy-array
    img = self.imgHelper.img_conversion(msg)
    self.my_img = np.copy(img)

    # put in current image stack
    index = self.img_cnt % self.mini_batch_size
    self.image_stack[index] = self.my_img

    # print("Image number " + str(self.img_cnt))
    # print("Image index " + str(index))
    self.img_cnt += 1
    self.pics_since_action += 1

  # receive ONE new image
  def get_image(self):
    # wait for the next image
    nr_images = self.img_cnt
    while (self.img_cnt <= nr_images):
      pass
    # save current image
    ret = self.my_img

    '''
    #plot image
    plt.imshow(ret)
    plt.title("Original")
    plt.show()
    '''

    # return current image
    return ret

  # fill image stack with consecutive images --> not used at the moment
  # number images = mini batch size
  def get_consecutive_images(self):
    for i in range(self.mini_batch_size):
      self.image_stack[i] = self.get_image()
    first_img = np.array([self.image_stack[0]])
    sec_img = np.array([self.image_stack[1]])
    last_img = np.array([self.image_stack[-1]])
    self.my_mult_img[0] = self.image_stack.flatten()

    '''
    # plot image
    plt.imshow(first_img)
    plt.title("First image")
    plt.show()

    plt.imshow(sec_img)
    plt.title("Second image")
    plt.show()
    '''

    return self.my_mult_img, last_img

  # receive multiple images
  def get_multiple_images(self):
    # wait for the next image
    self.get_image()
    # get current version of image stack
    images = self.image_stack
    if(len(images) > 1):
      # flatten image stack
      self.my_mult_img[0] = images.flatten()
    # return current images
    return self.my_mult_img

  # get multiple images for network
  # get single image for state
  def shape_images(self):
    # get multiple images for network
    my_imgs = self.get_multiple_images()
    # my_imgs = self.get_consecutive_images()
    # get single image to calculate state
    my_img = np.zeros(shape=[1, 50])
    my_img[0] = my_imgs[:, (
      self.mini_batch_size-1)*50:self.mini_batch_size*50]

    '''
    # plot image
    plt.imshow(my_img)
    plt.title("Single image")
    plt.show()
    plt.imshow(my_imgs[:, (
      self.mini_batch_size-1)*50:self.mini_batch_size*50]) 
    plt.title("Last of multiple images")
    plt.show()
    '''

    return my_imgs, my_img

  # skip the amount of given frames --> not used at the moment
  def skip_images(self, nr):
    for i in range(nr):
      self.get_image()

  '''-----------------------Driving methods------------------------'''
  # sets fields of Twist variable so robot drives sharp left
  def sharp_left(self):
    vel_msg = Twist()
    vel_msg.linear.x = self.speed + self.sharp
    vel_msg.linear.y = self.speed - self.sharp
    vel_msg.linear.z = 0
    vel_msg.angular.x = 0
    vel_msg.angular.y = 0
    vel_msg.angular.z = 0
    # print("SHARP LEFT")
    return vel_msg

  # sets fields of Twist variable so robot drives slightly left
  def slightly_left(self):
    vel_msg = Twist()
    vel_msg.linear.x = self.speed + self.slightly
    vel_msg.linear.y = self.speed - self.slightly
    vel_msg.linear.z = 0
    vel_msg.angular.x = 0
    vel_msg.angular.y = 0
    vel_msg.angular.z = 0
    # print("SLIGHTLY LEFT")
    return vel_msg

  # sets fields of Twist variable so robot drives left
  def left(self):
    vel_msg = Twist()
    vel_msg.linear.x = self.speed + self.middle
    vel_msg.linear.y = self.speed - self.middle
    vel_msg.linear.z = 0
    vel_msg.angular.x = 0
    vel_msg.angular.y = 0
    vel_msg.angular.z = 0
    # print("LEFT")
    return vel_msg

  # sets fields of Twist variable so robot drives forward
  def forward(self):
    vel_msg = Twist()
    vel_msg.linear.x = self.speed
    vel_msg.linear.y = self.speed
    vel_msg.linear.z = 0
    vel_msg.angular.x = 0
    vel_msg.angular.y = 0
    vel_msg.angular.z = 0
    # print("FORWARD")
    return vel_msg

  # sets fields of Twist variable so robot drives slightly right
  def slightly_right(self):
    vel_msg = Twist()
    vel_msg.linear.x = self.speed - self.slightly
    vel_msg.linear.y = self.speed + self.slightly
    vel_msg.linear.z = 0
    vel_msg.angular.x = 0
    vel_msg.angular.y = 0
    vel_msg.angular.z = 0
    # print("SLIGHTLY RIGHT")
    return vel_msg

  # sets fields of Twist variable so robot drives right
  def right(self):
    vel_msg = Twist()
    vel_msg.linear.x = self.speed - self.middle
    vel_msg.linear.y = self.speed + self.middle
    vel_msg.linear.z = 0
    vel_msg.angular.x = 0
    vel_msg.angular.y = 0
    vel_msg.angular.z = 0
    # print("RIGHT")
    return vel_msg

  # sets fields of Twist variable so robot drives sharp right
  def sharp_right(self):
    vel_msg = Twist()
    vel_msg.linear.x = self.speed - self.sharp
    vel_msg.linear.y = self.speed + self.sharp
    vel_msg.linear.z = 0
    vel_msg.angular.x = 0
    vel_msg.angular.y = 0
    vel_msg.angular.z = 0
    # print("SHARP RIGHT")
    return vel_msg

  # sets fields of Twist variable to stop robot and puts the robot
  # back to starting position
  def stop(self):
    print("Stop")
    vel_msg = Twist()
    vel_msg.linear.x = 0.0
    vel_msg.linear.y = 0.0
    vel_msg.linear.z = 0
    vel_msg.angular.x = 0
    vel_msg.angular.y = 0
    vel_msg.angular.z = 0
    return vel_msg

  '''----------------------Position methods------------------------'''
  '''
  https://answers.gazebosim.org//question/18372/getting-model-state-via-rospy/
  '''
  def set_position(self, x, y, z):
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

  # choose one of five given positions randomly
  def select_starting_pos(self):
    # straight line from far left going into right curve
    self.x_position = -2
    self.y_position = 4.56540826549
    self.z_position = -0.0298790967155

    '''
    # choose random number between 0 and 1
    rand = random.uniform(0, 1)
    #print("rand = " + str(rand))
    if(rand <= (1.0/5.0)):
      #initial starting position
      self.x_position = -3.4032014349
      self.y_position = -6.22487658223
      self.z_position = -0.0298790967155
      #print("case 0")
    elif (rand > (1.0/5.0) and rand <= (2.0 / 5.0)):
      # straight line (long) going into left curve
      self.x_position = -0.9032014349
      self.y_position = -6.22487658223
      self.z_position = -0.0298790967155
      #print("case 1")
    elif (rand > (2.0 / 5.0) and rand <= (3.0 / 5.0)):
      # sharp left curve
      self.x_position = 0.930205421421
      self.y_position = -5.77364575559
      self.z_position = -0.0301045554742
      #print("case 2")
    elif (rand > (3.0 / 5.0) and rand <= (4.0 / 5.0)):
      # sharp right curve
      self.x_position = 1.1291257432
      self.y_position = -3.37940826549
      self.z_position = -0.0298815752691
      #print("case 3")
    elif:
      # straight line going into right curve
      self.x_position = 2.4132014349
      self.y_position = 4.56540826549
      self.z_position = -0.0298790967155
      #print("case 4")
    else: 
      # straight line from far left going into right curve
      self.x_position = -2
      self.y_position = 4.56540826549
      self.z_position = -0.0298790967155
      '''

  '''
  https://answers.ros.org/question/261782/how-to-use-getmodelstate-service-from-gazebo-in-python/
  '''
  def get_position(self):
    model_coordinates = rospy.ServiceProxy('/gazebo/get_model_state',
                                           GetModelState)
    object_coordinates = model_coordinates("three_pi", "")
    x_position = object_coordinates.pose.position.x
    y_position = object_coordinates.pose.position.y
    z_position = object_coordinates.pose.position.z
    return x_position, y_position, z_position

  # puts robot back to starting position
  def reset_environment(self):
    # select position
    self.select_starting_pos()

    # set robot to it
    self.set_position(self.x_position, self.y_position,
                      self.z_position)

  '''-----------------------Network methods------------------------'''
  # bellman equation for double q network
  def bellman(self, curr_Q, next_Q, action, reward, done):
    expected_Q = np.copy(curr_Q)
    for i in range(len(curr_Q)):
      max_Q = np.max(next_Q[i])
      index = int(action[i])
      # like here:
      '''https://pylessons.com/CartPole-reinforcement-learning/'''
      if(done[i]):
        expected_Q[i, index] = reward[i]
      else:
        expected_Q[i, index] = reward[i] + self.gamma * max_Q
      # in q-matrix it was:
      #qstar(s, a) = Rt+1 + (1-alpha)*q(s, a) * gamma*maxqstar(s', a')
      # expected_Q[i, index] = reward[i] + (1 - self.alpha) * \
                    # curr_Q[i, index] * self.gamma * max_Q
    return expected_Q

  # get random (batch of) experiences and learn with it
  # use target network for optimal q-values
  def replay_memory(self):
    # get random batch of memories
    memory_batch = self.memory.get_random_experience(
      batch_size=self.batch_size)

    mem_states = np.zeros(shape=[len(memory_batch),
                                 self.mini_batch_size * 50])
    mem_last_states = np.zeros(shape=[len(memory_batch),
                                      self.mini_batch_size * 50])
    mem_actions = np.zeros(shape=[len(memory_batch)])
    mem_rewards = np.zeros(shape=[len(memory_batch)])
    mem_dones = np.zeros(shape=[len(memory_batch)])

    for i in range(len(memory_batch)):
      mem_states[i] = memory_batch[i].get("state")
      mem_last_states[i] = memory_batch[i].get("last_state")
      mem_actions[i] = memory_batch[i].get("action")
      mem_rewards[i] = memory_batch[i].get("reward")
      mem_dones[i] = memory_batch[i].get("done")

    print("memory size = " + str(np.shape(mem_last_states)))

    # compute actual q-values with the 'last' states of the
    # memory
    curr_Q = self.policy_net.use_network(state=mem_last_states)
    # compute next q-values with 'current' states of the memory
    next_Q = self.target_net.use_network(state=mem_states)
    # use Bellman equation to compute target q-values
    expected_Qs = self.bellman(curr_Q, next_Q, mem_actions,
                                    mem_rewards, mem_dones)

    loss = self.policy_net.update_weights(state=mem_last_states, \
                                          targets=expected_Qs,
                                          batch_size=self.batch_size)

    #print("curr_Qs =\n\t" + str(curr_Q))
    #print("expected_Qs =\n\t" + str(expected_Qs))
    print("last loss = " + str(loss))

  # save neural networks
  def save_model(self):
    path = os.path.dirname(os.path.realpath(__file__))
    # print("PATH IS = " + str(path))
    online_path = (path + "/online")
    # os.mkdir(online_path)
    online_file_path = (online_path + "/online_net" + str(
      self.path_nr) + ".h5")
    self.policy_net.model.save(online_file_path)

    target_path = (path + "/target")
    target_file_path = (target_path + "/target_net" + str(
      self.path_nr) + ".h5")
    self.target_net.model.save(target_file_path)

  # load already existing model
  def load_model(self):
    path = "/home/elisabeth/catkin_ws/src/DeepNeuralNetwork" \
           "/dqn_keras_3_1/src/online/online_net0.h5"

    target_file_path = \
      "/home/elisabeth/catkin_ws/src/DeepNeuralNetwork/dqn_keras_3_1/src/target/target_net0.h5"
    self.policy_net = tf.keras.models.load_model(path)
    self.target_net = tf.keras.models.load_model(target_file_path)

  '''--------------------Robot process methods---------------------'''
  # get current state and reward
  def get_robot_state(self, my_img):
    # safe last state
    self.last_state = self.curr_state
    print("Last state: " + str(self.state_strings.get(self.last_state)))
    # get new / current state
    # self.curr_state = self.bot.get_state(my_img)
    self.curr_state = self.imgHelper.get_line_state(my_img)
    print("Current state: " + str(self.state_strings.get(
      self.curr_state)))

    # calculate reward
    reward = self.bot.calculate_reward(self.last_state,
                                       self.curr_action)
    print("Reward: " + str(reward))

    return self.curr_state, reward

  # stop robot, reset environment and increase episode counter
  def begin_new_episode(self):
    self.step_array[self.episode_counter] = self.steps_in_episode
    # decay the probability of exploring
    self.decay_epsilon()
    # stop robot
    self.stopRobot()
    # set robot back to starting position
    self.reset_environment()
    print("NEW EPISODE: ", self.episode_counter)
    # skip the next image
    self.get_image()

  # send the ROS message
  def execute_action(self, action):
    # execute action
    vel = Twist()
    directions = {
      0: self.sharp_left,
      1: self.left,
      2: self.slightly_left,
      3: self.forward,
      4: self.slightly_right,
      5: self.right,
      6: self.sharp_right,
      7: self.stop
    }
    function = directions.get(action)
    vel = function()
    self.velocity_publisher.publish(vel)

  '''---------------Exploration exploitation trade off----------------'''
  # exponentially decay epsilon
  def decay_epsilon(self):
    '''
    # new
    if (self.episode_counter > self.start_decaying):
      if self.epsilon > self.epsilon_min:
        self.epsilon *= self.epsilon_decay
    self.exploration_prob = self.epsilon
    '''
    '''
    # deep lizard
    # old
    if(self.episode_counter > self.start_decaying):
      self.exploration_prob = self.min_exploration_rate + \
        (self.max_exploration_rate - self.min_exploration_rate) * \
        np.exp(-self.decay_rate * self.episode_counter)
    '''
    # own implementation
    if(self.episode_counter > self.start_decaying):
      self.exploration_prob -= self.decay_per_episode

  # calculate the value that has to be subtracted from the
  # exploration probability each episode
  def calc_decay_per_episode(self):
    total_decay_episodes = self.max_episodes - self.start_decaying
    total_decay_steps = self.exploration_prob - self.min_exploration_rate
    decay_per_episode = float(total_decay_steps) / \
                        float(total_decay_episodes)
    return decay_per_episode

  # decide whether to explore or to exploit
  def epsilon_greedy(self, e):
    # random number
    exploration_rate_threshold = random.uniform(0, 1)
    print("Exploration prob = " + str(e))
    print("threshold = " + str(exploration_rate_threshold))

    if (exploration_rate_threshold < e):
      # explore
      self.explorationMode = True
      return True
    else:
      # exploit
      self.explorationMode = False
      return False

  # select and execute action
  def get_next_action(self, my_imgs):
    # if exploring: choose random action
    if (self.epsilon_greedy(self.exploration_prob)):
      print("Exploring")
      # take random action
      action_arr = np.random.randint(low=0, high=7, size=10)
      np.random.shuffle(action_arr)
      action = action_arr[0]
      print("random action is = " + str(action))
      print("Action: " + self.action_strings.get(action))
      self.execute_action(action)
      self.curr_action = action
    # if exploiting: choose best action
    else:
      print("Exploiting")
      # get q-values by feeding images to the DQN
      # without updating its weights
      self.q_values = self.policy_net.use_network(
        state=my_imgs)
      # choose action by selecting highest q -value
      action = np.argmax(self.q_values)
      print("Action: " + self.action_strings.get(action))
      self.execute_action(action)
      self.curr_action = action
    self.actions += 1
    self.pics_since_action = 0
    return self.curr_action

  '''------------------------Ending methods------------------------'''
  # publishes stopping message
  def stopRobot(self):
    vel = self.stop()
    self.velocity_publisher.publish(vel)

  # if user pressed ctrl+c --> stop the robot
  def shutdown(self):
    print("Stopping")
    # publish
    self.vel_msg = self.stop()
    self.velocity_publisher.publish(self.vel_msg)
    # calculate average steps per episode
    self.calc_average_steps()
    # save neural networks
    if(self.learn):
      self.save_model()

  '''--------------------Drive without learning--------------------'''
  # use trained network to drive
  def drive(self):
    # if in first iteration after learning is finished
    if(self.first_time):
      # calculate average steps per episode
      self.calc_average_steps()
      # make sound to signal that learning is finished
      sound.make_sound()
      # save neural network
      self.save_model()
      # indicate end of first iteration after learning is finished
      self.first_time = False

    # wait for new image(s)
    my_imgs, my_img = self.shape_images()

    # observe next state and reward
    self.curr_state, reward = self.get_robot_state(my_img)

    # stop robot
    stop_arr=[self.lost_line]
    if(self.curr_state in stop_arr):
      self.stopRobot()

    # get q-values by feeding images to the DQN
    # without updating its weights
    self.q_values = self.policy_net.use_network(state=my_imgs)
    # choose action by selecting highest q -value
    action = np.argmax(self.q_values)
    print("Action: " + self.action_strings.get(action))

    # execute action
    self.execute_action(action)
    # stop if line is lost (hopefully never, if
    # robot learned properly)
    if (self.curr_state == self.lost_line):
      self.reset_environment()

  def calc_average_steps(self):
    sum = np.sum(self.step_array)
    average = float(sum) / float(self.max_episodes)
    print("Average steps per episode = " + str(average))

  # use pre-existing trained network to drive
  def test(self):
    # if in first iteration after learning is finished
    if(self.first_time):
      # make robot drive faster
      # self.speed *= 1.5
      # load existing model
      self.load_model()
      # indicate end of first iteration after learning is finished
      self.first_time = False

    # wait for new image(s)
    my_imgs, my_img = self.shape_images()

    # observe next state and reward
    self.curr_state, reward = self.get_robot_state(my_img)
    # stop robot
    stop_arr=[self.lost_line]
    if(self.curr_state in stop_arr):
      self.stopRobot()

    # get q-values by feeding images to the DQN
    # without updating its weights
    self.q_values = self.policy_net.predict(my_imgs)
    # choose action by selecting highest q -value
    action = np.argmax(self.q_values)

    # execute action
    self.execute_action(action)
    # stop if line is lost (hopefully never, if
    # robot learned properly)
    if (self.curr_state == self.lost_line):
      self.reset_environment()

  '''------------------------Main Program--------------------------'''
  # main program
  def reinf_main(self):
    # tell program what to do on shutdown (user presses ctrl+c)
    rospy.on_shutdown(self.shutdown)

    # set starting position of the robot
    self.reset_environment()

    # starting images
    # my_imgs, my_img = self.shape_images()
    my_imgs, my_img = self.shape_images()

    # main loop
    try:
      while not rospy.is_shutdown():
        # drive with existing network
        # test network input
        print("Testing!")
        self.test()

    except rospy.ROSInterruptException:
      pass

# start node
if __name__ == '__main__':
  node = Node()
  node.reinf_main()
