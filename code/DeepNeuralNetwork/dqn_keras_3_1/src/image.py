#!/usr/bin/env python

# import own scripts
import main as rm
import matrix

# import numpy
import numpy as np

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


class MyImage:
  # constructor
  def __init__(self):
    # Initialize the CvBridge class
    self.bridge = CvBridge()

  # Try to convert the ROS Image message to a cv Image
  # returns already segmented image (black and white)
  def img_conversion(self, ros_img):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(ros_img, "passthrough")
    except CvBridgeError as e:
      rospy.logerr("CvBridge Error: {0}".format(e))

    #print("Shape of received image = " + str(np.shape(cv_image)))
    small_img = self.crop(cv_image)
    #print("Shape of cropped image = " + str(np.shape(small_img)))
    #print("")
    #print("Received img = ")
    #print(small_img)
    seg_img = self.segmentation(small_img)
    #print("Segmented img = ")
    #print(seg_img)

    return seg_img

  # take middle row of image
  def crop(self, img):
    height = (len(img))
    return img[(height/2):((height/2)+1)]

  ########################################### QUELLE ###################################################
  ''' https://realpython.com/python-opencv-color-spaces/ oder 3-openCV_img_segmentation.pdf '''

  # takes now a GRAYSCALE image of shape (1, 50)
  # divides picture into two segments: floor and line
  # sets the background (floor) pixels to black (0)
  # and the line pixels to white (255)
  def segmentation(self, img):
    # set color range
    light_black = (0)
    dark_black = (10)

    # black and white image (2D Array): 255 (>50)
    # => NOT in color range, 0 to 50 => IN color range
    # line in received image is black = (0)
    # background in received image is grey = (74)
    # >10 will be set to 0 (black)
    # => background will be set to black.. sorry
    # 0 to 10 will be set to 1 -> 255 (white)
    # => line will be set to white.. sorry again
    mask = cv.inRange(img, light_black, dark_black)
    return mask
    ##############################################################################################

  # counts the amount of black pixels (background) of an image
  # starting from the upper left corner
  # to the upper right corner
  # until the first white pixel (beginning of line) is found
  # first started counting the first ten rows -- changed to just one
  # row, hence the for-loop dummy
  def count_pxl(self, img):
    result = 0

    # for-loop dummy
    for i in range(
      1):  # go from row 0 to 1 in steps of 1 (= the first row)
      k = 0
      j = img[i, k]
      # print("J = " + str(j))
      while j < 251:  # as long as current pixel is black (is background)
        result += 1
        k += 1
        if (k < len(img[i])):  # check if it's still in bounds
          j = img[i, k]  # jump to next pixel
        else:
          break

    return result

  # where is middle of the line
  # state is dependent on middle of the line
  def get_line_state(self, img):
    # get left edge of line
    left = self.count_pxl(img)
    # flip image vertically (pixel on the right will be on the left,
    # pixel on top stays on top)
    reversed_img = np.flip(img, 1)
    # get right edge of line (start counting from the right)
    right = self.count_pxl(reversed_img)

    # get width of image (should be 50)
    width = np.size(img[0])

    # get right edge of line (start counting from the left)
    absolute_right = width - right
    # middle is between left and right edge
    middle = float(left + absolute_right) / 2.0

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
