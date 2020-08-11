#!/usr/bin/env python3

'''
rosrun image_view image_view image:=/camera/image_raw
right click to save image
'''
import math

########################################### QUELLE ###############################################
''' https://dabit-industries.github.io/turtlebot2-tutorials/14b-OpenCV2_Python.html oder 5-ROS_to_OpenCV.pdf '''
#import numpy
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import hsv_to_rgb

# Import OpenCV libraries and tools
import cv2 as cv
from cv_bridge import CvBridge, CvBridgeError

# Initialize the CvBridge class
bridge = CvBridge()

# Define a function to show the image in an OpenCV Window
def show_image(img):
    cv.imshow("Image Window", img)
    cv.waitKey(3)

##################################################################################################
 
########################################### QUELLE ###############################################
''' https://docs.opencv.org/master/da/d22/tutorial_py_canny.html oder 2-openCV_canny_edge_algo.pdf ''' 
#detect the edges in the picture with canny edge detection algorithm
#displays original and edge image
def edge_detection(img):
    #edge detection
    edge_img = cv.Canny(img, 0, 50)    
    
    #show image before edge detection
    plt.subplot(1, 2, 1) 
    plt.imshow(img)
    #plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.show()
    
    #show edge image
    plt.subplot(1, 2, 2)
    plt.imshow(edge_img)
    #plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.show()
    
    return edge_img    
  
##################################################################################################  

########################################### QUELLE ###############################################
''' https://realpython.com/python-opencv-color-spaces/ oder 3-openCV_img_segmentation.pdf '''  
#divides picture into two segments: 0 = floor (grey) 1 = line (black)
#sets the floor pixels to white and the line pixels to black for an easier
#edge detection
def segmentation(img):
    #set color range
    
    light_black = (0, 0, 0)
    dark_black = (25, 25, 25)    
    
    #view the light and dark color
    '''
    lb_square = np.full((10, 10, 3), light_black, dtype=np.uint8) / 255.0
    db_square = np.full((10, 10, 3), dark_black, dtype=np.uint8) / 255.0
    plt.subplot(1, 2, 1)
    plt.imshow(lb_square)
    plt.subplot(1, 2, 2)
    plt.imshow(db_square)
    plt.show() '''
    
    #black and white image (2Darray): 255 => in color range, 0 => NOT in color range
    mask = cv.inRange(img, light_black, dark_black)    
    
    #switch 0s to 255s and 255s to 0s (we want the line black and the floor white, not the other way around)
    #print(mask)
    for i in range(len(mask)):
        for j in range(len(mask[i])):
            if mask[i, j] == 0:
                mask[i, j] = 255
            else:
                mask[i, j] = 0
                
    #print(mask)
    
    #show the mask      
    '''
    plt.subplot(1, 2, 1)
    plt.imshow(mask, cmap="gray")
    plt.show() '''
    
    return mask
##################################################################################################

#removes 30 pixel from the bottom edge of the image 
#'broken' pixel at the edge will be ignored
def crop_image(img):
    #print("Cropped")
    ''' equals 450
    img_width = np.size(img[0])
    img_height = np.size(img)
    new_end = img_width-30 '''
    new_img = img[0:450]
    return new_img


#counts the amount of white pixels of an image 
#starting from the upper left corner 
#to the upper right corner 
#until the first black pixel is found
#started counting the first ten rows -- changed to just one row
def count_pxl(img):
    result = 0
    
    for i in range(1):                 #go from row 0 to 1 in steps of 1 (= the first row)
        k = 0
        j = img[i, k]                   
        while j == 255:                 #as long as current pixel is white (255)
            result += 1
            k += 1
            if(k < len(img[i])):        #check if it's still in bounds
                j = img[i, k]           #jump to next pixel
            else:
                break
            
    return result
    
#counts the amount of white pixel from the upper left corner 
#to the bottom left corner 
#until the first black pixel is found   
def count_pxl_vert(img):
    result = 0
    for i in range(0, len(img), 1):
        if(img[i, 0] == 255):
            result += 1
        elif(img[i, 0] == 0):
            break
            
    #return distance to black pixel
    return result
    
def calc_ratios(img):        
    ############## HORIZONTALLY ###############
    #LEFT TOP
    cnt_left_top = count_pxl(img)    
    #print("Count left top = " + str(cnt_left_top))    
    
    #LEFT BOTTOM
    ''' 
    #test
    arr = np.arange(20).reshape(4, 5)
    print(arr)
    print(np.flip(arr, 0))
    '''
    
    reversed_img = np.flip(img, 0)
    cnt_left_bot = count_pxl(reversed_img)
    
    #do not divide by zero
    if(cnt_left_bot == 0):
        cnt_left_bot = 1
    #print("Count left bottom = " + str(cnt_left_bot)) 
    
    ratio_left = float(cnt_left_top) / float(cnt_left_bot)
    #print("Ratio left = " + str(ratio_left))
    
    #RIGHT TOP  
    '''    
    #test    
    arr = np.arange(20).reshape(4, 5)
    print(arr)
    vert_reversed_imgay = np.flip(arr, 1)
    print(vert_reversed_imgay)  
    '''    
    
    vert_reversed_img = np.flip(img, 1)
    cnt_right_top = count_pxl(vert_reversed_img)
    
    #print("Count right top = " + str(cnt_right_top))    
   
    #RIGHT BOTTOM
    double_reversed_img = np.flip(vert_reversed_img, 0)
    cnt_right_bot = count_pxl(double_reversed_img)
    
    #print("Count right bottom = " + str(cnt_right_bot))
    
    #do not divide by zero
    if(cnt_right_bot == 0):
        cnt_right_bot = 1
        
    ratio_right = float(cnt_right_top) / float(cnt_right_bot)    
    #print("Ratio right = " + str(ratio_right))
    
    ############## VERTICALLY ###############
    cnt_left_vert_top = count_pxl_vert(img)
    #print("\n\nLeft Up Down = " + str(cnt_left_vert_top))
    
    reversed_img = np.flip(img, 0)
    cnt_left_vert_bot = count_pxl_vert(reversed_img)
    #print("Left Down Up = " + str(cnt_left_vert_bot))
    #do not divide by zero
    if(cnt_left_vert_bot == 0):
        cnt_left_vert_bot = 1
    ratio_vert_left = float(cnt_left_vert_top) / float(cnt_left_vert_bot)
    #print("Ratio Vert Left = " + str(ratio_vert_left))
    
    vert_reversed_img = np.flip(img, 1)
    cnt_right_vert_top = count_pxl_vert(vert_reversed_img)
    #print("Right Up Down = " + str(cnt_right_vert_top))
    
    double_reversed_img = np.flip(vert_reversed_img, 0)
    cnt_right_vert_bot = count_pxl_vert(double_reversed_img)
    #print("Right Down Up = " + str(cnt_right_vert_bot))
    #do not divide by zero
    if(cnt_right_vert_bot == 0):
        cnt_right_vert_bot = 1
    ratio_vert_right = float(cnt_right_vert_top) / float(cnt_right_vert_bot)
    #print("Ratio Vert Right = " + str(ratio_vert_right))
    
    return (ratio_left, ratio_right, ratio_vert_left, ratio_vert_right,
    cnt_left_top, cnt_left_bot, cnt_left_vert_top, cnt_left_vert_bot, 
    cnt_right_top, cnt_right_bot, cnt_right_vert_top, cnt_right_vert_bot)

    
def calc_curve(img):
    #calculate (count) white pixel and corresponding ratios
    (ratio_left, ratio_right, ratio_vert_left, ratio_vert_right,
    cnt_left_top, cnt_left_bot, cnt_left_vert_top, cnt_left_vert_bot, 
    cnt_right_top, cnt_right_bot, cnt_right_vert_top, cnt_right_vert_bot) = calc_ratios(img)  
    
    #number of all pixel    
    all_pixels = np.size(img)
    
    #number of all black pixel 
    all_black_pixels = all_pixels - np.count_nonzero(img)

    #number of pixel in one row and one column
    num_row = np.size(img[0])
    num_col = img.shape[0]

    #calculate the angle of the line towards the bottom via tangens
    h = img.shape[0]        #the height of the triangle = the height of the image
    w = np.size(img[0])     #the width of the triangle = the width of the image
    
    d1 = cnt_left_top 
    d2 = cnt_left_bot
    d3 = d2 - d1 
    
    #result
    alpha = 90.0
    curve = "default"
    
    #cases where NO angle needs to be calculated    
    #no line in picture
    if(all_black_pixels == 0):
        curve = "backwards"   
    #line is at upper left corner
    elif(cnt_left_bot == num_row and cnt_left_vert_top < num_col):
        curve = "left"
    #line is at upper right corner 
    elif(cnt_left_bot == num_row and cnt_right_vert_top < num_col):
        curve = "right"    

    #cases where angles DO need to be calculated
    else:
        #line goes through top and bottom 
        if(cnt_left_top < num_row):
            d1 = cnt_left_top 
            d2 = cnt_left_bot
            d3 = d2 - d1
            tanAlpha = [(float(h) / float(d3))]
        #line does NOT go through top but through bottom
        elif(cnt_left_top == num_row and not cnt_left_bot == num_row):
            #line goes through left side of image
            if(cnt_left_vert_top < num_col):        
                d1 = cnt_left_top 
                d2 = cnt_left_bot
                d3 = h - d1
                tanAlpha = [(float(d3) / float(d2))]
            #line goes through right side of image 
            elif(cnt_right_vert_top < num_col):
                d1 = cnt_right_vert_top
                d2 = cnt_right_bot
                d3 = h - d4
                tanAlpha = [(float(d3) / float(d2))]
        #line does NOT got through top AND bottom 
        elif(cnt_left_top == num_row and cnt_left_bot == num_row):
            #line goes through left side of image
            if(cnt_left_vert_top < cnt_right_vert_top):
                d1 = cnt_left_vert_top 
                d2 = cnt_left_vert_bot
                d3 = h - d1 - d2 
                tanAlpha = [(float(d3) / float(w))]
            #line goes through right side of image 
            if(cnt_right_vert_top < cnt_left_vert_top):
                d1 = cnt_right_vert_top 
                d2 = cnt_right_vert_bot
                d3 = h - d1 - d2 
                tanAlpha = [(float(d3) / float(w))]            
    
        #print("h = " + str(h) + ", w = " + str(w) + ", d1 = " + str(d1) + ", d2 = " + str(d2) + ", d3 = " + str(d3))
        #print("Tangens Alpha Array = " + str(tanAlpha))
        alpha = np.arctan(tanAlpha)                 #calculate tangens 
        alpha = math.degrees(alpha)                 #convert to degrees
        print("Alpha Angle = " + str(alpha))
    
        if(alpha > 0 and alpha <= 30):
            curve = "sharp left"
        elif(alpha > 30 and alpha <= 80):
            curve = "left"
        elif(alpha > 80 or alpha <= -80):
            curve = "straight"
        elif(alpha > -80 and alpha <= -30):
            curve = "right"
        elif(alpha > -30 and alpha < 0):
            curve = "sharp right"
    
    '''
    #image width   
    width_of_img = np.size(img[200])
    #print("Image width = " + str(width_of_img))    
    
    #number of pixels in ten rows
    num_pix_10 = width_of_img * 10 
    #print("Ten rows = " + str(num_pix_10)) 
    
    #number of all pixel    
    all_pixels = np.size(img)
    #print("Number all pixels = " + str(all_pixels))
    
    #is there a line?
    all_black_pixels = all_pixels - np.count_nonzero(img)
    #print("Number black pixels = " + str(all_black_pixels))

    #if I missed a case... just go straight
    curve = "straight"    
    
    #line goes from left to right
    if(cnt_left_top < (cnt_right_top - (num_pix_10 / 2))): 
        print("Case 1")    
        curve = "left"
        
    #special case 'line does not touch top'
    elif(cnt_left_top == num_pix_10):                                     
        if(ratio_vert_left == 1 and not ratio_vert_right == 1):
            print("Case 2")
            curve = "sharp right"
        elif(ratio_vert_left < ratio_vert_right):
            print("Case 3")
            curve = "sharp left"            
        if(ratio_vert_right == 1 and not ratio_vert_left == 1):
            print("Case 4")
            curve = "sharp left"
        elif(ratio_vert_right < ratio_vert_left):
            print("Case 5")
            curve = "sharp right"            
    
    #line goes from right to left
    elif((cnt_left_top - (num_pix_10 / 2)) > cnt_right_top):
        print("Case 6")
        curve = "right"
        
    #line is straight
    else:
        print("Case 7")
        curve = "straight"        
                
    #line is not in the picture 
    if(all_black_pixels == 0):
        print("Case 8")
        curve = "backwards"
    '''
    
    #return curve for robot
    return curve 
    
def count_pxl(img):
    result = 0

    # for-loop dummy
    for i in range(
      1):  # go from row 0 to 1 in steps of 1 (= the first row)
      k = 0
      #j = img[i, k, 0]
      j = img[i, k]
      print("J = " + str(j))
      while j > 45:  # as long as current pixel is black (is background)
        result += 1
        k += 1
        if (k < len(img[i])):  # check if it's still in bounds
          #j = img[i, k, 0]  # jump to next pixel
          j = img[i, k]  # jump to next pixel
        else:
          break

    return result

def get_state(img):
    # get left edge of line
    left = count_pxl(img)
    # flip image vertically (pixel on the right will be on the left,
    # pixel on top stays on top)
    reversed_img = np.flip(img, 1)
    # get right edge of line (start counting from the right)
    right = count_pxl(reversed_img)

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

def make_average_img(img_arr):
    amount = len(img_arr)
    new_img = img_arr[0]
    for i in range(1, amount):
        save = img_arr[i]
        new_img = np.add(new_img, save)
        
    new_img = np.true_divide(new_img, amount)
    return new_img
    
def slice_img(img):
    #new_img = np.zeros(shape=[1, len(img[0])])
    new_img = np.zeros(shape=[1, 50])
    #for i in range(len(img[0])):
    for i in range(50):
        new_img[0, i] = img[0, i, 0]
    return new_img

if __name__=='__main__':   
    #open test image
    path = '/home/elisabeth/catkin_ws/src/Q-Learning/drive_three_pi/src/img/'
    img0 = cv.imread(path + 'lost1.jpg')
    img0 = slice_img(img0)
    img1 = np.array([[74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 46, 46, 10, 10, 10, 10, 10, 10, 10, 10, 46, 45, 44, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 22, 22, 0, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74]])
    img2 = cv.imread(path + 'lost3.jpg')        
    img2 = slice_img(img2)
    img3 = np.array([[74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 10, 10, 10, 10, 10, 10, 10, 10, 46, 46, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74]])
    img_arr = [img0, img1, img2, img3]
    
    imgs = np.empty(shape=[1, 200])
    for i in range(len(imgs[0])):
        if(i < 50):
            imgs[0, i] = img0[0, i]
        elif(i >= 50 and i < 100):
            imgs[0, i] = img1[0, i-50]
        elif(i >= 100 and i < 150):
            imgs[0, i] = img2[0, i-100]
        else:
            imgs[0, i] = img3[0, i-150]            
    print("imgs before = " + str(imgs))
    
    imgs = np.reshape(imgs, (4, 50))
    new_img = make_average_img(imgs)
    
    #img = cv.imread('img/test.jpg')        #check
    #img = cv.imread('img/test1.jpg')       #check
    #img = cv.imread('img/test2.jpg')       #check
    #img = cv.imread('img/test3.jpg')       #check
    #img = cv.imread('img/test4.jpg')       #check
    #img = cv.imread('img/test5.jpg')       #check
    #img = cv.imread('img/test6.jpg')       #check
    #img = cv.imread('img/test7.jpg')       #check
    #img = cv.imread('img/test8.jpg')       #check
    #img = cv.imread('img/test9.jpg')       #check
    #img = cv.imread('img/test10.jpg')      #check
    #img = cv.imread('img/test11.jpg')      #check
    #img = cv.imread('img/test12.jpg')      #check
    #img = cv.imread('img/test13.jpg')      #check
    #img = cv.imread('img/test14.jpg')      #check
    #img = cv.imread('img/test15.jpg')      #check           #why is the whole picture black?? should be white
    #img = cv.imread('img/test16.jpg')      #check
    #img = cv.imread('img/test17.jpg')      #check
    #img = cv.imread('img/test18.jpg')      #check
    #img = cv.imread('img/test19.jpg')      #
    # print("Image array = " + str(img))
    # print(type(img))
    
    plt.imshow(img0)
    print("Image 1 = " + str(img0))
    print("imgs[0] after = " + str(imgs[0]))
    plt.show()
    
    plt.imshow(img1)
    print("Image 2 = " + str(img1))
    print("imgs[1] after = " + str(imgs[1]))
    plt.show()
    
    plt.imshow(img2)
    print("Image 3 = " + str(img2))
    print("imgs[2] after = " + str(imgs[2]))
    plt.show()
        
    plt.imshow(img3)
    print("Image 4 = " + str(img3))
    print("imgs[2] after = " + str(imgs[2]))
    plt.show()
    
    plt.imshow(new_img)
    print("New Image = " + str(new_img))
    plt.show()
    
    # plt.imshow(img)    
    # plt.show()
    
    
    # print("Dimensions = " + str(img.shape))
    state0 = get_state(img0)
    state1 = get_state(img1)
    state2 = get_state(img2)
    state3 = get_state(img3)
    average_state = (state0 + state1 + state2 + state3) / 4
    state = get_state(new_img)
    print("Average state = " + str(average_state))
    print("State = " + str(state))
    
    #crop image
    #print("Dimensions before = " + str(img.shape))
    #cropped_img = crop_image(img)
    #print("Dimensions after = " + str(cropped_img.shape))
    
    #segmentation
    #seg_img = segmentation(img)
    #print(seg_img)
    #plt.imshow(seg_img, cmap = "gray")
    #plt.show()
    
    #choose steering direction
    #curve = calc_curve(seg_img)
    #print("Robot should drive " +  str(curve))    