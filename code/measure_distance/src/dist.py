#!/usr/bin/env python

import rospy, time, math, cv2, sys

#message types
from sensor_msgs.msg import Range

#global variables
distance = 0.0

#callback
def callback(msg):
    rospy.loginfo("Callback invoked")
    global distance
    distance = msg.range

#react to distance
def process_dist():
    if(distance >= 1):
        print("Street is free; Distance = " + str(distance))
    elif(distance < 1 and distance >= 0.5):
        print("... I see something...; Distance = " + str(distance))
    else:
        print("TOO CLOSE!!; Distance = " + str(distance))

#shutdown method
def shutdown():
    print("Ctrl+c")

if __name__=='__main__':
    #ROS node
    rospy.init_node('distance')
    #subscriber to sonar sensor
    sub=rospy.Subscriber('/sensor/sonar_front', Range,
                         callback)

    rate = rospy.Rate(1)
    rospy.on_shutdown(shutdown)

    try:
        while not rospy.is_shutdown():
            # reaction
            process_dist()
        rate.sleep()
    except rospy.ROSInterruptException:
        print("Interrupted")