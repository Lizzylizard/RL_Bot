#!/usr/bin/env python
import rospy
#from /home/elisabeth/catkin_ws/src/ROS_Packages/my_msgs.msg import VelJoint
from my_msgs.msg import VelJoint
 
def move():
    # Starts a new node
    rospy.init_node('move_three_pi', anonymous=True)
    velocity_publisher = rospy.Publisher('/cmd_vel', VelJoint, queue_size=10)
    vel_msg = VelJoint()

    #Receiveing the user's input
    print("Let's move your robot")
    speed = float(input("Input your speed:"))
    distance = float(input("Type your distance:"))
    isForward = bool(input("Foward?: "))#True or False

    #Checking if the movement is forward or backwards
    if(isForward):
        vel_msg.left_vel = abs(speed)
        vel_msg.right_vel = abs(speed)
    else:
        vel_msg.left_vel = -abs(speed)
        vel_msg.right_vel = -abs(speed)
    #Since we are moving just in x-axis
    '''vel_msg.linear.y = 0
    vel_msg.linear.z = 0
    vel_msg.angular.x = 0
    vel_msg.angular.y = 0
    vel_msg.angular.z = 0'''

    while not rospy.is_shutdown():

        #Setting the current time for distance calculus
        t0 = rospy.Time.now().to_sec()
        current_distance = 0

        #Loop to move the turtle in an specified distance
        while(current_distance < distance):
            #Publish the velocity
            velocity_publisher.publish(vel_msg)
            #Takes actual time to velocity calculus
            t1=rospy.Time.now().to_sec()
            #Calculates distancePoseStamped
            current_distance= speed*(t1-t0)
        #After the loop, stops the robot
        vel_msg.left_vel = float(0)
        vel_msg.right_vel = float(0)
        #Force the robot to stop
        velocity_publisher.publish(vel_msg)

if __name__ == '__main__':
    try:
        #Testing our function
        move()
    except rospy.ROSInterruptException: pass