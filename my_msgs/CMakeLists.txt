cmake_minimum_required(VERSION 2.8.3)
project(my_msgs)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  #ADDED
  #actionlib_msgs
  geometry_msgs
  message_generation
  std_msgs
)

## Generate messages in the 'msg' folder
add_message_files(
  DIRECTORY msg 
  FILES
  VelJoint.msg
)

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  geometry_msgs
  std_msgs
)

catkin_package(
 CATKIN_DEPENDS 
  message_generation 
  std_msgs 
  geometry_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

