cmake_minimum_required(VERSION 2.8.3)
project(vel_joint_motors)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  gazebo_ros
  roscpp
  my_msgs
  #ADDED
  message_generation
  geometry_msgs
)

find_package(gazebo REQUIRED)

#ADDED
add_message_files(
  DIRECTORY /home/elisabeth/Dokumente/Bachelor/catkin_ws/src/ROS_Packages/my_msgs/msg
  FILES
  VelJoint.msg
)

generate_messages(
  DEPENDENCIES
#ADDED
  geometry_msgs
  std_msgs
)

catkin_package(
 CATKIN_DEPENDS 
 my_msgs
 roscpp 
 gazebo_ros
 #ADDED
 message_runtime
 geometry_msgs
)

link_directories(
  ${GAZEBO_LIBRARY_DIRS}
)

include_directories(
  include 
  ${Boost_INCLUDE_DIR} 
  ${catkin_INCLUDE_DIRS} 
  ${GAZEBO_INCLUDE_DIRS}
)

add_library(
  vel_joint_motors_plugin 
  src/VelJointMotors.cpp
)

add_dependencies(
  vel_joint_motors_plugin 
  ${${PROJECT_NAME}_EXPORTED_TARGETS} 
  ${catkin_EXPORTED_TARGETS}
#ADDED
  my_msgs 
  ${${PROJECT_NAME}_EXPORTED_TARGETS} 
  ${catkin_EXPORTED_TARGETS}
)

target_link_libraries(
  vel_joint_motors_plugin 
  ${catkin_LIBRARIES} 
  ${GAZEBO_LIBRARIES}
)
