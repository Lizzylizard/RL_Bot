# generated from genmsg/cmake/pkg-genmsg.cmake.em

message(STATUS "vel_joint_motors: 1 messages, 0 services")

set(MSG_I_FLAGS "-Ivel_joint_motors:/home/elisabeth/Dokumente/Bachelor/catkin_ws/src/ROS_Packages/my_msgs/msg;-Igeometry_msgs:/opt/ros/melodic/share/geometry_msgs/cmake/../msg;-Istd_msgs:/opt/ros/melodic/share/std_msgs/cmake/../msg")

# Find all generators
find_package(gencpp REQUIRED)
find_package(geneus REQUIRED)
find_package(genlisp REQUIRED)
find_package(gennodejs REQUIRED)
find_package(genpy REQUIRED)

add_custom_target(vel_joint_motors_generate_messages ALL)

# verify that message/service dependencies have not changed since configure



get_filename_component(_filename "/home/elisabeth/Dokumente/Bachelor/catkin_ws/src/ROS_Packages/my_msgs/msg/VelJoint.msg" NAME_WE)
add_custom_target(_vel_joint_motors_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "vel_joint_motors" "/home/elisabeth/Dokumente/Bachelor/catkin_ws/src/ROS_Packages/my_msgs/msg/VelJoint.msg" ""
)

#
#  langs = gencpp;geneus;genlisp;gennodejs;genpy
#

### Section generating for lang: gencpp
### Generating Messages
_generate_msg_cpp(vel_joint_motors
  "/home/elisabeth/Dokumente/Bachelor/catkin_ws/src/ROS_Packages/my_msgs/msg/VelJoint.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/vel_joint_motors
)

### Generating Services

### Generating Module File
_generate_module_cpp(vel_joint_motors
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/vel_joint_motors
  "${ALL_GEN_OUTPUT_FILES_cpp}"
)

add_custom_target(vel_joint_motors_generate_messages_cpp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_cpp}
)
add_dependencies(vel_joint_motors_generate_messages vel_joint_motors_generate_messages_cpp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/elisabeth/Dokumente/Bachelor/catkin_ws/src/ROS_Packages/my_msgs/msg/VelJoint.msg" NAME_WE)
add_dependencies(vel_joint_motors_generate_messages_cpp _vel_joint_motors_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(vel_joint_motors_gencpp)
add_dependencies(vel_joint_motors_gencpp vel_joint_motors_generate_messages_cpp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS vel_joint_motors_generate_messages_cpp)

### Section generating for lang: geneus
### Generating Messages
_generate_msg_eus(vel_joint_motors
  "/home/elisabeth/Dokumente/Bachelor/catkin_ws/src/ROS_Packages/my_msgs/msg/VelJoint.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/vel_joint_motors
)

### Generating Services

### Generating Module File
_generate_module_eus(vel_joint_motors
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/vel_joint_motors
  "${ALL_GEN_OUTPUT_FILES_eus}"
)

add_custom_target(vel_joint_motors_generate_messages_eus
  DEPENDS ${ALL_GEN_OUTPUT_FILES_eus}
)
add_dependencies(vel_joint_motors_generate_messages vel_joint_motors_generate_messages_eus)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/elisabeth/Dokumente/Bachelor/catkin_ws/src/ROS_Packages/my_msgs/msg/VelJoint.msg" NAME_WE)
add_dependencies(vel_joint_motors_generate_messages_eus _vel_joint_motors_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(vel_joint_motors_geneus)
add_dependencies(vel_joint_motors_geneus vel_joint_motors_generate_messages_eus)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS vel_joint_motors_generate_messages_eus)

### Section generating for lang: genlisp
### Generating Messages
_generate_msg_lisp(vel_joint_motors
  "/home/elisabeth/Dokumente/Bachelor/catkin_ws/src/ROS_Packages/my_msgs/msg/VelJoint.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/vel_joint_motors
)

### Generating Services

### Generating Module File
_generate_module_lisp(vel_joint_motors
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/vel_joint_motors
  "${ALL_GEN_OUTPUT_FILES_lisp}"
)

add_custom_target(vel_joint_motors_generate_messages_lisp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_lisp}
)
add_dependencies(vel_joint_motors_generate_messages vel_joint_motors_generate_messages_lisp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/elisabeth/Dokumente/Bachelor/catkin_ws/src/ROS_Packages/my_msgs/msg/VelJoint.msg" NAME_WE)
add_dependencies(vel_joint_motors_generate_messages_lisp _vel_joint_motors_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(vel_joint_motors_genlisp)
add_dependencies(vel_joint_motors_genlisp vel_joint_motors_generate_messages_lisp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS vel_joint_motors_generate_messages_lisp)

### Section generating for lang: gennodejs
### Generating Messages
_generate_msg_nodejs(vel_joint_motors
  "/home/elisabeth/Dokumente/Bachelor/catkin_ws/src/ROS_Packages/my_msgs/msg/VelJoint.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/vel_joint_motors
)

### Generating Services

### Generating Module File
_generate_module_nodejs(vel_joint_motors
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/vel_joint_motors
  "${ALL_GEN_OUTPUT_FILES_nodejs}"
)

add_custom_target(vel_joint_motors_generate_messages_nodejs
  DEPENDS ${ALL_GEN_OUTPUT_FILES_nodejs}
)
add_dependencies(vel_joint_motors_generate_messages vel_joint_motors_generate_messages_nodejs)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/elisabeth/Dokumente/Bachelor/catkin_ws/src/ROS_Packages/my_msgs/msg/VelJoint.msg" NAME_WE)
add_dependencies(vel_joint_motors_generate_messages_nodejs _vel_joint_motors_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(vel_joint_motors_gennodejs)
add_dependencies(vel_joint_motors_gennodejs vel_joint_motors_generate_messages_nodejs)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS vel_joint_motors_generate_messages_nodejs)

### Section generating for lang: genpy
### Generating Messages
_generate_msg_py(vel_joint_motors
  "/home/elisabeth/Dokumente/Bachelor/catkin_ws/src/ROS_Packages/my_msgs/msg/VelJoint.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/vel_joint_motors
)

### Generating Services

### Generating Module File
_generate_module_py(vel_joint_motors
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/vel_joint_motors
  "${ALL_GEN_OUTPUT_FILES_py}"
)

add_custom_target(vel_joint_motors_generate_messages_py
  DEPENDS ${ALL_GEN_OUTPUT_FILES_py}
)
add_dependencies(vel_joint_motors_generate_messages vel_joint_motors_generate_messages_py)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/elisabeth/Dokumente/Bachelor/catkin_ws/src/ROS_Packages/my_msgs/msg/VelJoint.msg" NAME_WE)
add_dependencies(vel_joint_motors_generate_messages_py _vel_joint_motors_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(vel_joint_motors_genpy)
add_dependencies(vel_joint_motors_genpy vel_joint_motors_generate_messages_py)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS vel_joint_motors_generate_messages_py)



if(gencpp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/vel_joint_motors)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/vel_joint_motors
    DESTINATION ${gencpp_INSTALL_DIR}
  )
endif()
if(TARGET geometry_msgs_generate_messages_cpp)
  add_dependencies(vel_joint_motors_generate_messages_cpp geometry_msgs_generate_messages_cpp)
endif()
if(TARGET std_msgs_generate_messages_cpp)
  add_dependencies(vel_joint_motors_generate_messages_cpp std_msgs_generate_messages_cpp)
endif()

if(geneus_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/vel_joint_motors)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/vel_joint_motors
    DESTINATION ${geneus_INSTALL_DIR}
  )
endif()
if(TARGET geometry_msgs_generate_messages_eus)
  add_dependencies(vel_joint_motors_generate_messages_eus geometry_msgs_generate_messages_eus)
endif()
if(TARGET std_msgs_generate_messages_eus)
  add_dependencies(vel_joint_motors_generate_messages_eus std_msgs_generate_messages_eus)
endif()

if(genlisp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/vel_joint_motors)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/vel_joint_motors
    DESTINATION ${genlisp_INSTALL_DIR}
  )
endif()
if(TARGET geometry_msgs_generate_messages_lisp)
  add_dependencies(vel_joint_motors_generate_messages_lisp geometry_msgs_generate_messages_lisp)
endif()
if(TARGET std_msgs_generate_messages_lisp)
  add_dependencies(vel_joint_motors_generate_messages_lisp std_msgs_generate_messages_lisp)
endif()

if(gennodejs_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/vel_joint_motors)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/vel_joint_motors
    DESTINATION ${gennodejs_INSTALL_DIR}
  )
endif()
if(TARGET geometry_msgs_generate_messages_nodejs)
  add_dependencies(vel_joint_motors_generate_messages_nodejs geometry_msgs_generate_messages_nodejs)
endif()
if(TARGET std_msgs_generate_messages_nodejs)
  add_dependencies(vel_joint_motors_generate_messages_nodejs std_msgs_generate_messages_nodejs)
endif()

if(genpy_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/vel_joint_motors)
  install(CODE "execute_process(COMMAND \"/usr/bin/python2\" -m compileall \"${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/vel_joint_motors\")")
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/vel_joint_motors
    DESTINATION ${genpy_INSTALL_DIR}
  )
endif()
if(TARGET geometry_msgs_generate_messages_py)
  add_dependencies(vel_joint_motors_generate_messages_py geometry_msgs_generate_messages_py)
endif()
if(TARGET std_msgs_generate_messages_py)
  add_dependencies(vel_joint_motors_generate_messages_py std_msgs_generate_messages_py)
endif()
