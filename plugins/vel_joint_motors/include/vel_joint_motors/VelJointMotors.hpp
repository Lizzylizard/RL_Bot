#pragma once

#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>
#include <thread>
#include "ros/ros.h"
#include "ros/callback_queue.h"
#include "ros/subscribe_options.h"
#include "my_msgs/VelJoint.h"
#include "geometry_msgs/Twist.h"

namespace gazebo
{
class VelJointMotors : public ModelPlugin
{
public:
    VelJointMotors();
    ~VelJointMotors();

private:
    physics::ModelPtr model_;
    event::ConnectionPtr updateConnection_;
    std::unique_ptr<ros::NodeHandle> nh_;

    // wheel subscriber
    ros::Subscriber wheel_speed_sub_;
    ros::CallbackQueue wheel_callback_queue_;
    std::thread wheel_queue_thread_;

    // variables
    double old_secs_;
    double freq_update_ = 30.0;
    double left_wheel_speed_ = 0.0;
    double right_wheel_speed_ = 0.0;
    double wheel_torque_ = 100.0;

    std::string wheel_speed_;
    std::string left_wheel_name_;
    std::string right_wheel_name_;

    void Load(physics::ModelPtr _parent, sdf::ElementPtr _sdf);
    void wheelVelCallback(const geometry_msgs::Twist::ConstPtr &vel_msg);
    void OnUpdate();

    void queueThread();
};
} // namespace gazebo