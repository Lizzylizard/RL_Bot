#include <vel_joint_motors/VelJointMotors.hpp>

namespace gazebo
{
VelJointMotors::VelJointMotors() {}
VelJointMotors::~VelJointMotors() {}

void VelJointMotors::Load(physics::ModelPtr _parent, sdf::ElementPtr _sdf)
{
    model_ = _parent;
    updateConnection_ = event::Events::ConnectWorldUpdateBegin(std::bind(&VelJointMotors::OnUpdate, this));
    old_secs_ = ros::Time::now().toSec();

    if (_sdf->HasElement("wheelTorque"))
        wheel_torque_ = _sdf->Get<double>("wheelTorque");

    if (_sdf->HasElement("leftJoint"))
        left_wheel_name_ = _sdf->Get<std::string>("leftJoint");

    if (_sdf->HasElement("rightJoint"))
        right_wheel_name_ = _sdf->Get<std::string>("rightJoint");

    wheel_speed_ = "/cmd_vel";

    ROS_INFO("Wheel torque %f", wheel_torque_);
    ROS_INFO("Left motor %s", left_wheel_name_.c_str());
    ROS_INFO("Right motor %s", right_wheel_name_.c_str());

    if (!ros::isInitialized())
    {
        ROS_INFO("Initializing...");
        int argc = 0;
        char **argv = NULL;
        ros::init(argc, argv, "set_joint_wheel_speed", ros::init_options::NoSigintHandler);
        return;
    }

    nh_.reset(new ros::NodeHandle("earthquake_rosnode"));

    // set maximum force on joints
    model_->GetJoint(left_wheel_name_)->SetParam("fmax", 0, wheel_torque_);
    model_->GetJoint(right_wheel_name_)->SetParam("fmax", 0, wheel_torque_);

    // initialize joint speed
    model_->GetJoint(left_wheel_name_)->SetParam("vel", 0, left_wheel_speed_);
    model_->GetJoint(right_wheel_name_)->SetParam("vel", 0, right_wheel_speed_);

    //ros::SubscribeOptions so = ros::SubscribeOptions::create<my_msgs::VelJoint>(wheel_speed_, 1, boost::bind(&VelJointMotors::wheelVelCallback, this, _1), ros::VoidPtr(), &wheel_callback_queue_);
	ros::SubscribeOptions so = ros::SubscribeOptions::create<geometry_msgs::Twist>(wheel_speed_, 1, boost::bind(&VelJointMotors::wheelVelCallback, this, _1), ros::VoidPtr(), &wheel_callback_queue_);
	
    wheel_speed_sub_ = nh_->subscribe(so);
    wheel_queue_thread_ = std::thread(std::bind(&VelJointMotors::queueThread, this));
    ROS_INFO("Loaded plugin %s", model_->GetName().c_str());
}

void VelJointMotors::OnUpdate()
{
    double new_secs = ros::Time::now().toSec();
    double time_diff = new_secs - old_secs_;
    double max_time_diff = 0.0;

    if (freq_update_ != 0.0)
    {
        max_time_diff = 1.0 / freq_update_;
    }

    if (time_diff > max_time_diff && time_diff != 0.0)
    {
        old_secs_ = new_secs;

        model_->GetJoint(left_wheel_name_)->SetParam("vel", 0, left_wheel_speed_);
        model_->GetJoint(right_wheel_name_)->SetParam("vel", 0, right_wheel_speed_);
    }
}

//void VelJointMotors::wheelVelCallback(const my_msgs::VelJoint::ConstPtr &vel_msg)
void VelJointMotors::wheelVelCallback(const geometry_msgs::Twist::ConstPtr &vel_msg)
{
    //left_wheel_speed_ = vel_msg->left_vel;
    //right_wheel_speed_ = vel_msg->right_vel;
	
	left_wheel_speed_ = vel_msg->linear.x;
    right_wheel_speed_ = vel_msg->linear.y;

    // ROS_INFO("[l, r]  [%.2f, %.2f]", vel_msg->left_vel, vel_msg->right_vel);
}

void VelJointMotors::queueThread()
{
    static const double timeout = 0.01;
    while (nh_->ok())
    {
        wheel_callback_queue_.callAvailable(ros::WallDuration(timeout));
    }
}

GZ_REGISTER_MODEL_PLUGIN(VelJointMotors)
} // namespace gazebo