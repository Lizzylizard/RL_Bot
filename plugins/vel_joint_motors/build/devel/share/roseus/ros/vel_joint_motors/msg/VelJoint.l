;; Auto-generated. Do not edit!


(when (boundp 'vel_joint_motors::VelJoint)
  (if (not (find-package "VEL_JOINT_MOTORS"))
    (make-package "VEL_JOINT_MOTORS"))
  (shadow 'VelJoint (find-package "VEL_JOINT_MOTORS")))
(unless (find-package "VEL_JOINT_MOTORS::VELJOINT")
  (make-package "VEL_JOINT_MOTORS::VELJOINT"))

(in-package "ROS")
;;//! \htmlinclude VelJoint.msg.html


(defclass vel_joint_motors::VelJoint
  :super ros::object
  :slots (_left_vel _right_vel ))

(defmethod vel_joint_motors::VelJoint
  (:init
   (&key
    ((:left_vel __left_vel) 0.0)
    ((:right_vel __right_vel) 0.0)
    )
   (send-super :init)
   (setq _left_vel (float __left_vel))
   (setq _right_vel (float __right_vel))
   self)
  (:left_vel
   (&optional __left_vel)
   (if __left_vel (setq _left_vel __left_vel)) _left_vel)
  (:right_vel
   (&optional __right_vel)
   (if __right_vel (setq _right_vel __right_vel)) _right_vel)
  (:serialization-length
   ()
   (+
    ;; float32 _left_vel
    4
    ;; float32 _right_vel
    4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; float32 _left_vel
       (sys::poke _left_vel (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _right_vel
       (sys::poke _right_vel (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; float32 _left_vel
     (setq _left_vel (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _right_vel
     (setq _right_vel (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;;
   self)
  )

(setf (get vel_joint_motors::VelJoint :md5sum-) "74ad8f7ea4d888606e4f41069cec47ff")
(setf (get vel_joint_motors::VelJoint :datatype-) "vel_joint_motors/VelJoint")
(setf (get vel_joint_motors::VelJoint :definition-)
      "float32 left_vel
float32 right_vel
")



(provide :vel_joint_motors/VelJoint "74ad8f7ea4d888606e4f41069cec47ff")


