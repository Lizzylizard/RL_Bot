;; Auto-generated. Do not edit!


(when (boundp 'my_msgs::VelJoint)
  (if (not (find-package "MY_MSGS"))
    (make-package "MY_MSGS"))
  (shadow 'VelJoint (find-package "MY_MSGS")))
(unless (find-package "MY_MSGS::VELJOINT")
  (make-package "MY_MSGS::VELJOINT"))

(in-package "ROS")
;;//! \htmlinclude VelJoint.msg.html


(defclass my_msgs::VelJoint
  :super ros::object
  :slots (_left_vel _right_vel ))

(defmethod my_msgs::VelJoint
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

(setf (get my_msgs::VelJoint :md5sum-) "74ad8f7ea4d888606e4f41069cec47ff")
(setf (get my_msgs::VelJoint :datatype-) "my_msgs/VelJoint")
(setf (get my_msgs::VelJoint :definition-)
      "float32 left_vel
float32 right_vel
")



(provide :my_msgs/VelJoint "74ad8f7ea4d888606e4f41069cec47ff")


