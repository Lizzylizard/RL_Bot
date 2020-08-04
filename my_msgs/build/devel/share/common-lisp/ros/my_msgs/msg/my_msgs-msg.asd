
(cl:in-package :asdf)

(defsystem "my_msgs-msg"
  :depends-on (:roslisp-msg-protocol :roslisp-utils )
  :components ((:file "_package")
    (:file "VelJoint" :depends-on ("_package_VelJoint"))
    (:file "_package_VelJoint" :depends-on ("_package"))
  ))