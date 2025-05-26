#!/usr/bin/env python
import rclpy
import pinocchio as pin
import numpy as np
import eigenpy

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from geometry_msgs.msg import PoseStamped, Transform, Quaternion, TransformStamped
from tf2_ros.transform_broadcaster import TransformBroadcaster


def transform_msg_to_se3(transform: Transform) -> pin.SE3:
    t = np.array(
        [
            transform.translation.x,
            transform.translation.y,
            transform.translation.z,
        ]
    )
    q = eigenpy.Quaternion(
        transform.rotation.w,
        transform.rotation.x,
        transform.rotation.y,
        transform.rotation.z,
    )
    return pin.SE3(q, t)


class TfToPose(rclpy.node.Node):
    def __init__(self):
        super().__init__("apriltag_to_msg")
        self.camera_frame = "camera_color_optical_frame"
        self.world_frame = "fer_link0"
        # self.object_frame = "tless-obj_000031"
        self.object_frame = "tless-obj_000031"

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.pose_pub = self.create_publisher(PoseStamped, "/object/detections", 5)
        self.timer = self.create_timer(0.01, self.update)

    def update(self):
        t = TransformStamped()

        # Read message content and assign it to
        # corresponding tf variables
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "tless-obj_000031"
        t.child_frame_id = "current_object"
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0

        # Send the transformation
        self.tf_broadcaster.sendTransform(t)
        try:
            cMo_msg = self.tf_buffer.lookup_transform(
                self.camera_frame, self.object_frame, rclpy.time.Time()
            )
            cMo = transform_msg_to_se3(cMo_msg.transform)
            wMc_msg = self.tf_buffer.lookup_transform(
                self.world_frame, self.camera_frame, cMo_msg.header.stamp
            )
            wMo = pin.se3ToXYZQUAT(transform_msg_to_se3(wMc_msg.transform) * cMo)
        except TransformException as ex:
            self.get_logger().info(
                f"Could not transform {self.world_frame} to {self.object_frame}: {ex}"
            )
            return
        ps = PoseStamped(header=cMo_msg.header)
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x = wMo[0]
        ps.pose.position.y = wMo[1]
        ps.pose.position.z = wMo[2]
        quat = Quaternion()
        quat.x = wMo[3]
        quat.y = wMo[4]
        quat.z = wMo[5]
        quat.w = wMo[6]
        ps.pose.orientation = quat

        self.pose_pub.publish(ps)


def main():
    rclpy.init()
    node = TfToPose()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()


if __name__ == "__main__":
    main()
