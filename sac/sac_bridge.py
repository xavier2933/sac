#!/usr/bin/env python3
import json
import time
import zmq
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32, Bool
import math
import csv
import os
from rclpy.time import Time
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

# --- Helper Functions ---

def pose_to_array(msg: Pose):
    """Convert Pose to array [x, y, z, qx, qy, qz, qw]"""
    p = msg.position
    o = msg.orientation
    return [p.x, p.y, p.z, o.x, o.y, o.z, o.w]

def quaternion_from_euler_z(yaw):
    """Convert yaw (z-axis rotation) to quaternion."""
    half_yaw = yaw / 2.0
    return np.array([
        0.0,
        0.0,
        math.sin(half_yaw),
        math.cos(half_yaw)
    ])

def quaternion_multiply(q1, q2):
    """Multiply two quaternions [x, y, z, w]."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ])

# --- SAC ROS Bridge ---

class SACRosBridge(Node):
    def __init__(self):
        super().__init__("sac_ros_bridge")

        # === TF Setup ===
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # === CSV LOGGING ===
        self.log_dir = os.path.join(os.getcwd(), "sac_log_data")
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.obs_file_name = os.path.join(self.log_dir, "observations.csv")
        self.action_file_name = os.path.join(self.log_dir, "actions.csv")
        
        # Observation CSV
        self.obs_csv_file = open(self.obs_file_name, 'w', newline='')
        self.obs_csv_writer = csv.writer(self.obs_csv_file)
        obs_header = ["time_sec", "time_ns", 
                      "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6",
                      "block_x", "block_y", "block_z", "block_qx", "block_qy", "block_qz", "block_qw",
                      "actual_x", "actual_y", "actual_z", "actual_qx", "actual_qy", "actual_qz", "actual_qw",
                      "wrist_angle", "gripper_state", "left_contact", "right_contact"]
        self.obs_csv_writer.writerow(obs_header)
        
        # Action CSV
        self.act_csv_file = open(self.action_file_name, 'w', newline='')
        self.act_csv_writer = csv.writer(self.act_csv_file)
        act_header = ["time_sec", "time_ns", "dx", "dy", "dz", "wrist", "grip"]
        self.act_csv_writer.writerow(act_header)
        
        # Data cache
        self.data_cache = {
            "arm_joints": [np.nan] * 6,
            "block_pose": [np.nan] * 7,
            "actual_pose": [np.nan] * 7,
            "wrist_angle": [np.nan],
            "gripper_state": [np.nan],
            "left_contact": [np.nan],
            "right_contact": [np.nan],
        }
        
        self.get_logger().info(f"SAC Log files created in: {self.log_dir}")
        
        # === ROS Subscriptions ===
        self.create_subscription(JointState, "/joint_states", self.cb_joint_states, 10)
        self.create_subscription(Pose, "/block_pose", self.cb_block_pose, 10)
        self.create_subscription(Pose, "/actual_end_effector_pose", self.cb_actual_pose, 10)
        self.create_subscription(Float32, "/wrist_angle", self.cb_wrist_angle, 10)
        self.create_subscription(Bool, "/gripper_command", self.cb_gripper_state, 10)
        self.create_subscription(Bool, "/left_contact_detected", self.cb_left_contact, 10)
        self.create_subscription(Bool, "/right_contact_detected", self.cb_right_contact, 10)

        # === ROS Publishers ===
        self.pub_target = self.create_publisher(Pose, "/bc_target_pose", 10)
        self.pub_gripper = self.create_publisher(Bool, "/gripper_cmd_aut", 10)
        self.pub_wrist = self.create_publisher(Float32, "/bc_wrist_angle", 10)
        self.pub_aut = self.create_publisher(Bool, "/autonomous_mode", 10)
        self.pub_reset = self.create_publisher(Bool, "/reset_env", 10)

        # Observation cache for ZMQ
        self.obs_cache = {} 

        # === ZeroMQ Setup ===
        ctx = zmq.Context()
        # Bridge publishes observations on 5558
        self.pub = ctx.socket(zmq.PUB)
        self.pub.bind("tcp://127.0.0.1:5556")
        # Bridge receives actions on 5559
        self.sub = ctx.socket(zmq.SUB)
        self.sub.bind("tcp://127.0.0.1:5557")
        self.sub.setsockopt_string(zmq.SUBSCRIBE, "")

        self.timer = self.create_timer(0.1, self.timer_callback)
        
        # Track current target for delta actions
        self.current_target_pos = None
        self.current_wrist = None
        self.current_gripper = 0.0
        
        # Action scaling (same as Dreamer)
        self.position_delta_scale = 0.005  # 5mm
        self.wrist_delta_scale = 2.0       # 2°
        
        # Debug tracking
        self.received_topics = set()
        
        self.get_logger().info("🤖 SAC ROS bridge ready.")
        self.get_logger().info(f"   Position delta scale: {self.position_delta_scale*100:.1f}cm per unit")
        self.get_logger().info(f"   Wrist delta scale: {self.wrist_delta_scale:.1f}° per unit")

    def __del__(self):
        """Close file handles on destruction."""
        if hasattr(self, 'obs_csv_file') and not self.obs_csv_file.closed:
            self.obs_csv_file.close()
            self.get_logger().info("Observation log closed.")
        if hasattr(self, 'act_csv_file') and not self.act_csv_file.closed:
            self.act_csv_file.close()
            self.get_logger().info("Action log closed.")

    def get_start_pose(self):
        """Get current robot pose from TF and convert to Unity frame."""
        try:
            t = self.tf_buffer.lookup_transform('world', 'panda_hand', Time())
            
            ros_x = t.transform.translation.x
            ros_y = t.transform.translation.y
            ros_z = t.transform.translation.z
            
            # ROS -> Unity frame conversion
            unity_x = -ros_y
            unity_y = ros_z
            unity_z = ros_x
            
            self.current_target_pos = np.array([unity_x, unity_y, unity_z])
            
            if self.current_wrist is None:
                self.current_wrist = 0.0
                self.get_logger().info("ℹ Defaulting start wrist to 0.0")
            
            self.get_logger().info(f"✓ Initialized start pose from TF:")
            self.get_logger().info(f"  ROS:   [{ros_x:.3f}, {ros_y:.3f}, {ros_z:.3f}]")
            self.get_logger().info(f"  Unity: [{unity_x:.3f}, {unity_y:.3f}, {unity_z:.3f}]")
            return True
            
        except TransformException:
            return False

    def log_observation(self, current_time: Time):
        """Log latest observation state to CSV."""
        timestamp = [current_time.nanoseconds // 10**9, current_time.nanoseconds % 10**9]
        
        row = timestamp + \
              self.data_cache["arm_joints"] + \
              self.data_cache["block_pose"] + \
              self.data_cache["actual_pose"] + \
              self.data_cache["wrist_angle"] + \
              self.data_cache["gripper_state"] + \
              self.data_cache["left_contact"] + \
              self.data_cache["right_contact"]
              
        self.obs_csv_writer.writerow(row)
        self.obs_csv_file.flush()

    # === Observation Callbacks ===
    
    def cb_joint_states(self, msg: JointState):
        current_time = self.get_clock().now()
        data = list(msg.position[:6])  # Only first 6 joints
        self.obs_cache["arm_joints"] = data
        self.data_cache["arm_joints"] = data
        self.log_observation(current_time)
        if "arm_joints" not in self.received_topics:
            self.received_topics.add("arm_joints")
            self.get_logger().info("✓ Receiving arm_joints")
    
    def cb_block_pose(self, msg: Pose):
        current_time = self.get_clock().now()
        data = pose_to_array(msg)
        self.obs_cache["block_pose"] = data
        self.data_cache["block_pose"] = data
        self.log_observation(current_time)
        if "block_pose" not in self.received_topics:
            self.received_topics.add("block_pose")
            self.get_logger().info("✓ Receiving block_pose")
    
    def cb_actual_pose(self, msg: Pose):
        current_time = self.get_clock().now()
        
        # Transform ROS -> Unity frame
        ros_p = msg.position
        unity_x = -ros_p.y
        unity_y = ros_p.z
        unity_z = ros_p.x
        
        o = msg.orientation
        data = [unity_x, unity_y, unity_z, o.x, o.y, o.z, o.w]
        
        self.obs_cache["actual_pose"] = data
        self.data_cache["actual_pose"] = data
        
        self.log_observation(current_time)
        if "actual_pose" not in self.received_topics:
            self.received_topics.add("actual_pose")
            self.get_logger().info("✓ Receiving actual_pose")
    
    def cb_wrist_angle(self, msg: Float32):
        current_time = self.get_clock().now()
        data = [msg.data]
        self.obs_cache["wrist_angle"] = data
        self.data_cache["wrist_angle"] = data
        
        if self.current_wrist is None:
            self.current_wrist = msg.data
            self.get_logger().info(f"✓ Initialized start wrist: {self.current_wrist:.1f}")
        
        self.log_observation(current_time)
        if "wrist_angle" not in self.received_topics:
            self.received_topics.add("wrist_angle")
            self.get_logger().info("✓ Receiving wrist_angle")
    
    def cb_gripper_state(self, msg: Bool):
        current_time = self.get_clock().now()
        data = [float(msg.data)]
        self.obs_cache["gripper_state"] = data
        self.data_cache["gripper_state"] = data
        self.log_observation(current_time)
        if "gripper_state" not in self.received_topics:
            self.received_topics.add("gripper_state")
            self.get_logger().info("✓ Receiving gripper_state")
    
    def cb_left_contact(self, msg: Bool):
        current_time = self.get_clock().now()
        data = [float(msg.data)]
        self.obs_cache["left_contact"] = data
        self.data_cache["left_contact"] = data
        self.log_observation(current_time)
        if "left_contact" not in self.received_topics:
            self.received_topics.add("left_contact")
            self.get_logger().info("✓ Receiving left_contact")
    
    def cb_right_contact(self, msg: Bool):
        current_time = self.get_clock().now()
        data = [float(msg.data)]
        self.obs_cache["right_contact"] = data
        self.data_cache["right_contact"] = data
        self.log_observation(current_time)
        if "right_contact" not in self.received_topics:
            self.received_topics.add("right_contact")
            self.get_logger().info("✓ Receiving right_contact")

    # --- Timer and Action Publishing ---
    
    def timer_callback(self):
        # Try to initialize start pose if not done yet
        if self.current_target_pos is None:
            self.get_start_pose()

        # Send observations to SAC - EXACTLY like Dreamer
        if self.obs_cache:
            msg = json.dumps(self.obs_cache)
            self.pub.send_string(msg)
            
            required = ["arm_joints", "actual_pose"]
            if all(k in self.obs_cache for k in required) and "_logged_complete" not in self.received_topics:
                self.get_logger().info("✓ Sending observations to SAC (arm_joints + actual_pose)")
                self.received_topics.add("_logged_complete")

        # Receive actions from SAC agent
        try:
            while self.sub.poll(timeout=0):
                msg_str = self.sub.recv_string(flags=zmq.NOBLOCK)
                # print(f"[Bridge] DEBUG: Received message: {msg_str}")
                data = json.loads(msg_str)
                
                if "reset" in data and data["reset"]:
                    reset_msg = Bool()
                    reset_msg.data = True
                    false_msg = Bool()
                    false_msg.data = False

                    # 1. Publish false immediately
                    self.pub_aut.publish(false_msg)

                    # 2. Wait 0.5 seconds
                    time.sleep(0.5)

                    # 3. Publish TRUE reset
                    self.pub_reset.publish(reset_msg)
                    self.get_logger().info("↻ Reset command sent")

                    # 4. Wait 3 seconds before allowing actions again
                    time.sleep(3.0)
                    self.pub_aut.publish(reset_msg)

                    # 5. Clear states to force re-sync
                    self.current_target_pos = None
                    self.current_wrist = None
                    self.current_gripper = 0.0

                    self.get_logger().info("↻ Reset complete - observations will continue") 
                    
                if "action" in data:
                    action = np.array(data["action"], dtype=np.float32)
                    self.publish_actions(action)
        except zmq.Again:
            pass

    def publish_actions(self, act):
        """
        Publish SAC actions to ROS.
        Actions are normalized deltas in [-1, 1]: [dx, dy, dz, dwrist, dgrip]
        """
        if self.current_target_pos is None:
            return
        
        if self.current_wrist is None:
            self.current_wrist = 0.0

        dx_norm, dy_norm, dz_norm, dwrist_norm, grip_delta = act[:5]
        
        # Scale normalized deltas
        dx = dx_norm * self.position_delta_scale
        dy = dy_norm * self.position_delta_scale
        dz = dz_norm * self.position_delta_scale
        dwrist = dwrist_norm * self.wrist_delta_scale
        
        # Apply deltas
        new_x = self.current_target_pos[0] + dx
        new_y = self.current_target_pos[1] + dy
        new_z = self.current_target_pos[2] + dz
        new_wrist = self.current_wrist + dwrist
        
        self.current_gripper += grip_delta
        self.current_gripper = np.clip(self.current_gripper, 0.0, 1.0)
        
        # Clamp to workspace (Unity Frame)
        new_x = np.clip(new_x, -0.2, 0.2)
        new_y = np.clip(new_y, 0.15, 0.5)
        new_z = np.clip(new_z, 0.2, 0.5)
        new_wrist = np.clip(new_wrist, -180.0, 180.0)
        
        # Update tracked position
        self.current_target_pos = np.array([new_x, new_y, new_z])
        self.current_wrist = new_wrist

        # Construct orientation from wrist angle
        base_down_q = np.array([0.7071068, -0.7071068, 0.0, 0.0])
        q = quaternion_from_euler_z(math.radians(new_wrist))
        final_q = quaternion_multiply(q, base_down_q)

        # Create and publish pose message
        pose_msg = Pose()
        pose_msg.position.x = float(new_x)
        pose_msg.position.y = float(new_y)
        pose_msg.position.z = float(new_z)
        pose_msg.orientation.x = final_q[0]
        pose_msg.orientation.y = final_q[1]
        pose_msg.orientation.z = final_q[2]
        pose_msg.orientation.w = final_q[3]
        
        self.pub_target.publish(pose_msg)

        # Publish gripper
        grip_msg = Bool()
        grip_msg.data = bool(self.current_gripper > 0.5)
        self.pub_gripper.publish(grip_msg)
        
        # Publish wrist
        wrist_msg = Float32()
        wrist_msg.data = float(new_wrist)
        self.pub_wrist.publish(wrist_msg)
        
        self.get_logger().info(
            f"📤 Target: pos=[{new_x:.3f}, {new_y:.3f}, {new_z:.3f}], "
            f"wrist={new_wrist:.1f}°, grip={grip_msg.data}"
        )
        
        # Log action to CSV
        current_time = self.get_clock().now()
        timestamp = [current_time.nanoseconds // 10**9, current_time.nanoseconds % 10**9]
        self.act_csv_writer.writerow(timestamp + list(act[:5]))
        self.act_csv_file.flush()


def main():
    rclpy.init()
    node = SACRosBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()