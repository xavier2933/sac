import time
import json
import zmq
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class SimpleReachReward:
    def __init__(self, target_pos=np.array([0.1, 0.35, 0.35])):
        self.target_pos = np.array(target_pos)
        self.prev_distance = None
        self.episode_min_distance = float('inf')  # ← Add this

    def reset(self):
        print("[SimpleReachReward] Target:", self.target_pos)
        if self.episode_min_distance != float('inf'):
            print(f"[SimpleReachReward] Episode best distance: {self.episode_min_distance*100:.2f}cm")
        self.prev_distance = None
        self.episode_min_distance = float('inf')

    def __call__(self, obs):
        if 'actual_pose' not in obs:
            return 0.0

        current_pos = np.atleast_1d(obs['actual_pose'])[:3]
        distance = np.linalg.norm(current_pos - self.target_pos)
        
        reward = 0.0

        # 1. Exponential Proximity (Markovian ✓)
        proximity = np.exp(-10.0 * distance)
        reward += proximity
        
        # 2. Progress Bonus (now Markovian)
        if self.prev_distance is not None:
            progress = self.prev_distance - distance
            reward += 5.0 * progress  # Scale up since it's now step-to-step
        
        self.prev_distance = distance
        
        # 3. Hold Bonus (Markovian ✓)
        if distance < 0.04:
            reward += 0.1
        
        # 4. Success Bonuses (Markovian ✓)
        if distance < 0.02:
            reward += 1.0
        if distance < 0.01:
            reward += 2.0
            
        return float(reward)


class SACRobotArmEnv(gym.Env):
    """
    Gymnasium wrapper for real robot arm with SAC.
    MASKED VERSION: Only uses joint positions (6) and actual end-effector pose (7).
    Total observation dimension: 13
    """
    metadata = {'render.modes': []}
    
    def __init__(self, ip='127.0.0.1', port_sub=5556, port_pub=5557, hz=10.0):
        super().__init__()
        
        self.hz = hz
        self.rate_duration = 1.0 / hz
        
        # Action scaling (same as Dreamer)
        self.action_scale = np.array([0.005, 0.005, 0.005, 2.0, 1.0])
        
        # Workspace limits (must match bridge!)
        self.pos_min = np.array([-0.2, 0.15, 0.2])
        self.pos_max = np.array([0.2, 0.5, 0.5])
        self.wrist_min = -180.0
        self.wrist_max = 180.0
        
        # Reward function (same as Dreamer)
        self.reward_fn = SimpleReachReward(target_pos=np.array([0.1, 0.35, 0.35]))
        
        # ZMQ Setup
        print(f"[SACRobotArmEnv] Connecting to ZMQ bridge at {ip}...")
        self.ctx = zmq.Context()
        
        # Subscribe to observations from bridge (bridge publishes on 5558)
        self.sub = self.ctx.socket(zmq.SUB)
        self.sub.connect(f"tcp://{ip}:{port_sub}")
        self.sub.setsockopt_string(zmq.SUBSCRIBE, "")
        # Set timeout to avoid infinite blocking
        self.sub.setsockopt(zmq.RCVTIMEO, 5000)  # 5 second timeout
        
        # Publish actions to bridge (bridge subscribes on 5559)
        self.pub = self.ctx.socket(zmq.PUB)
        self.pub.connect(f"tcp://{ip}:{port_pub}")
        
        # ZMQ needs time to establish connections
        print("[SACRobotArmEnv] Waiting for ZMQ connection to stabilize...")
        time.sleep(1.0)
        print("[SACRobotArmEnv] Connection ready")
        
        # State tracking
        self.step_count = 0
        self.max_episode_steps = 500
        
        # Define Gym spaces - MASKED VERSION
        # Observation: [arm_joints(6) + actual_pose(7)] = 13 dimensions
        obs_dim = 6 + 7  # MASKED: Only joint positions and actual pose
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(obs_dim,), 
            dtype=np.float32
        )
        
        # Action: [dx, dy, dz] normalized to [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(3,), 
            dtype=np.float32
        )
        
        print(f"[SACRobotArmEnv] Initialized (MASKED: joints + pose only)")
        print(f"  Observation space: {self.observation_space.shape}")
        print(f"  Action space: {self.action_space.shape}")

    def _receive_obs(self, clear_stale=True):
        """
        Receive observation from bridge.
        
        Args:
            clear_stale: If True, drain all queued messages and get the most recent one.
                        This ensures we always get fresh observations at ~10Hz.
        """
        if clear_stale:
            # Drain all stale messages to get the most recent observation
            latest_msg = None
            cleared = 0
            
            # Set short timeout for draining
            self.sub.setsockopt(zmq.RCVTIMEO, 100)
            
            try:
                # Keep reading until we hit timeout (no more messages)
                while True:
                    try:
                        latest_msg = self.sub.recv_string(flags=zmq.NOBLOCK)
                        cleared += 1
                    except zmq.Again:
                        break
            finally:
                # Restore normal timeout
                self.sub.setsockopt(zmq.RCVTIMEO, 5000)
            
            # If we got at least one message, use the latest
            if latest_msg is not None:
                obs_dict = json.loads(latest_msg)
                required = ['arm_joints', 'actual_pose']
                if all(k in obs_dict for k in required):
                    return obs_dict
        
        # Fallback: blocking receive with retries
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                msg_str = self.sub.recv_string()
                obs_dict = json.loads(msg_str)
                
                # Check for required keys
                required = ['arm_joints', 'actual_pose']
                if all(k in obs_dict for k in required):
                    return obs_dict
                else:
                    missing = [k for k in required if k not in obs_dict]
                    print(f"[SACRobotArmEnv] Missing required keys: {missing}, retrying...")
                    retry_count += 1
                    
            except zmq.Again:
                print(f"[SACRobotArmEnv] Timeout waiting for observations (attempt {retry_count+1}/{max_retries})")
                retry_count += 1
                time.sleep(1.0)
            except Exception as e:
                print(f"[SACRobotArmEnv] Error receiving obs: {e}")
                retry_count += 1
                time.sleep(0.5)
        
        # If we get here, we failed to receive valid observations
        raise RuntimeError("Failed to receive observations from bridge after multiple attempts. "
                         "Is sac_bridge.py running and receiving data from ROS?")

    def _obs_dict_to_array(self, obs_dict):
        """
        Convert observation dictionary to flat numpy array.
        MASKED VERSION: Only uses [arm_joints(6), actual_pose(7)] = 13 dims
        """
        # Extract and flatten components
        arm_joints = np.array(obs_dict['arm_joints'][:6], dtype=np.float32)
        actual_pose = np.array(obs_dict['actual_pose'][:7], dtype=np.float32)
        
        # Concatenate into single observation vector (MASKED - no wrist, gripper, contacts)
        obs = np.concatenate([
            arm_joints,
            actual_pose
        ])
        
        return obs

    def reset(self, seed=None, options=None):
        """Reset the environment."""
        super().reset(seed=seed)
        
        print("[SACRobotArmEnv] Resetting...")
        
        # Reset reward function state
        self.reward_fn.reset()
        
        # Send reset command (send twice for reliability, like Dreamer)
        self.pub.send_string(json.dumps({"reset": True}))
        time.sleep(2.0)
        
        self.pub.send_string(json.dumps({"reset": True}))
        time.sleep(2.0)
        
        # Clear any stale messages from the socket (CRITICAL for ZMQ PUB/SUB)
        print("[SACRobotArmEnv] Clearing stale messages...")
        cleared = 0
        self.sub.setsockopt(zmq.RCVTIMEO, 100)  # Short timeout for clearing
        try:
            while True:
                self.sub.recv_string(flags=zmq.NOBLOCK)
                cleared += 1
        except zmq.Again:
            pass
        if cleared > 0:
            print(f"[SACRobotArmEnv] Cleared {cleared} stale messages")
        
        # Restore normal timeout
        self.sub.setsockopt(zmq.RCVTIMEO, 5000)
        
        # Now receive fresh observation
        print("[SACRobotArmEnv] Waiting for fresh observation...")
        obs_dict = self._receive_obs()
        obs = self._obs_dict_to_array(obs_dict)
        
        self.step_count = 0
        
        # Store dict for reward computation
        self.last_obs_dict = obs_dict
        
        print("[SACRobotArmEnv] Reset complete!")
        
        # Gymnasium requires returning (obs, info)
        info = {}
        return obs, info

    def step(self, action):
        """
        Execute one step in the environment.
        
        Args:
            action: numpy array of shape (5,) with normalized deltas [-1, 1]
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        # print(f"[SACRobotArmEnv] DEBUG: Step called")
        loop_start = time.time()
        
        # Ensure action is numpy array
        action = np.array(action, dtype=np.float32)
        
        # Pad action with 0.0 for wrist and gripper: [dx, dy, dz, 0, 0]
        full_action = np.concatenate([action, [0.0, 0.0]])
        
        # Send action to bridge (raw normalized deltas)
        action_msg = {
            "action": full_action.tolist()
        }
        # print(f"[SACRobotArmEnv] DEBUG: Sending action: {action}") 
        self.pub.send_string(json.dumps(action_msg))
        
        # Rate limiting
        elapsed = time.time() - loop_start
        sleep_time = self.rate_duration - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
        
        # Receive next observation
        obs_dict = self._receive_obs()
        obs = self._obs_dict_to_array(obs_dict)
        
        # Compute reward using same function as Dreamer
        reward = self.reward_fn(obs_dict)
        
        # Update step counter
        self.step_count += 1
        
        # Check termination conditions
        terminated = False  # Task success
        truncated = self.step_count >= self.max_episode_steps
        
        # Optional: Early termination on success
        distance = None
        if 'actual_pose' in obs_dict:
            current_pos = np.array(obs_dict['actual_pose'][:3])
            target_pos = self.reward_fn.target_pos
            distance = np.linalg.norm(current_pos - target_pos)
            if distance < 0.01:  # Within 1cm = success
                terminated = True
                print(f"[SACRobotArmEnv] ✓ Success! Distance: {distance*100:.2f}cm")
        
        self.last_obs_dict = obs_dict
        
        info = {
            'step': self.step_count,
            'distance': distance
        }
        
        return obs, reward, terminated, truncated, info

    def close(self):
        """Clean up resources."""
        self.ctx.term()
        print("[SACRobotArmEnv] Closed")