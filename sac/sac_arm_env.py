import time
import json
import zmq
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class SimpleReachReward:
    """Same reward function as Dreamer setup."""
    def __init__(self, target_pos=np.array([0.1, 0.35, 0.35])):
        self.target_pos = np.array(target_pos)
        self.best_distance = float('inf') 

    def reset(self):
        print("[SimpleReachReward] Target:", self.target_pos)
        self.best_distance = float('inf')

    def __call__(self, obs):
        if 'actual_pose' not in obs:
            return 0.0

        current_pos = np.atleast_1d(obs['actual_pose'])[:3]
        distance = np.linalg.norm(current_pos - self.target_pos)
        
        reward = 0.0

        # 1. Exponential Proximity
        proximity = np.exp(-10.0 * distance)
        reward += proximity
        
        # 2. Progress Bonus
        if distance < self.best_distance:
            if self.best_distance != float('inf'):
                progress_bonus = 1.0 * (self.best_distance - distance)
                reward += progress_bonus
                
            self.best_distance = distance
        
        # 3. Hold Bonus
        hold_bonus = 0.0
        if distance < 0.04:  # within 4 cm
            hold_bonus += 0.1
        
        reward += hold_bonus

        # 4. Final Success Bonuses
        if distance < 0.02:  # within 2 cm
            reward += 1.0
        if distance < 0.01:  # within 1 cm
            reward += 2.0
            
        return float(reward)


class SACRobotArmEnv(gym.Env):
    """
    Gymnasium wrapper for real robot arm with SAC.
    Maintains same interface and reward function as Dreamer setup.
    """
    metadata = {'render.modes': []}
    
    def __init__(self, ip='127.0.0.1', port_sub=5558, port_pub=5559, hz=10.0):
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
        
        # Publish actions to bridge (bridge subscribes on 5559)
        self.pub = self.ctx.socket(zmq.PUB)
        self.pub.connect(f"tcp://{ip}:{port_pub}")
        
        # State tracking
        self.step_count = 0
        self.max_episode_steps = 500  # Configurable episode length
        
        # Define Gym spaces
        # Observation: [arm_joints(6) + actual_pose(7) + wrist(1) + gripper(1) + contacts(2)]
        obs_dim = 6 + 7 + 1 + 1 + 2  # = 17
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(obs_dim,), 
            dtype=np.float32
        )
        
        # Action: [dx, dy, dz, dwrist, dgrip] normalized to [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(5,), 
            dtype=np.float32
        )
        
        print(f"[SACRobotArmEnv] Initialized")
        print(f"  Observation space: {self.observation_space.shape}")
        print(f"  Action space: {self.action_space.shape}")

    def _receive_obs(self):
        """Blocking receive of observation from bridge."""
        while True:
            try:
                msg_str = self.sub.recv_string()
                obs_dict = json.loads(msg_str)
                
                # Check for required keys
                required = ['arm_joints', 'actual_pose', 'wrist_angle',
                           'gripper_state', 'left_contact', 'right_contact']
                if all(k in obs_dict for k in required):
                    return obs_dict
            except Exception as e:
                print(f"[SACRobotArmEnv] Error receiving obs: {e}")
                time.sleep(0.1)

    def _obs_dict_to_array(self, obs_dict):
        """
        Convert observation dictionary to flat numpy array.
        Format: [arm_joints(6), actual_pose(7), wrist(1), gripper(1), contacts(2)]
        """
        # Handle missing block_pose by using zeros (same as Dreamer env)
        if 'block_pose' not in obs_dict:
            obs_dict['block_pose'] = [0.0] * 7
        
        # Extract and flatten components
        arm_joints = np.array(obs_dict['arm_joints'][:6], dtype=np.float32)
        actual_pose = np.array(obs_dict['actual_pose'][:7], dtype=np.float32)
        wrist = np.array([obs_dict['wrist_angle'][0]], dtype=np.float32)
        gripper = np.array([obs_dict['gripper_state'][0]], dtype=np.float32)
        left_contact = np.array([obs_dict['left_contact'][0]], dtype=np.float32)
        right_contact = np.array([obs_dict['right_contact'][0]], dtype=np.float32)
        
        # Concatenate into single observation vector
        obs = np.concatenate([
            arm_joints,
            actual_pose,
            wrist,
            gripper,
            left_contact,
            right_contact
        ])
        
        return obs

    def reset(self, seed=None, options=None):
        """Reset the environment."""
        super().reset(seed=seed)
        
        print("[SACRobotArmEnv] Resetting...")
        
        # Reset reward function state
        self.reward_fn.reset()
        
        # Send reset command (send twice for reliability)
        self.pub.send_string(json.dumps({"reset": True}))
        time.sleep(2.0)
        
        self.pub.send_string(json.dumps({"reset": True}))
        time.sleep(2.0)
        
        # Receive initial observation
        obs_dict = self._receive_obs()
        obs = self._obs_dict_to_array(obs_dict)
        
        self.step_count = 0
        
        # Store dict for reward computation
        self.last_obs_dict = obs_dict
        
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
        loop_start = time.time()
        
        # Ensure action is numpy array
        action = np.array(action, dtype=np.float32)
        
        # Send action to bridge (raw normalized deltas)
        action_msg = {
            "action": action.tolist()
        }
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
        terminated = False  # Task success (could add distance threshold)
        truncated = self.step_count >= self.max_episode_steps
        
        # Optional: Early termination on success
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
            'distance': distance if 'distance' in locals() else None
        }
        
        return obs, reward, terminated, truncated, info

    def close(self):
        """Clean up resources."""
        self.ctx.term()
        print("[SACRobotArmEnv] Closed")