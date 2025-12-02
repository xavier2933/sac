#!/usr/bin/env python3
"""
SAC Training Script for Robot Arm Reaching Task
Uses Stable-Baselines3 for SAC implementation
"""

import os
import argparse
from datetime import datetime
import numpy as np
import torch

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    CheckpointCallback, 
    EvalCallback,
    CallbackList
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure

from sac_arm_env import SACRobotArmEnv


class RewardLoggingCallback:
    """Custom callback to log episode rewards and metrics."""
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_count = 0
        
    def __call__(self, locals_dict, globals_dict):
        """Called at each step."""
        # Check if episode ended
        if locals_dict.get('dones', [False])[0]:
            # Extract episode info from monitor wrapper
            infos = locals_dict.get('infos', [{}])
            if len(infos) > 0 and 'episode' in infos[0]:
                ep_info = infos[0]['episode']
                ep_reward = ep_info['r']
                ep_length = ep_info['l']
                
                self.episode_rewards.append(ep_reward)
                self.episode_lengths.append(ep_length)
                self.episode_count += 1
                
                # Log to console
                print(f"\n{'='*60}")
                print(f"Episode {self.episode_count} Complete")
                print(f"  Reward: {ep_reward:.2f}")
                print(f"  Length: {ep_length} steps")
                if len(self.episode_rewards) >= 10:
                    recent_mean = np.mean(self.episode_rewards[-10:])
                    print(f"  Recent Mean (10 eps): {recent_mean:.2f}")
                print(f"{'='*60}\n")
                
        return True


def make_env(ip='127.0.0.1', port_sub=5558, port_pub=5559, hz=10.0):
    """Create and wrap environment."""
    def _init():
        env = SACRobotArmEnv(ip=ip, port_sub=port_sub, port_pub=port_pub, hz=hz)
        # Monitor wrapper tracks episode statistics
        env = Monitor(env)
        return env
    return _init


def train_sac(args):
    """Main training function."""
    
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"sac_robotarm_{timestamp}"
    log_dir = os.path.join(args.log_dir, exp_name)
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"SAC Training - Robot Arm Reaching Task")
    print(f"{'='*60}")
    print(f"Experiment: {exp_name}")
    print(f"Log dir: {log_dir}")
    print(f"Device: {args.device}")
    print(f"Total timesteps: {args.total_timesteps:,}")
    print(f"{'='*60}\n")
    
    # Create environment
    print("Creating environment...")
    env = DummyVecEnv([make_env(
        ip=args.ip,
        port_sub=args.port_sub,
        port_pub=args.port_pub,
        hz=args.hz
    )])
    
    # Configure logger
    logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
    
    # SAC Hyperparameters
    # These are tuned for continuous control tasks
    sac_kwargs = {
        'policy': 'MlpPolicy',
        'env': env,
        'learning_rate': args.learning_rate,
        'buffer_size': args.buffer_size,
        'learning_starts': args.learning_starts,
        'batch_size': args.batch_size,
        'tau': args.tau,
        'gamma': args.gamma,
        'train_freq': args.train_freq,
        'gradient_steps': args.gradient_steps,
        'ent_coef': args.ent_coef,
        'target_update_interval': 1,
        'verbose': 1,
        'device': args.device,
        'tensorboard_log': log_dir,
    }
    
    # Create SAC agent
    print("\nInitializing SAC agent...")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Buffer size: {args.buffer_size:,}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Entropy coefficient: {args.ent_coef}")
    
    if args.resume_from:
        print(f"  Resuming from: {args.resume_from}")
        model = SAC.load(args.resume_from, env=env, **sac_kwargs)
    else:
        model = SAC(**sac_kwargs)
    
    model.set_logger(logger)
    
    # Setup callbacks
    callbacks = []
    
    # Checkpoint callback - save model periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=os.path.join(log_dir, 'checkpoints'),
        name_prefix='sac_model',
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    callbacks.append(checkpoint_callback)
    
    # Reward logging callback
    reward_callback = RewardLoggingCallback(log_dir)
    callbacks.append(reward_callback)
    
    callback = CallbackList(callbacks)
    
    # Train
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60 + "\n")
    
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callback,
            log_interval=10,
            progress_bar=True
        )
        
        # Save final model
        final_path = os.path.join(log_dir, 'final_model')
        model.save(final_path)
        print(f"\n✓ Training complete! Final model saved to: {final_path}")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user")
        interrupt_path = os.path.join(log_dir, 'interrupted_model')
        model.save(interrupt_path)
        print(f"✓ Model saved to: {interrupt_path}")
    
    finally:
        env.close()
    
    print("\n" + "="*60)
    print("Training Summary")
    print("="*60)
    print(f"Episodes completed: {reward_callback.episode_count}")
    if len(reward_callback.episode_rewards) > 0:
        print(f"Mean reward: {np.mean(reward_callback.episode_rewards):.2f}")
        print(f"Best reward: {np.max(reward_callback.episode_rewards):.2f}")
        print(f"Mean episode length: {np.mean(reward_callback.episode_lengths):.1f}")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Train SAC agent on robot arm reaching task'
    )
    
    # Environment args
    parser.add_argument('--ip', type=str, default='127.0.0.1',
                        help='IP address of ZMQ bridge')
    parser.add_argument('--port-sub', type=int, default=5558,
                        help='Port to subscribe to observations')
    parser.add_argument('--port-pub', type=int, default=5559,
                        help='Port to publish actions')
    parser.add_argument('--hz', type=float, default=10.0,
                        help='Control frequency (Hz)')
    
    # Training args
    parser.add_argument('--total-timesteps', type=int, default=100000,
                        help='Total training timesteps')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--buffer-size', type=int, default=100000,
                        help='Replay buffer size')
    parser.add_argument('--learning-starts', type=int, default=1000,
                        help='Steps before learning starts')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--tau', type=float, default=0.005,
                        help='Soft update coefficient')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--train-freq', type=int, default=1,
                        help='Update frequency')
    parser.add_argument('--gradient-steps', type=int, default=1,
                        help='Gradient steps per update')
    parser.add_argument('--ent-coef', type=str, default='auto',
                        help='Entropy coefficient (auto or float)')
    
    # Logging args
    parser.add_argument('--log-dir', type=str, default='./sac_logs',
                        help='Directory for logs and checkpoints')
    parser.add_argument('--save-freq', type=int, default=5000,
                        help='Save model every N steps')
    
    # System args
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda'],
                        help='Device to use for training')
    parser.add_argument('--resume-from', type=str, default=None,
                        help='Path to model to resume from')
    
    args = parser.parse_args()
    
    # Convert ent_coef from string if needed
    if args.ent_coef != 'auto':
        args.ent_coef = float(args.ent_coef)
    
    # Run training
    train_sac(args)


if __name__ == '__main__':
    main()