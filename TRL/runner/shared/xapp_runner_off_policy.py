import time
import numpy as np
import torch
from trl.runner.shared.base_runner import Runner
from trl.utils.shared_buffer import SharedReplayBuffer
import wandb
import imageio
from collections import defaultdict

def _t2n(x):
    return x.detach().cpu().numpy()

class OffPolicyxAppRunner(Runner):
    def __init__(self, config):
        super(OffPolicyxAppRunner, self).__init__(config)
        self.prev_n_agents = 0
        
        # Off-policy specific parameters
        self.buffer_size = getattr(config, 'buffer_size', 100000)
        self.batch_size = getattr(config, 'batch_size', 256)
        self.train_interval = getattr(config, 'train_interval', 1)  
        self.min_buffer_size = getattr(config, 'min_buffer_size', 1000) 
        
        self.buffers = {}  # {num_agents: buffer}
        self.current_buffer = None

    def _init_buffer(self, num_agents=None):
        if num_agents is None:
            num_agents = self.num_agents
            
        if num_agents == 0:
            self.current_buffer = None
            return

        if num_agents in self.buffers:
            self.current_buffer = self.buffers[num_agents]
            return

        agent_obs_space = self.envs.observation_space[0]
        agent_action_space = self.envs.action_space[0]
        agent_share_obs_space = self.envs.share_observation_space[0]
        
        buffer = SharedReplayBuffer(
            self.all_args, 
            num_agents,
            agent_obs_space,
            agent_share_obs_space,
            agent_action_space,
            "off_policy",
            buffer_size=self.buffer_size
        )
        
        self.buffers[num_agents] = buffer
        self.current_buffer = buffer
        print(f"Created new off-policy buffer for {num_agents} agents")

    def run(self):
        self.warmup()   
        start_time = time.time()
        total_steps = 0
        
        while total_steps < self.num_env_steps:
            if self.use_linear_lr_decay:
                frac = total_steps / float(self.num_env_steps) if self.num_env_steps > 0 else 0.0
                if hasattr(self.trainer, 'policy') and hasattr(self.trainer.policy, 'lr_decay'):
                    self.trainer.policy.lr_decay(frac)

            if self.num_agents == 0:
                self.warmup()
                continue

            values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect()
            obs, rewards, dones, infos = self.envs.step(actions_env)
            
            #  agent  
            real_n_agents_after_step = obs.shape[1] if obs.ndim == 3 else obs.shape[0] if obs.ndim == 2 else 0
            
            if real_n_agents_after_step == self.num_agents:
                self.store_transition(obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic)
                total_steps += self.n_rollout_threads
                
                if total_steps % self.train_interval == 0 and self.can_train():
                    train_infos = self.train_off_policy()
                    
                    if (total_steps // self.n_rollout_threads) % self.log_interval == 0:
                        self.log_info(train_infos, total_steps, start_time)
                        
            else:
                print(f"Agent count changed: {self.num_agents} -> {real_n_agents_after_step}")
                self.handle_agent_count_change(real_n_agents_after_step, obs)
                
            if (total_steps // self.n_rollout_threads) % self.save_interval == 0 or total_steps >= self.num_env_steps:
                self.save()
                
            if self.use_eval and (total_steps // self.n_rollout_threads) % self.eval_interval == 0:
                self.eval(total_steps)

    def handle_agent_count_change(self, new_agent_count, obs):
        """agent"""
        self.prev_n_agents = self.num_agents
        self.num_agents = new_agent_count
        
        if hasattr(self.envs, 'num_ue'):
            self.envs.num_ue = self.num_agents
        if hasattr(self.envs, '_rebuild_spaces_based_on_num_ue'):
            self.envs._rebuild_spaces_based_on_num_ue()
        
        self._init_buffer(self.num_agents)
        
        if self.current_buffer is not None and obs is not None:
            self.store_initial_obs(obs)

    def store_initial_obs(self, obs):
        """"""
        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs
            
        self.current_buffer.obs[0] = obs.copy()
        self.current_buffer.share_obs[0] = share_obs.copy()

    def store_transition(self, obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic):
        """ transition?replay buffer"""
        if self.current_buffer is None:
            return
            
        current_num_agents = obs.shape[1]
        
        # RNN
        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), *self.current_buffer.rnn_states_critic.shape[3:]), dtype=np.float32)
        
        masks = np.ones((self.n_rollout_threads, current_num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(current_num_agents, axis=1)
        else:
            share_obs = obs

        self.current_buffer.insert_transition(
            share_obs, obs, rnn_states, rnn_states_critic, 
            actions, action_log_probs, values, rewards, masks
        )

    def can_train(self):
        if self.current_buffer is None:
            return False
        return self.current_buffer.size() >= self.min_buffer_size

    def train_off_policy(self):
        if not self.can_train():
            return {}
            
        batch_data = self.current_buffer.sample(self.batch_size)
        
        self.trainer.prep_training()
        train_infos = self.trainer.train_off_policy(batch_data)
        
        return train_infos

    def warmup(self):
        obs = self.envs.reset()
        self.num_agents = obs.shape[1] if obs.ndim == 3 else obs.shape[0]
        self.prev_n_agents = self.num_agents
        
        self._init_buffer()
        if self.current_buffer is not None:
            self.store_initial_obs(obs)

    @torch.no_grad()
    def collect(self):
        """  (     )"""
        if self.current_buffer is None:
            return None, None, None, None, None, None
            
        self.trainer.prep_rollout()
        
        #    
        current_obs = self.current_buffer.get_current_obs()
        current_share_obs = self.current_buffer.get_current_share_obs()
        current_rnn_states = self.current_buffer.get_current_rnn_states()
        current_rnn_states_critic = self.current_buffer.get_current_rnn_states_critic()
        current_masks = self.current_buffer.get_current_masks()
        
        #   (    )
        value, action, action_log_prob, rnn_states, rnn_states_critic = \
            self.trainer.policy.get_actions_with_exploration(
                np.concatenate(current_share_obs),
                np.concatenate(current_obs),
                np.concatenate(current_rnn_states),
                np.concatenate(current_rnn_states_critic),
                np.concatenate(current_masks)
            )
        
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))
        
        actions_env = actions
        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

    @torch.no_grad()
    def eval(self, total_num_steps):
        """    ,   ?agent """
        print("Evaluation...")
        eval_episode_rewards = []
        
        eval_obs = self.eval_envs.reset()
        num_agents_in_eval = eval_obs.shape[1] if eval_obs.ndim == 3 else eval_obs.shape[0]

        if num_agents_in_eval == 0:
            return

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, num_agents_in_eval, self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, num_agents_in_eval, 1), dtype=np.float32)

        for eval_step in range(self.episode_length):
            if num_agents_in_eval == 0:
                break

            if self.use_centralized_V:
                eval_share_obs = eval_obs.reshape(self.n_eval_rollout_threads, -1)
                eval_share_obs = np.expand_dims(eval_share_obs, 1).repeat(num_agents_in_eval, axis=1)
            else:
                eval_share_obs = eval_obs
            
            self.trainer.prep_rollout()
            eval_action, eval_rnn_states_new = self.trainer.policy.act(
                np.concatenate(eval_share_obs),
                np.concatenate(eval_obs),
                np.concatenate(eval_rnn_states),
                np.concatenate(eval_masks),
                deterministic=True
            )
            
            eval_actions = np.array(np.split(_t2n(eval_action), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states_new), self.n_eval_rollout_threads))

            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions)
            eval_episode_rewards.append(eval_rewards)
            
            #  agent  
            prev_num_agents_in_eval = num_agents_in_eval
            num_agents_in_eval = eval_obs.shape[1] if eval_obs.ndim == 3 else eval_obs.shape[0]

            if num_agents_in_eval != prev_num_agents_in_eval:
                #  RNN  
                if num_agents_in_eval < prev_num_agents_in_eval:
                    eval_rnn_states = eval_rnn_states[:, :num_agents_in_eval, ...]
                else:
                    #  RNN 
                    new_rnn_states = np.zeros((self.n_eval_rollout_threads, num_agents_in_eval, self.recurrent_N, self.hidden_size), dtype=np.float32)
                    new_rnn_states[:, :prev_num_agents_in_eval, ...] = eval_rnn_states
                    eval_rnn_states = new_rnn_states
            
            eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, num_agents_in_eval, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

        #    
        eval_episode_rewards = np.array(eval_episode_rewards, dtype=object)
        valid_rewards = [r for r in eval_episode_rewards if r.size > 0]
        if len(valid_rewards) > 0:
            summed_rewards = np.sum(np.concatenate(valid_rewards, axis=1), axis=0)
            eval_average_episode_rewards = np.mean(summed_rewards)
            print(f"eval_average_episode_rewards: {eval_average_episode_rewards}")
            eval_env_infos = {'eval_average_episode_rewards': eval_average_episode_rewards}
            self.log_env(eval_env_infos, total_num_steps)

    def log_info(self, train_infos, total_steps, start_time):
        elapsed = time.time() - start_time
        fps = int(total_steps / elapsed)
        print(f"Steps {total_steps}/{self.num_env_steps}, FPS {fps}")
        
        #    ( ?buffer)
        if self.current_buffer is not None and self.current_buffer.size() > 0:
            avg_rewards = self.current_buffer.get_average_rewards()
            train_infos["avg_episode_rewards"] = avg_rewards
            train_infos["buffer_size"] = self.current_buffer.size()
            train_infos["num_agents"] = self.num_agents
            
        self.log_train(train_infos, total_steps)