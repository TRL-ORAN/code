import time
import numpy as np
import torch
from trl.runner.shared.base_runner import Runner
from trl.utils.shared_buffer import SharedReplayBuffer
import wandb
import imageio
import sys
import signal
import os
import csv

def _t2n(x):
    return x.detach().cpu().numpy()

rewards_G = []
throughput_G = []
latancy_G = []
prbs_G = []
penalty_G = []

def append_reward_to_csv(file_path, rewards, throughput, latancy, prbs, penalty):
        file_exists = os.path.isfile(file_path)
        with open(file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            #if not file_exists:
            writer.writerow(['Reward', 'Throughput', 'Latancy', 'PRBs', 'Penalty'])
            #writer.writerow(['end', 'end', 'end', 'end', 'end', 'end'])
            for reward, throughput, latancy, prbs, penalty in zip(rewards, throughput, latancy, prbs, penalty):
                writer.writerow([reward, throughput, latancy, prbs, penalty])

def signal_handler(sig, frame):
    append_reward_to_csv('/hodata/rewards.csv', rewards_G, throughput_G, latancy_G, prbs_G, penalty_G)
    sys.exit(0)

class xAppRunner(Runner):
    def __init__(self, config):
        super(xAppRunner, self).__init__(config)
        self.prev_n_agents = 0
    
    def _init_buffer(self):
        if self.num_agents == 0:
            self.buffer = None
            return

        agent_obs_space = self.envs.observation_space[0]
        agent_action_space = self.envs.action_space[0]
        agent_share_obs_space = self.envs.share_observation_space[0]
        
        self.buffer = SharedReplayBuffer(self.all_args, 
                                    self.num_agents,
                                    agent_obs_space,
                                    agent_share_obs_space,
                                    agent_action_space,
                                    "ran")
        # print(f"buffer init {self.num_agents} agents")
    def run(self):
        self.warmup()   

        start_time = time.time()
        total_steps = 0
        segment_step = 0 

        signal.signal(signal.SIGINT, signal_handler)
        
        while total_steps < self.num_env_steps:
            if self.use_linear_lr_decay:
                frac = total_steps / float(self.num_env_steps) if self.num_env_steps > 0 else 0.0
                if hasattr(self.trainer, 'policy') and hasattr(self.trainer.policy, 'lr_decay'):
                    self.trainer.policy.lr_decay(frac)

            if self.num_agents == 0:
                self.warmup()
                continue

            num_agents_for_collect = self.num_agents 

            if hasattr(self.trainer.policy, 'num_agents'):
                self.trainer.policy.num_agents = self.num_agents
            if hasattr(self.trainer, 'num_agents'):
                self.trainer.num_agents = self.num_agents

            values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(segment_step)            
            obs, rewards, dones, infos, t, d, b, p = self.envs.step(actions_env)      

            reward_totoal = rewards.sum()
            rewards_G.append(reward_totoal)
            throughput_G.append(t)
            latancy_G.append(d)
            prbs_G.append(b)
            penalty_G.append(p)

            real_n_agents_after_step = obs.shape[1] if obs.ndim == 3 else obs.shape[0] if obs.ndim == 2 else 0
            
            agent_count_changed = (real_n_agents_after_step != num_agents_for_collect)

            if not agent_count_changed:
                self.insert((obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic))
                segment_step += 1
                total_steps += self.n_rollout_threads
            else:
                print(f"num_agent change: {num_agents_for_collect} -> {real_n_agents_after_step} end buffer here")

                if self.buffer is not None and segment_step > 0:
                    self.buffer.masks[segment_step] = np.zeros((self.n_rollout_threads, num_agents_for_collect, 1), dtype=np.float32)

            if segment_step >= self.episode_length or agent_count_changed or total_steps >= self.num_env_steps:
                if self.buffer is None or segment_step == 0:
                    print("...")
                else:
                    self.compute()
                    train_infos = self.train()
                    print("##########self.num_agents", self.num_agents)
                    print("One Train~~~~~~~~~~~~~~~~~~~~~~~")
                    
                    x = total_steps % self.episode_length
                    print("x", x)

                    print("(total_steps - x) // self.n_rollout_threads)", (total_steps - x) // self.n_rollout_threads)
                    if ((total_steps - x) // self.n_rollout_threads) % self.save_interval == 0 or total_steps >= self.num_env_steps:
                        self.save(total_steps // self.n_rollout_threads)

                    # Logging
                    if ((total_steps - x) // self.n_rollout_threads) % self.log_interval == 0:
                        # WANDB
                        elapsed = time.time() - start_time
                        fps = int(total_steps / elapsed)
                        print(f"Steps {total_steps}/{self.num_env_steps}, FPS {fps}.")
                        if self.env_name == "ORAN":   # here should be ran
                            env_infos = {}
                            if isinstance(infos, dict):
                                infos_list = [infos]
                            else:
                                infos_list = infos
                            for aid in range(self.num_agents):     
                                # rews = [info[aid].get('individual_reward', 0) for info in infos]
                                # rews = [info.get(aid, {}).get('individual_reward', 0) for info in infos]
                                # rews = [info.get(aid, {}).get('individual_reward', 0) if isinstance(info, dict) else 0 for info in infos]
                                rews = [
                                    step_info.get(aid, {}).get('individual_reward', 0)
                                    for step_info in infos_list
                                ]
                                env_infos[f"agent{aid}/individual_rewards"] = rews
                            train_infos.update(env_infos)
                        train_infos["avg_episode_rewards"] = np.mean(self.buffer.rewards) * self.episode_length
                        self.log_train(train_infos, total_steps)
                        # PRINT LOG
                        self.log_info(train_infos, total_steps, start_time)

                if agent_count_changed:
                    self.num_agents = real_n_agents_after_step
                    self.prev_n_agents = self.num_agents
                    
                    if hasattr(self.trainer.policy, 'num_agents'):
                        self.trainer.policy.num_agents = self.num_agents
                    if hasattr(self.trainer, 'num_agents'):
                        self.trainer.num_agents = self.num_agents
                        
                    if hasattr(self.envs, 'num_ue'): 
                        self.envs.num_ue = self.num_agents 
                    if hasattr(self.envs, '_rebuild_spaces_based_on_num_ue'):
                        self.envs._rebuild_spaces_based_on_num_ue()
                
                self._init_buffer()
                
                if self.buffer is not None and agent_count_changed:
                    current_obs_for_new_segment = obs
                    if self.use_centralized_V:
                        share_obs_for_new_segment = current_obs_for_new_segment.reshape(self.n_rollout_threads, -1)
                        share_obs_for_new_segment = np.expand_dims(share_obs_for_new_segment, 1).repeat(self.num_agents, axis=1)
                    else:
                        share_obs_for_new_segment = current_obs_for_new_segment
                    
                    self.buffer.share_obs[0] = share_obs_for_new_segment.copy()
                    self.buffer.obs[0] = current_obs_for_new_segment.copy()
                
                segment_step = 0

                if self.use_eval and ((total_steps - x) // self.n_rollout_threads) % self.eval_interval == 0:
                    self.eval(total_steps)


    def warmup(self):
        # obs = self.envs.reset()
        # self.num_agents = obs.shape[1] if obs.ndim == 3 else obs.shape[0]
        self._init_buffer()
        obs,  self.num_agents = self.envs.get_all_state()
        self.prev_n_agents = self.num_agents
        
        self._init_buffer()
        if self.buffer is None: return

        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_states, rnn_states_critic \
            = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                            np.concatenate(self.buffer.obs[step]),
                            np.concatenate(self.buffer.rnn_states[step]),
                            np.concatenate(self.buffer.rnn_states_critic[step]),
                            np.concatenate(self.buffer.masks[step]))
        
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))
        
        actions_env = actions
        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

    def insert(self, data):
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data
        
        current_num_agents = obs.shape[1]

        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, current_num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(current_num_agents, axis=1)
        else:
            share_obs = obs

        self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs, values, rewards, masks)

    @torch.no_grad()
    def eval(self, total_num_steps):
        print("Evaluation...")
        eval_episode_rewards = []
        
        eval_obs = self.eval_envs.reset()
        num_agents_in_eval = eval_obs.shape[1]

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, num_agents_in_eval, self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, num_agents_in_eval, 1), dtype=np.float32)
        

        if hasattr(self.trainer.policy, 'num_agents'):
            self.trainer.policy.num_agents = num_agents_in_eval
        if hasattr(self.trainer, 'num_agents'):
            self.trainer.num_agents = num_agents_in_eval
        

        for eval_step in range(self.episode_length):
            if num_agents_in_eval == 0:
                break
            current_num_agents = eval_obs.shape[1]
            agent_count_changed = (current_num_agents != num_agents_in_eval)
            
            if agent_count_changed:
                print(f"Eval: num_agent change: {num_agents_in_eval} -> {current_num_agents}")
                
                prev_num_agents_in_eval = num_agents_in_eval
                num_agents_in_eval = current_num_agents
                
                self.num_agents = num_agents_in_eval
                self.prev_n_agents = self.num_agents
                
                if hasattr(self.trainer.policy, 'num_agents'):
                    self.trainer.policy.num_agents = self.num_agents
                if hasattr(self.trainer, 'num_agents'):
                    self.trainer.num_agents = self.num_agents
                    
                if num_agents_in_eval == 0:
                    print("All agents removed during evaluation, ending evaluation...")
                    break
                elif num_agents_in_eval < prev_num_agents_in_eval:
                    eval_rnn_states = eval_rnn_states[:, :num_agents_in_eval, ...]
                elif num_agents_in_eval > prev_num_agents_in_eval:
                    new_eval_rnn_states = np.zeros((self.n_eval_rollout_threads, num_agents_in_eval, self.recurrent_N, self.hidden_size), dtype=np.float32)
                    new_eval_rnn_states[:, :prev_num_agents_in_eval, ...] = eval_rnn_states
                    eval_rnn_states = new_eval_rnn_states
                
                eval_masks = np.ones((self.n_eval_rollout_threads, num_agents_in_eval, 1), dtype=np.float32)

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
                                                deterministic=True)
            eval_actions = np.array(np.split(_t2n(eval_action), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states_new), self.n_eval_rollout_threads))

            eval_obs, eval_rewards, eval_dones, eval_infos, _, _, _, _ = self.eval_envs.step(eval_actions)

            eval_episode_rewards.append(eval_rewards)
            
            eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            
            eval_masks = np.ones((self.n_eval_rollout_threads, num_agents_in_eval, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards, dtype=object)
        valid_rewards = [r for r in eval_episode_rewards if r.size > 0]
        if len(valid_rewards) > 0:
            summed_rewards = np.sum(np.concatenate(valid_rewards, axis=1), axis=0)
            eval_average_episode_rewards = np.mean(summed_rewards)
            print(f"eval_average_episode_rewards: {eval_average_episode_rewards}")
            eval_env_infos = {'eval_average_episode_rewards': [eval_average_episode_rewards]}
            self.log_env(eval_env_infos, total_num_steps)


    def log_info(self, train_infos, total_steps, start_time):
        elapsed = time.time() - start_time
        fps = int(total_steps / elapsed)
        print(f" {total_steps}/{self.num_env_steps},FPS {fps}.")
        train_infos["avg_episode_rewards"] = np.mean(self.buffer.rewards) * self.episode_length
        self.log_train(train_infos, total_steps)
