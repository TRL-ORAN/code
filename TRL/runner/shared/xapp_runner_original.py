import time
import numpy as np
import torch
from trl.runner.shared.base_runner import Runner
from trl.utils.shared_buffer import SharedReplayBuffer
import wandb
import imageio

def _t2n(x):
    return x.detach().cpu().numpy()

class xAppRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for the MPEs. See parent class for details."""
    def __init__(self, config):
        super(xAppRunner, self).__init__(config)

    def _init_buffer(self):    # 
        agent_obs_space = self.envs.observation_space[0]
        agent_action_space = self.envs.action_space[0] # continues
        agent_share_obs_space = self.envs.share_observation_space[0]
        self.buffer = SharedReplayBuffer(self.all_args, 
                                    self.num_agents,
                                    agent_obs_space,
                                    agent_share_obs_space,
                                    agent_action_space,
                                    "ran")
        print(":", self.buffer.share_obs[0].shape)

    def run(self):
        self.warmup()   

        start_time = time.time()
        total_steps = 0
        
        self.prev_n_agents = self.num_agents
        segment_step = 0 
        
        while total_steps < self.num_env_steps:
            if self.use_linear_lr_decay:
                frac = total_steps / float(self.num_env_steps) if self.num_env_steps > 0 else 0.0
                if hasattr(self.trainer, 'policy') and hasattr(self.trainer.policy, 'lr_decay'):
                    self.trainer.policy.lr_decay(frac)

            num_agents_for_collect = self.num_agents 
            values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(segment_step)
            
            obs, rewards, dones, infos = self.envs.step(actions_env)   
            real_n_agents_after_step = obs.shape[1] if obs.ndim == 3 and obs.shape[0] == self.n_rollout_threads else 0
            if obs.ndim == 2 and self.n_rollout_threads == 1 : 
                 real_n_agents_after_step = obs.shape[0]

            if real_n_agents_after_step == num_agents_for_collect:
                self.insert((obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic))
                segment_step += 1
                total_steps += self.n_rollout_threads
            else:
                print(f"Agent count changed: {num_agents_for_collect} -> {real_n_agents_after_step}. Ending segment.")
                self.num_agents = real_n_agents_after_step
                # self._init_buffer()
                if segment_step > 0 and self.buffer is not None:
                    if hasattr(self.buffer, 'masks') and self.buffer.masks is not None and \
                       self.buffer.masks.shape[0] > segment_step and \
                       self.buffer.masks.shape[1] == self.n_rollout_threads and \
                       self.buffer.masks.shape[2] == num_agents_for_collect:
                        self.buffer.masks[segment_step] = np.zeros((self.n_rollout_threads, num_agents_for_collect, 1), dtype=np.float32)

            if segment_step >= self.episode_length or real_n_agents_after_step != self.prev_n_agents or total_steps >= self.num_env_steps:

                self.prev_n_agents = real_n_agents_after_step
                self.num_agents = real_n_agents_after_step 
                if hasattr(self.envs, 'num_ue'): 
                    self.envs.num_ue = self.num_agents 
                
                if hasattr(self.envs, '_rebuild_spaces_based_on_num_ue'):
                    self.envs._rebuild_spaces_based_on_num_ue()
                else:
                    pass 

                self.compute()
                train_infos = self.train()

                print("one time train...")

                if (total_steps // self.n_rollout_threads) % self.save_interval == 0 or total_steps >= self.num_env_steps:
                    self.save()

                # Logging
                if (total_steps // self.n_rollout_threads) % self.log_interval == 0:
                    elapsed = time.time() - start_time
                    fps = int(total_steps / elapsed)
                    print(f"Steps {total_steps}/{self.num_env_steps}, FPS {fps}.")
                    if self.env_name == "ran":   # here should be ran
                        env_infos = {}
                        if isinstance(infos, dict):
                            infos_list = [infos]
                        else:
                            infos_list = infos
                        for aid in range(self.num_agents):     
                            rews = [
                                step_info.get(aid, {}).get('individual_reward', 0)
                                for step_info in infos_list
                            ]
                            env_infos[f"agent{aid}/individual_rewards"] = rews
                        train_infos.update(env_infos)
                    train_infos["avg_episode_rewards"] = np.mean(self.buffer.rewards) * self.episode_length
                    self.log_train(train_infos, total_steps)
                    if self.env_name == "ran":  # here should be ran
                        self.log_env(env_infos, total_steps)

                self._init_buffer() 
                print("self.buffer.share_obs[0]:", self.buffer.share_obs[0].shape)
                current_obs_for_new_segment = obs 
                
                if self.use_centralized_V:
                    share_obs_for_new_segment_flat = current_obs_for_new_segment.reshape(self.n_rollout_threads, -1)
                    share_obs_for_new_segment = np.expand_dims(share_obs_for_new_segment_flat, 1).repeat(self.num_agents, axis=1)
                else:
                    share_obs_for_new_segment = current_obs_for_new_segment
                
                self.buffer.share_obs[0] = share_obs_for_new_segment.copy()
                self.buffer.obs[0] = current_obs_for_new_segment.copy()
                
                if hasattr(self.buffer, 'rnn_states') and self.buffer.rnn_states is not None and self.buffer.rnn_states.size > 0 :
                    self.buffer.rnn_states[0].fill(0)
                    self.buffer.rnn_states_critic[0].fill(0)
                    self.buffer.masks[0].fill(1)

                segment_step = 0

                if self.use_eval and total_steps % self.eval_interval == 0:
                    self.eval(total_steps)
        

    def warmup(self):
        # reset env
        self._init_buffer()
        obs,  self.num_agents = self.envs.get_all_state()

        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        self.buffer.share_obs[0] = share_obs.copy()
        print("self.buffer.share_obs[0]:", self.buffer.share_obs[0].shape)
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
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))
        # rearrange action
        actions_env = actions

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

    def insert(self, data):
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs, values, rewards, masks)

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_rewards = []
        eval_obs = self.eval_envs.reset()
        if self.use_centralized_V:
            eval_share_obs = eval_obs.reshape(self.n_eval_rollout_threads, -1)
            eval_share_obs = np.expand_dims(eval_share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            eval_share_obs = eval_obs

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        for eval_step in range(self.episode_length):
            self.trainer.prep_rollout()
            eval_action, eval_rnn_states = self.trainer.policy.act(
                                                np.concatenate(eval_share_obs),
                                                np.concatenate(eval_obs),
                                                np.concatenate(eval_rnn_states),
                                                np.concatenate(eval_masks),
                                                deterministic=True)
            eval_actions = np.array(np.split(_t2n(eval_action), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))
            
            eval_actions_env = eval_actions

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards)
        eval_env_infos = {}
        eval_env_infos['eval_average_episode_rewards'] = np.sum(np.array(eval_episode_rewards), axis=0)
        eval_average_episode_rewards = np.mean(eval_env_infos['eval_average_episode_rewards'])
        print("eval average episode rewards of agent: " + str(eval_average_episode_rewards))
        self.log_env(eval_env_infos, total_num_steps)

    @torch.no_grad()
    def render(self):
        """Visualize the env."""
        envs = self.envs
        
        all_frames = []
        for episode in range(self.all_args.render_episodes):
            obs = envs.reset()
            if self.all_args.save_gifs:
                image = envs.render('rgb_array')[0][0]
                all_frames.append(image)
            else:
                envs.render('human')

            rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
            
            episode_rewards = []
            
            for step in range(self.episode_length):
                calc_start = time.time()

                self.trainer.prep_rollout()
                action, rnn_states = self.trainer.policy.act(np.concatenate(obs),
                                                    np.concatenate(rnn_states),
                                                    np.concatenate(masks),
                                                    deterministic=True)
                actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
                rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))

                if envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
                    for i in range(envs.action_space[0].shape):
                        uc_actions_env = np.eye(envs.action_space[0].high[i]+1)[actions[:, :, i]]
                        if i == 0:
                            actions_env = uc_actions_env
                        else:
                            actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
                elif envs.action_space[0].__class__.__name__ == 'Discrete':
                    actions_env = np.squeeze(np.eye(envs.action_space[0].n)[actions], 2)
                else:
                    raise NotImplementedError

                # Obser reward and next obs
                obs, rewards, dones, infos = envs.step(actions_env)
                episode_rewards.append(rewards)

                rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                if self.all_args.save_gifs:
                    image = envs.render('rgb_array')[0][0]
                    all_frames.append(image)
                    calc_end = time.time()
                    elapsed = calc_end - calc_start
                    if elapsed < self.all_args.ifi:
                        time.sleep(self.all_args.ifi - elapsed)
                else:
                    envs.render('human')

            print("average episode rewards is: " + str(np.mean(np.sum(np.array(episode_rewards), axis=0))))

        if self.all_args.save_gifs:
            imageio.mimsave(str(self.gif_dir) + '/render.gif', all_frames, duration=self.all_args.ifi)
