import time
import os

import imageio
import numpy as np
from itertools import chain
import torch

from utils.util import update_linear_schedule
from runner.separated.base_runner import Runner


def _t2n(x):
    return x.detach().cpu().numpy()


# ========== 新增：检测指标跟踪器 ==========
class DetectionMetrics:
    """
    滑动窗口的检测指标计算器
    解决原有代码中每step计算导致的震荡问题
    """
    def __init__(self, window_size=100):
        """
        Args:
            window_size: 滑动窗口大小（step数），建议设为log_interval的整数倍
        """
        self.window_size = window_size
        self.reset()
    
    def reset(self):
        """重置所有累积指标"""
        self.tp_buffer = []
        self.fp_buffer = []
        self.fn_buffer = []
        self.tn_buffer = []
    
    def update(self, tp, fp, fn, tn):
        """
        更新缓冲区
        Args:
            tp, fp, fn, tn: 当前step的统计量（标量）
        """
        self.tp_buffer.append(tp)
        self.fp_buffer.append(fp)
        self.fn_buffer.append(fn)
        self.tn_buffer.append(tn)
        
        # 保持窗口大小
        if len(self.tp_buffer) > self.window_size:
            self.tp_buffer.pop(0)
            self.fp_buffer.pop(0)
            self.fn_buffer.pop(0)
            self.tn_buffer.pop(0)
    
    def compute(self):
        """
        计算滑动窗口内的全局指标
        Returns:
            dict: 包含precision, recall, f1, accuracy等指标
        """
        if len(self.tp_buffer) == 0:
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'accuracy': 0.0,
                'total_samples': 0,
                'attack_samples': 0,
            }
        
        total_tp = sum(self.tp_buffer)
        total_fp = sum(self.fp_buffer)
        total_fn = sum(self.fn_buffer)
        total_tn = sum(self.tn_buffer)
        
        # ✅ 修复：全局累积计算，避免除零
        if total_tp + total_fp > 0:
            precision = total_tp / (total_tp + total_fp)
        else:
            precision = 0.0  # 无预测为正样本时，精确率未定义，用0
        
        if total_tp + total_fn > 0:
            recall = total_tp / (total_tp + total_fn)
        else:
            recall = 0.0  # ✅ 修复：无真实正样本时，召回率用0而非1.0
        
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        
        # 准确率
        total_samples = total_tp + total_fp + total_fn + total_tn
        if total_samples > 0:
            accuracy = (total_tp + total_tn) / total_samples
        else:
            accuracy = 0.0
        
        # 攻击样本占比
        attack_samples = total_tp + total_fn
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'total_samples': int(total_samples),
            'attack_samples': int(attack_samples),
            'attack_ratio': attack_samples / max(1, total_samples),
            # 调试用：原始计数
            'tp': int(total_tp),
            'fp': int(total_fp),
            'fn': int(total_fn),
            'tn': int(total_tn),
        }


class EnvRunner(Runner):
    def __init__(self, config):
        super(EnvRunner, self).__init__(config)
        
        # ========== 新增：初始化检测指标跟踪器 ==========
        if self.env_name == 'Stage2Env' or self.env_name=="Stage2EnvISAC":
            # 窗口大小设为log_interval的2倍，确保有足够样本
            window_size = max(100, self.log_interval * self.episode_length * 2)
            self.det_metrics = DetectionMetrics(window_size=window_size)
            # print(f"[DetectionMetrics] Initialized with window_size={window_size}")

    def run(self):
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            success = 0
            col = 0
            timeout = 0
            
            if self.use_linear_lr_decay:
                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                (
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                    actions_env,
                ) = self.collect(step)

                # Observe reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions_env)
                
                # ========== 原有的成功率统计（针对MPE/MyEnv）==========
                if self.env_name == "MyEnv":
                    for i, done_list in enumerate(dones):
                        if done_list[0]:
                            if infos[i][0]["term_reason"] == "success":
                                success += 1
                            elif infos[i][0]["term_reason"] == "collision":
                                col += 1
                            elif infos[i][0]["term_reason"] == "timeout":
                                timeout += 1
                
                # ========== 新增：累积检测指标（针对Stage2Env）==========
                if self.env_name == 'Stage2Env' or self.env_name=="Stage2EnvISAC":
                    tp = 0
                    fp = 0
                    fn = 0
                    tn = 0
                    for info in infos:
                        # 安全获取指标（防止key不存在）
                        if "det_tp_fp_fn_tn" in info[0]:
                            tp += info[0]["det_tp_fp_fn_tn"][0]
                            fp += info[0]["det_tp_fp_fn_tn"][1]
                            fn += info[0]["det_tp_fp_fn_tn"][2]
                            tn += info[0]["det_tp_fp_fn_tn"][3]
                    
                    # 更新到滑动窗口
                    self.det_metrics.update(tp, fp, fn, tn)
                
                data = (
                    obs,
                    rewards,
                    dones,
                    infos,
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                )

                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()

            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            # save model
            if episode % self.save_interval == 0 or episode == episodes - 1:
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print(
                    "\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n".format(
                        self.all_args.scenario_name,
                        self.algorithm_name,
                        self.experiment_name,
                        episode,
                        episodes,
                        total_num_steps,
                        self.num_env_steps,
                        int(total_num_steps / (end - start)),
                    )
                )

                # ========== MPE/MyEnv的原有统计 ==========
                if self.env_name == "MPE" or self.env_name == "MyEnv":
                    train_infos[0].update({
                        "success_rate": success * 1.0 / max(1, (success + col + timeout)),
                        "collision_rate": col * 1.0 / max(1, (success + col + timeout)),
                        "timeout_rate": timeout * 1.0 / max(1, (success + col + timeout)),
                    })
                    for agent_id in range(self.num_agents):
                        train_infos[agent_id].update(
                            {
                                "average_episode_rewards": np.mean(self.buffer[agent_id].rewards)
                                * self.episode_length
                            }
                        )
                
                # ========== Stage2Env的检测指标统计 ==========
                if self.env_name == 'Stage2Env' or  self.env_name=="Stage2EnvISAC":
                    # 计算滑动窗口内的全局指标
                    det_stats = self.det_metrics.compute()
                    
                    # # 打印详细统计（方便调试）
                    # print(
                    #     f"[DetectionMetrics] "
                    #     f"Precision: {det_stats['precision']:.4f}, "
                    #     f"Recall: {det_stats['recall']:.4f}, "
                    #     f"F1: {det_stats['f1']:.4f}, "
                    #     f"Accuracy: {det_stats['accuracy']:.4f}, "
                    #     f"Attack Ratio: {det_stats['attack_ratio']:.4f} "
                    #     f"(TP={det_stats['tp']}, FP={det_stats['fp']}, "
                    #     f"FN={det_stats['fn']}, TN={det_stats['tn']})"
                    # )
                    
                    for agent_id in range(self.num_agents):
                        train_infos[agent_id].update(
                            {
                                # 基础奖励
                                "average_episode_rewards": np.mean(self.buffer[agent_id].rewards)
                                * self.episode_length,
                                # ✅ 修复后的检测指标（全局累积）
                                "det_precision": det_stats['precision'],
                                "det_recall": det_stats['recall'],
                                "f1": det_stats['f1'],
                                # 额外指标
                                "det_accuracy": det_stats['accuracy'],
                                "det_attack_ratio": det_stats['attack_ratio'],
                                # 原始计数（用于调试）
                                "det_tp": det_stats['tp'],
                                "det_fp": det_stats['fp'],
                                "det_fn": det_stats['fn'],
                                "det_tn": det_stats['tn'],
                            }
                        )
                    
                    # ========== 可选：定期重置窗口（避免过度平滑）==========
                    # 如果想每隔N个log_interval重置一次，取消下面的注释
                    # reset_interval = 10  # 每10个log_interval重置
                    # if episode % (self.log_interval * reset_interval) == 0 and episode > 0:
                    #     print(f"[DetectionMetrics] Resetting window at episode {episode}")
                    #     self.det_metrics.reset()
                
                self.log_train(train_infos, episode)
            
            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs = self.envs.reset()  # shape = [env_num, agent_num, obs_dim]

        share_obs = []
        for o in obs:
            share_obs.append(list(chain(*o)))
        share_obs = np.array(share_obs)  # shape = [env_num, agent_num * obs_dim]

        for agent_id in range(self.num_agents):
            if not self.use_centralized_V:
                share_obs = np.array(list(obs[:, agent_id]))
            self.buffer[agent_id].share_obs[0] = share_obs.copy()
            self.buffer[agent_id].obs[0] = np.array(list(obs[:, agent_id])).copy()

    @torch.no_grad()
    def collect(self, step):
        values = []
        actions = []
        temp_actions_env = []
        action_log_probs = []
        rnn_states = []
        rnn_states_critic = []

        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            value, action, action_log_prob, rnn_state, rnn_state_critic = self.trainer[
                agent_id
            ].policy.get_actions(
                self.buffer[agent_id].share_obs[step],
                self.buffer[agent_id].obs[step],
                self.buffer[agent_id].rnn_states[step],
                self.buffer[agent_id].rnn_states_critic[step],
                self.buffer[agent_id].masks[step],
            )
            # [agents, envs, dim]
            values.append(_t2n(value))
            action = _t2n(action)
            # rearrange action
            if self.envs.action_space[agent_id].__class__.__name__ == "MultiDiscrete":
                for i in range(self.envs.action_space[agent_id].shape):
                    uc_action_env = np.eye(self.envs.action_space[agent_id].high[i] + 1)[action[:, i]]
                    if i == 0:
                        action_env = uc_action_env
                    else:
                        action_env = np.concatenate((action_env, uc_action_env), axis=1)
            elif self.envs.action_space[agent_id].__class__.__name__ == "Discrete":
                action_env = np.squeeze(np.eye(self.envs.action_space[agent_id].n)[action], 1)
            else:
                # TODO 这里改造成自己环境需要的形式即可
                # TODO Here, you can change the action_env to the form you need
                action_env = action
                # raise NotImplementedError

            actions.append(action)
            temp_actions_env.append(action_env)
            action_log_probs.append(_t2n(action_log_prob))
            rnn_states.append(_t2n(rnn_state))
            rnn_states_critic.append(_t2n(rnn_state_critic))

        # [envs, agents, dim]
        actions_env = []
        for i in range(self.n_rollout_threads):
            one_hot_action_env = []
            for temp_action_env in temp_actions_env:
                one_hot_action_env.append(temp_action_env[i])
            actions_env.append(one_hot_action_env)

        values = np.array(values).transpose(1, 0, 2)
        actions = np.array(actions).transpose(1, 0, 2)
        action_log_probs = np.array(action_log_probs).transpose(1, 0, 2)
        rnn_states = np.array(rnn_states).transpose(1, 0, 2, 3)
        rnn_states_critic = np.array(rnn_states_critic).transpose(1, 0, 2, 3)

        return (
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
            actions_env,
        )

    def insert(self, data):
        (
            obs,
            rewards,
            dones,
            infos,
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
        ) = data

        rnn_states[dones == True] = np.zeros(
            ((dones == True).sum(), self.recurrent_N, self.hidden_size),
            dtype=np.float32,
        )
        rnn_states_critic[dones == True] = np.zeros(
            ((dones == True).sum(), self.recurrent_N, self.hidden_size),
            dtype=np.float32,
        )
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        share_obs = []
        for o in obs:
            share_obs.append(list(chain(*o)))
        share_obs = np.array(share_obs)

        for agent_id in range(self.num_agents):
            if not self.use_centralized_V:
                share_obs = np.array(list(obs[:, agent_id]))

            self.buffer[agent_id].insert(
                share_obs,
                np.array(list(obs[:, agent_id])),
                rnn_states[:, agent_id],
                rnn_states_critic[:, agent_id],
                actions[:, agent_id],
                action_log_probs[:, agent_id],
                values[:, agent_id],
                rewards[:, agent_id],
                masks[:, agent_id],
            )

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_rewards = []
        eval_obs = self.eval_envs.reset()

        eval_rnn_states = np.zeros(
            (
                self.n_eval_rollout_threads,
                self.num_agents,
                self.recurrent_N,
                self.hidden_size,
            ),
            dtype=np.float32,
        )
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        # ========== 新增：评估时也统计检测指标 ==========
        if self.env_name == 'Stage2Env'  or self.env_name=="Stage2EnvISAC":
            eval_det_metrics = DetectionMetrics(window_size=10000)  # 大窗口，累积整个eval

        for eval_step in range(self.episode_length):
            eval_temp_actions_env = []
            for agent_id in range(self.num_agents):
                self.trainer[agent_id].prep_rollout()
                eval_action, eval_rnn_state = self.trainer[agent_id].policy.act(
                    np.array(list(eval_obs[:, agent_id])),
                    eval_rnn_states[:, agent_id],
                    eval_masks[:, agent_id],
                    deterministic=True,
                )

                eval_action = eval_action.detach().cpu().numpy()
                # rearrange action
                if self.eval_envs.action_space[agent_id].__class__.__name__ == "MultiDiscrete":
                    for i in range(self.eval_envs.action_space[agent_id].shape):
                        eval_uc_action_env = np.eye(self.eval_envs.action_space[agent_id].high[i] + 1)[
                            eval_action[:, i]
                        ]
                        if i == 0:
                            eval_action_env = eval_uc_action_env
                        else:
                            eval_action_env = np.concatenate((eval_action_env, eval_uc_action_env), axis=1)
                elif self.eval_envs.action_space[agent_id].__class__.__name__ == "Discrete":
                    eval_action_env = np.squeeze(
                        np.eye(self.eval_envs.action_space[agent_id].n)[eval_action], 1
                    )
                else:
                    raise NotImplementedError

                eval_temp_actions_env.append(eval_action_env)
                eval_rnn_states[:, agent_id] = _t2n(eval_rnn_state)

            # [envs, agents, dim]
            eval_actions_env = []
            for i in range(self.n_eval_rollout_threads):
                eval_one_hot_action_env = []
                for eval_temp_action_env in eval_temp_actions_env:
                    eval_one_hot_action_env.append(eval_temp_action_env[i])
                eval_actions_env.append(eval_one_hot_action_env)

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
            eval_episode_rewards.append(eval_rewards)

            # ========== 新增：累积eval检测指标 ==========
            if self.env_name == 'Stage2Env'  or self.env_name=="Stage2EnvISAC":
                tp, fp, fn, tn = 0, 0, 0, 0
                for info in eval_infos:
                    if "det_tp_fp_fn_tn" in info[0]:
                        tp += info[0]["det_tp_fp_fn_tn"][0]
                        fp += info[0]["det_tp_fp_fn_tn"][1]
                        fn += info[0]["det_tp_fp_fn_tn"][2]
                        tn += info[0]["det_tp_fp_fn_tn"][3]
                eval_det_metrics.update(tp, fp, fn, tn)

            eval_rnn_states[eval_dones == True] = np.zeros(
                ((eval_dones == True).sum(), self.recurrent_N, self.hidden_size),
                dtype=np.float32,
            )
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards)

        eval_train_infos = []
        for agent_id in range(self.num_agents):
            eval_average_episode_rewards = np.mean(np.sum(eval_episode_rewards[:, :, agent_id], axis=0))
            info_dict = {"eval_average_episode_rewards": eval_average_episode_rewards}
            
            # ========== 新增：记录eval检测指标 ==========
            if self.env_name == 'Stage2Env'  or self.env_name=="Stage2EnvISAC":
                eval_det_stats = eval_det_metrics.compute()
                info_dict.update({
                    "eval_det_precision": eval_det_stats['precision'],
                    "eval_det_recall": eval_det_stats['recall'],
                    "eval_f1": eval_det_stats['f1'],
                    "eval_det_accuracy": eval_det_stats['accuracy'],
                })
                if agent_id == 0:  # 只打印一次
                    print(f"[Eval] Precision: {eval_det_stats['precision']:.4f}, "
                          f"Recall: {eval_det_stats['recall']:.4f}, "
                          f"F1: {eval_det_stats['f1']:.4f}")
            
            eval_train_infos.append(info_dict)
            print("eval average episode rewards of agent%i: " % agent_id + str(eval_average_episode_rewards))

        self.log_train(eval_train_infos, total_num_steps)

    @torch.no_grad()
    def render(self):
        # 渲染函数保持不变
        all_frames = []
        for episode in range(self.all_args.render_episodes):
            episode_rewards = []
            obs = self.envs.reset()
            if self.all_args.save_gifs:
                image = self.envs.render("rgb_array")[0][0]
                all_frames.append(image)

            rnn_states = np.zeros(
                (
                    self.n_rollout_threads,
                    self.num_agents,
                    self.recurrent_N,
                    self.hidden_size,
                ),
                dtype=np.float32,
            )
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

            for step in range(self.episode_length):
                calc_start = time.time()

                temp_actions_env = []
                for agent_id in range(self.num_agents):
                    if not self.use_centralized_V:
                        share_obs = np.array(list(obs[:, agent_id]))
                    self.trainer[agent_id].prep_rollout()
                    action, rnn_state = self.trainer[agent_id].policy.act(
                        np.array(list(obs[:, agent_id])),
                        rnn_states[:, agent_id],
                        masks[:, agent_id],
                        deterministic=True,
                    )

                    action = action.detach().cpu().numpy()
                    # rearrange action
                    if self.envs.action_space[agent_id].__class__.__name__ == "MultiDiscrete":
                        for i in range(self.envs.action_space[agent_id].shape):
                            uc_action_env = np.eye(self.envs.action_space[agent_id].high[i] + 1)[action[:, i]]
                            if i == 0:
                                action_env = uc_action_env
                            else:
                                action_env = np.concatenate((action_env, uc_action_env), axis=1)
                    elif self.envs.action_space[agent_id].__class__.__name__ == "Discrete":
                        action_env = np.squeeze(np.eye(self.envs.action_space[agent_id].n)[action], 1)
                    else:
                        raise NotImplementedError

                    temp_actions_env.append(action_env)
                    rnn_states[:, agent_id] = _t2n(rnn_state)

                # [envs, agents, dim]
                actions_env = []
                for i in range(self.n_rollout_threads):
                    one_hot_action_env = []
                    for temp_action_env in temp_actions_env:
                        one_hot_action_env.append(temp_action_env[i])
                    actions_env.append(one_hot_action_env)

                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions_env)
                episode_rewards.append(rewards)

                rnn_states[dones == True] = np.zeros(
                    ((dones == True).sum(), self.recurrent_N, self.hidden_size),
                    dtype=np.float32,
                )
                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                if self.all_args.save_gifs:
                    image = self.envs.render("rgb_array")[0][0]
                    all_frames.append(image)
                    calc_end = time.time()
                    elapsed = calc_end - calc_start
                    if elapsed < self.all_args.ifi:
                        time.sleep(self.all_args.ifi - elapsed)

            episode_rewards = np.array(episode_rewards)
            for agent_id in range(self.num_agents):
                average_episode_rewards = np.mean(np.sum(episode_rewards[:, :, agent_id], axis=0))
                print("eval average episode rewards of agent%i: " % agent_id + str(average_episode_rewards))

        if self.all_args.save_gifs:
            imageio.mimsave(
                str(self.gif_dir) + "/render.gif",
                all_frames,
                duration=self.all_args.ifi,
            )
