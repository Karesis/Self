import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import argparse
import copy
from tqdm import tqdm
import logging
import json
import threading
from queue import Queue
from torch.utils.tensorboard import SummaryWriter

from gomokuenv import GomokuEnv, RandomAgent
from model import GomokuModel, GomokuAgent

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("GomokuTrainer")

class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity=10000):
        """初始化缓冲区
        
        Args:
            capacity: 缓冲区容量
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done, info=None):
        """添加一条经验"""
        if info is None:
            info = {}
        self.buffer.append((state, action, reward, next_state, done, info))
    
    def sample(self, batch_size):
        """随机采样一批经验"""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        states, actions, rewards, next_states, dones, infos = zip(*random.sample(self.buffer, batch_size))
        return states, actions, rewards, next_states, dones, infos
    
    def __len__(self):
        """返回缓冲区中的经验数量"""
        return len(self.buffer)


class RewardDiffuser:
    """奖励扩散处理器"""
    
    def __init__(self, diffusion_rate=0.9, horizon=20):
        """初始化奖励扩散器
        
        Args:
            diffusion_rate: 扩散衰减率
            horizon: 扩散视野长度
        """
        self.diffusion_rate = diffusion_rate
        self.horizon = horizon
    
    def diffuse_rewards(self, raw_rewards, strategy_params=None):
        """计算扩散奖励
        
        Args:
            raw_rewards: 原始奖励序列 [r1, r2, ..., rT]
            strategy_params: 可选的策略参数序列 [(alpha1, beta1), ...]
        
        Returns:
            扩散后的奖励序列
        """
        T = len(raw_rewards)
        diffused_rewards = np.zeros_like(raw_rewards, dtype=np.float32)
        
        if strategy_params is None:
            # 使用固定扩散率
            diffusion_rates = [self.diffusion_rate] * T
            horizons = [self.horizon] * T
        else:
            # 根据策略参数动态调整扩散率和视野
            diffusion_rates = []
            horizons = []
            
            for alpha, beta in strategy_params:
                # beta越高(急迫度高)，扩散率越低，视野越短
                # alpha越高(信息场权重高)，扩散率越高
                rate = 0.7 + 0.2 * alpha - 0.1 * beta
                horizon = int(self.horizon * (1 - 0.5 * beta))
                
                diffusion_rates.append(rate)
                horizons.append(horizon)
        
        # 前向扩散（当前决策影响未来）
        for t in range(T):
            # 当前步的原始奖励
            diffused_rewards[t] += raw_rewards[t]
            
            # 未来步骤对当前的影响
            for h in range(1, min(horizons[t], T-t)):
                diffused_rewards[t] += raw_rewards[t+h] * (diffusion_rates[t] ** h)
        
        # 后向扩散（未来结果反馈当前）
        for t in reversed(range(T)):
            for h in range(1, min(horizons[t], t+1)):
                # t时刻的决策受到过去t-h时刻决策的影响
                diffused_rewards[t-h] += raw_rewards[t] * (diffusion_rates[t-h] ** h) * 0.5  # 半权重
        
        return diffused_rewards


class ExperienceCollector(threading.Thread):
    """线程类：用于收集经验"""
    
    def __init__(self, model, best_model, device, board_size, thread_id, episodes_to_collect, result_queue):
        """初始化线程
        
        Args:
            model: 当前模型
            best_model: 最佳模型
            device: 计算设备
            board_size: 棋盘大小
            thread_id: 线程ID
            episodes_to_collect: 要收集的局数
            result_queue: 结果队列，用于返回收集的经验
        """
        super(ExperienceCollector, self).__init__()
        self.model = model
        self.best_model = best_model
        self.device = device 
        self.board_size = board_size
        self.thread_id = thread_id
        self.episodes_to_collect = episodes_to_collect
        self.result_queue = result_queue
        
        # 设置线程随机种子
        self.rng = random.Random(42 + thread_id)
        
    def run(self):
        """线程执行函数"""
        # 使用线程特定的随机数生成器
        random.seed(42 + self.thread_id)
        np.random.seed(42 + self.thread_id)
        torch.manual_seed(42 + self.thread_id)
        
        # 创建智能体
        agent = GomokuAgent(
            model=self.model,
            device=self.device,
            name=f"CurrentModel_{self.thread_id}"
        )
        
        opponent = GomokuAgent(
            model=self.best_model,
            device=self.device,
            deterministic=True,
            name=f"BestModel_{self.thread_id}"
        )
        
        # 创建环境
        env = GomokuEnv(board_size=self.board_size)
        
        experiences = []
        episode_stats = []
        
        for _ in range(self.episodes_to_collect):
            # 随机决定谁先手
            if self.rng.random() < 0.5:
                black_agent, white_agent = agent, opponent
                is_black = True
            else:
                black_agent, white_agent = opponent, agent
                is_black = False
            
            # 记录状态、动作、奖励等
            episode_states = []
            episode_actions = []
            episode_rewards = []
            episode_dones = []
            episode_values = []
            episode_next_states = []
            episode_alpha_beta = []
            
            # 重置环境和智能体
            state = env.reset()
            black_agent.reset()
            white_agent.reset()
            
            done = False
            turn = 0
            
            # 游戏循环
            while not done:
                current_agent = black_agent if env.current_player == 1 else white_agent
                is_learning_agent = current_agent is agent
                
                # 只收集当前学习智能体的经验
                if is_learning_agent:
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    
                    # 获取历史序列特征
                    seq_features = None
                    if len(agent.feature_history) > 1:
                        seq_features = torch.FloatTensor(
                            np.concatenate(agent.feature_history, axis=0)
                        ).unsqueeze(0).to(self.device)
                        seq_features = seq_features.view(1, len(agent.feature_history), -1)
                    
                    # 记录状态
                    episode_states.append(state.copy())
                    
                    # 获取动作掩码
                    valid_moves_mask = torch.FloatTensor(
                        env.get_valid_moves_mask()
                    ).unsqueeze(0).to(self.device)
                    
                    # 获取策略参数
                    if seq_features is not None:
                        with torch.no_grad():
                            alpha, beta = self.model.strategic_params(seq_features)
                            episode_alpha_beta.append((alpha.item(), beta.item()))
                    else:
                        episode_alpha_beta.append((0.5, 0.5))
                    
                    # 使用模型获取动作
                    with torch.no_grad():
                        policy_logits, value = self.model(state_tensor, seq_features, valid_moves_mask)
                        policy = F.softmax(policy_logits, dim=1)
                        
                        # 根据策略采样动作（带探索）
                        if self.rng.random() < 0.05:  # 小概率随机探索
                            valid_indices = torch.nonzero(valid_moves_mask[0]).squeeze(1)
                            if len(valid_indices) > 0:
                                action_idx = valid_indices[self.rng.randint(0, len(valid_indices)-1)]
                            else:
                                # 如果没有有效动作（极少发生），随机选一个
                                action_idx = torch.randint(0, self.board_size * self.board_size, (1,))
                        else:
                            dist = torch.distributions.Categorical(policy)
                            action_idx = dist.sample()
                        
                        # 记录价值
                        episode_values.append(value.item())
                
                # 使用智能体接口选择动作
                action = current_agent.select_action(env)
                
                # 如果是学习智能体，记录动作
                if is_learning_agent:
                    # 将坐标转换为一维索引
                    action_idx_1d = action[0] * env.board_size + action[1]
                    episode_actions.append(action_idx_1d)
                
                # 执行动作
                next_state, reward, done, info = env.step(action)
                
                # 如果是学习智能体，记录结果
                if is_learning_agent:
                    episode_rewards.append(0.0)  # 中间步骤没有奖励
                    episode_dones.append(done)
                    episode_next_states.append(next_state.copy())
                
                # 更新状态
                state = next_state
                turn += 1
            
            # 计算最终奖励
            winner = info.get("winner", 0)
            learner_color = 1 if is_black else 2
            
            # 根据结果赋予奖励
            result_reward = 0.0
            if winner == learner_color:  # 获胜
                result_reward = 1.0
            elif winner != 0:  # 失败
                result_reward = -1.0
            
            # 更新最后一步的奖励
            if episode_rewards:
                episode_rewards[-1] = result_reward
                
                # 记录游戏统计
                episode_stats.append({
                    'reward': result_reward,
                    'length': turn,
                    'winner': winner
                })
                
                # 添加经验到局部存储
                experiences.append({
                    'states': episode_states,
                    'actions': episode_actions,
                    'rewards': episode_rewards,
                    'dones': episode_dones,
                    'values': episode_values,
                    'next_states': episode_next_states,
                    'alpha_beta': episode_alpha_beta
                })
        
        # 将收集的经验放入结果队列
        self.result_queue.put((experiences, episode_stats))


class PPOTrainer:
    """使用PPO算法和奖励扩散训练五子棋AI"""
    
    def __init__(self, 
                 model, 
                 board_size=15, 
                 lr=3e-4, 
                 gamma=0.99, 
                 clip_ratio=0.2,
                 value_coef=0.5, 
                 entropy_coef=0.01,
                 max_grad_norm=0.5,
                 diffusion_rate=0.9,
                 diffusion_horizon=20,
                 gae_lambda=0.95,
                 device="cuda" if torch.cuda.is_available() else "cpu",
                 checkpoint_dir="checkpoints",
                 tb_log_dir="runs"):
        """初始化PPO训练器
        
        Args:
            model: 要训练的模型
            board_size: 棋盘大小
            lr: 学习率
            gamma: 折扣因子
            clip_ratio: PPO裁剪率
            value_coef: 价值损失系数
            entropy_coef: 熵损失系数
            max_grad_norm: 梯度裁剪范数
            diffusion_rate: 奖励扩散率
            diffusion_horizon: 扩散视野
            gae_lambda: GAE(Generalized Advantage Estimation)参数
            device: 计算设备
            checkpoint_dir: 检查点保存目录
            tb_log_dir: TensorBoard日志目录
        """
        # 核心组件
        self.model = model.to(device)
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.reward_diffuser = RewardDiffuser(diffusion_rate, diffusion_horizon)
        
        # 超参数
        self.board_size = board_size
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.gae_lambda = gae_lambda
        self.device = device
        
        # 训练状态
        self.epochs = 0
        self.episodes = 0
        self.total_steps = 0
        self.best_score = -float("inf")
        
        # 经验缓冲区
        self.replay_buffer = ReplayBuffer(capacity=10000)
        
        # 最佳模型（用于评估和自对弈）
        self.best_model = copy.deepcopy(model)
        
        # 设置目录
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.checkpoint_dir = checkpoint_dir
        
        # TensorBoard
        self.writer = SummaryWriter(tb_log_dir)
        
        # 训练指标
        self.metrics = {
            "episode_rewards": [],
            "episode_lengths": [],
            "win_rates": [],
            "policy_losses": [],
            "value_losses": [],
            "entropy": [],
            "learning_rates": [],
            "alpha_values": [],  # 信息场权重
            "beta_values": [],   # 时间梯度权重
        }
    
    def collect_experience(self, num_episodes=10, opponent=None, silent=False, num_threads=4):
        """使用多线程收集经验数据
        
        Args:
            num_episodes: 自对弈局数
            opponent: 对手智能体，如果为None则使用自己对弈
            silent: 是否不打印进度
            num_threads: 线程数
            
        Returns:
            收集的经验数量
        """
        # 如果线程数为1，使用串行处理
        if num_threads <= 1:
            return self._collect_experience_serial(num_episodes, opponent, silent)
            
        # 为模型创建副本，确保每个线程有自己的模型副本，避免并发访问问题
        model_copies = [copy.deepcopy(self.model).to(self.device) for _ in range(num_threads)]
        best_model_copies = [copy.deepcopy(self.best_model).to(self.device) for _ in range(num_threads)]
        
        # 设置每个线程收集的局数
        episodes_per_thread = [num_episodes // num_threads] * num_threads
        # 处理余数
        for i in range(num_episodes % num_threads):
            episodes_per_thread[i] += 1
            
        # 创建结果队列
        result_queue = Queue()
        
        # 创建并启动线程
        threads = []
        for i in range(num_threads):
            thread = ExperienceCollector(
                model=model_copies[i],
                best_model=best_model_copies[i],
                device=self.device,
                board_size=self.board_size,
                thread_id=i,
                episodes_to_collect=episodes_per_thread[i],
                result_queue=result_queue
            )
            threads.append(thread)
            thread.start()
            
        if not silent:
            print(f"Collecting experience using {num_threads} threads...")
            
        # 等待所有线程完成
        for thread in tqdm(threads, desc="Waiting for threads", disable=silent):
            thread.join()
            
        # 从队列收集结果
        all_experiences = []
        all_stats = []
        
        for _ in tqdm(range(len(threads)), desc="Collecting results", disable=silent):
            experiences, stats = result_queue.get()
            all_experiences.extend(experiences)
            all_stats.extend(stats)
        
        # 处理收集到的经验
        total_collected = 0
        reward_diffuser = self.reward_diffuser  # 本地引用提高性能
        
        for exp in tqdm(all_experiences, desc="Processing experiences", disable=silent):
            # 对每局游戏分别应用奖励扩散
            diffused_rewards = reward_diffuser.diffuse_rewards(
                exp['rewards'],
                exp['alpha_beta']
            )
            
            # 计算GAE优势估计
            advantages = self._compute_advantages(
                exp['rewards'], 
                exp['values'], 
                exp['dones']
            )
            
            # 将经验添加到回放缓冲区
            for t in range(len(exp['states'])):
                self.replay_buffer.push(
                    exp['states'][t],
                    exp['actions'][t],
                    diffused_rewards[t],
                    exp['next_states'][t],
                    exp['dones'][t],
                    {
                        "value": exp['values'][t],
                        "advantage": advantages[t],
                        "alpha_beta": exp['alpha_beta'][t]
                    }
                )
                total_collected += 1
        
        # 更新度量指标
        for stat in all_stats:
            self.metrics["episode_rewards"].append(stat['reward'])
            self.metrics["episode_lengths"].append(stat['length'])
            self.episodes += 1
        
        # 清理不再需要的模型副本
        del model_copies
        del best_model_copies
        
        return total_collected
        
    def _collect_experience_serial(self, num_episodes=10, opponent=None, silent=False):
        """串行版本的经验收集（用于不支持多线程的情况）"""
        env = GomokuEnv(board_size=self.board_size)
        
        # 当前模型智能体
        agent = GomokuAgent(
            model=self.model,
            device=self.device,
            name="CurrentModel"
        )
        
        # 使用最佳模型或随机智能体作为对手
        if opponent is None:
            opponent = GomokuAgent(
                model=self.best_model,
                device=self.device,
                deterministic=True,  # 使用确定性策略增加稳定性
                name="BestModel"
            )
        
        episode_iterator = range(num_episodes)
        if not silent:
            episode_iterator = tqdm(episode_iterator, desc="Collecting experience")
        
        total_collected = 0
        
        for _ in episode_iterator:
            # 随机决定谁先手
            if random.random() < 0.5:
                black_agent, white_agent = agent, opponent
            else:
                black_agent, white_agent = opponent, agent
            
            # 记录每一步的状态、动作、奖励
            episode_states = []
            episode_actions = []
            episode_rewards = []
            episode_dones = []
            episode_values = []
            episode_next_states = []
            episode_alpha_beta = []  # 记录每步的α和β值
            
            # 重置环境和智能体
            state = env.reset()
            black_agent.reset()
            white_agent.reset()
            
            done = False
            turn = 0  # 用于记录步数
            
            # 游戏循环
            while not done:
                current_agent = black_agent if env.current_player == 1 else white_agent
                is_learning_agent = current_agent is agent
                
                # 只收集当前学习智能体的经验
                if is_learning_agent:
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    
                    # 获取历史序列特征
                    seq_features = None
                    if len(agent.feature_history) > 1:
                        seq_features = torch.FloatTensor(
                            np.concatenate(agent.feature_history, axis=0)
                        ).unsqueeze(0).to(self.device)
                        seq_features = seq_features.view(1, len(agent.feature_history), -1)
                    
                    # 获取动作前，先记录状态
                    episode_states.append(state)
                    
                    # 获取动作掩码
                    valid_moves_mask = torch.FloatTensor(
                        env.get_valid_moves_mask()
                    ).unsqueeze(0).to(self.device)
                    
                    # 获取策略参数
                    if seq_features is not None:
                        with torch.no_grad():
                            alpha, beta = self.model.strategic_params(seq_features)
                            episode_alpha_beta.append((alpha.item(), beta.item()))
                    else:
                        episode_alpha_beta.append((0.5, 0.5))
                    
                    # 使用模型获取动作
                    with torch.no_grad():
                        policy_logits, value = self.model(state_tensor, seq_features, valid_moves_mask)
                        policy = F.softmax(policy_logits, dim=1)
                        
                        # 根据策略采样动作
                        if random.random() < 0.05:  # 小概率随机探索
                            valid_indices = torch.nonzero(valid_moves_mask[0]).squeeze(1)
                            action_idx = valid_indices[random.randint(0, len(valid_indices)-1)]
                        else:
                            dist = torch.distributions.Categorical(policy)
                            action_idx = dist.sample()
                        
                        # 记录策略和值
                        episode_values.append(value.item())
                
                # 使用智能体接口选择动作
                action = current_agent.select_action(env)
                
                # 如果是学习智能体，记录选择的动作
                if is_learning_agent:
                    # 将坐标转换为一维索引
                    action_idx_1d = action[0] * env.board_size + action[1]
                    episode_actions.append(action_idx_1d)
                
                # 执行动作
                next_state, reward, done, info = env.step(action)
                
                # 如果是学习智能体，记录结果
                if is_learning_agent:
                    episode_rewards.append(0.0)  # 中间步骤没有奖励
                    episode_dones.append(done)
                    episode_next_states.append(next_state)
                
                # 更新状态
                state = next_state
                turn += 1
            
            # 游戏结束后计算奖励
            # 只对当前模型的步骤赋予奖励
            winner = info.get("winner", 0)
            
            # 获取学习智能体的颜色
            learner_color = 1 if black_agent is agent else 2
            
            # 根据结果赋予奖励
            result_reward = 0.0
            if winner == learner_color:  # 获胜
                result_reward = 1.0
            elif winner != 0:  # 失败
                result_reward = -1.0
            
            # 更新最后一步的奖励
            if episode_rewards:
                episode_rewards[-1] = result_reward
            
            # 使用动态奖励扩散
            if episode_rewards:
                diffused_rewards = self.reward_diffuser.diffuse_rewards(
                    episode_rewards,
                    episode_alpha_beta
                )
                
                # 计算GAE优势估计
                advantages = self._compute_advantages(
                    episode_rewards, 
                    episode_values, 
                    episode_dones
                )
                
                # 将经验添加到回放缓冲区
                for t in range(len(episode_states)):
                    self.replay_buffer.push(
                        episode_states[t],
                        episode_actions[t],
                        diffused_rewards[t],
                        episode_next_states[t],
                        episode_dones[t],
                        {
                            "value": episode_values[t],
                            "advantage": advantages[t],
                            "alpha_beta": episode_alpha_beta[t]
                        }
                    )
                    total_collected += 1
            
            # 记录指标
            if episode_rewards:
                self.metrics["episode_rewards"].append(result_reward)
                self.metrics["episode_lengths"].append(turn)
            
            self.episodes += 1
        
        return total_collected
    
    def _compute_advantages(self, rewards, values, dones):
        """计算GAE优势估计"""
        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            # 如果是最后一步，下一步值为0
            next_value = values[t+1] if t < len(values) - 1 else 0
            
            # 如果是终止状态，下一步值为0
            if dones[t]:
                next_value = 0
            
            # 计算TD误差
            delta = rewards[t] + self.gamma * next_value - values[t]
            
            # 计算GAE
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
        
        return advantages
    
    def update_policy(self, batch_size=128, num_updates=10, mini_batch_size=0, num_epochs=4):
        """使用PPO算法更新策略，支持批次化处理
        
        Args:
            batch_size: 总批处理大小
            num_updates: 每批数据更新次数
            mini_batch_size: 小批量大小，如果为0则使用整个batch
            num_epochs: 每个小批量的训练轮数
            
        Returns:
            平均策略损失和价值损失
        """
        if len(self.replay_buffer) < batch_size:
            logger.warning(f"Replay buffer too small ({len(self.replay_buffer)}/{batch_size}), skipping update")
            return 0, 0, 0
        
        avg_policy_loss = 0
        avg_value_loss = 0
        avg_entropy = 0
        
        # 记录累积α和β值
        cumulative_alpha = 0
        cumulative_beta = 0
        count = 0
        
        # 使用DataLoader加速批处理
        from torch.utils.data import DataLoader, TensorDataset
        
        for _ in range(num_updates):
            try:
                # 采样一批数据
                states_np, actions_np, rewards_np, _, _, infos = self.replay_buffer.sample(batch_size)
                
                # 提取优势和值估计
                advantages_np = np.array([info["advantage"] for info in infos])
                old_values_np = np.array([info["value"] for info in infos])
                
                # 提取并累加α和β值
                for info in infos:
                    alpha, beta = info.get("alpha_beta", (0.5, 0.5))
                    cumulative_alpha += alpha
                    cumulative_beta += beta
                    count += 1
                
                # 转换为张量
                states = torch.FloatTensor(np.array(states_np)).to(self.device)
                actions = torch.LongTensor(np.array(actions_np)).to(self.device)
                rewards = torch.FloatTensor(np.array(rewards_np)).to(self.device).unsqueeze(1)
                advantages = torch.FloatTensor(np.array(advantages_np)).to(self.device).unsqueeze(1)
                old_values = torch.FloatTensor(old_values_np).to(self.device).unsqueeze(1)
                
                # 标准化优势
                if advantages.shape[0] > 1:  # 至少需要2个样本才能标准化
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # 获取旧策略下的动作概率（一次性获取所有）
                with torch.no_grad():
                    old_policy_logits, _ = self.model(states)
                    old_policy = F.softmax(old_policy_logits, dim=1)
                    old_action_probs = torch.gather(old_policy, 1, actions.unsqueeze(1))
                
                # 创建数据集和加载器
                dataset = TensorDataset(states, actions, rewards, advantages, old_action_probs)
                
                # 如果mini_batch_size为0，使用整个batch
                actual_mini_batch_size = mini_batch_size if mini_batch_size > 0 else batch_size
                
                # 确保mini-batch大小不超过数据集大小
                actual_mini_batch_size = min(actual_mini_batch_size, len(dataset))
                
                # 创建DataLoader
                data_loader = DataLoader(
                    dataset,
                    batch_size=actual_mini_batch_size,
                    shuffle=True,
                    pin_memory=False,
                    num_workers=0  # 在主进程中加载数据
                )
                
                # 对每个mini-batch进行多轮训练
                for epoch_idx in range(num_epochs):
                    for mini_states, mini_actions, mini_rewards, mini_advantages, mini_old_probs in data_loader:
                        # 批量前向传播
                        policy_logits, values = self.model(mini_states)
                        
                        # 计算动作概率和熵
                        policy = F.softmax(policy_logits, dim=1)
                        dist = torch.distributions.Categorical(policy)
                        action_probs = torch.gather(policy, 1, mini_actions.unsqueeze(1))
                        entropy = dist.entropy().mean()
                        
                        # 计算概率比
                        ratio = action_probs / (mini_old_probs + 1e-8)
                        
                        # 计算裁剪后的损失
                        surr1 = ratio * mini_advantages
                        surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * mini_advantages
                        policy_loss = -torch.min(surr1, surr2).mean()
                        
                        # 计算值函数损失
                        value_loss = F.mse_loss(values, mini_rewards)
                        
                        # 计算总损失
                        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                        
                        # 梯度更新
                        self.optimizer.zero_grad()
                        loss.backward()
                        
                        # 梯度裁剪
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        
                        self.optimizer.step()
                        
                        # 累积损失（只累加最后一个epoch的损失用于记录）
                        if epoch_idx == num_epochs - 1:
                            avg_policy_loss += policy_loss.item() / (len(data_loader) * num_updates)
                            avg_value_loss += value_loss.item() / (len(data_loader) * num_updates)
                            avg_entropy += entropy.item() / (len(data_loader) * num_updates)
            
            except Exception as e:
                logger.error(f"Error during policy update: {str(e)}")
                # 继续下一次更新
                continue
        
        # 记录平均α和β值
        if count > 0:
            self.metrics["alpha_values"].append(cumulative_alpha / count)
            self.metrics["beta_values"].append(cumulative_beta / count)
        
        # 记录损失
        self.metrics["policy_losses"].append(avg_policy_loss)
        self.metrics["value_losses"].append(avg_value_loss)
        self.metrics["entropy"].append(avg_entropy)
        
        return avg_policy_loss, avg_value_loss, avg_entropy
    
    def evaluate(self, num_games=20, opponent=None, silent=False):
        """评估当前模型
        
        Args:
            num_games: 评估游戏局数
            opponent: 对手智能体，默认为随机智能体
            silent: 是否静默模式
        
        Returns:
            胜率（0-1之间）
        """
        env = GomokuEnv(board_size=self.board_size)
        
        # 创建当前模型智能体
        agent = GomokuAgent(
            model=self.model,
            device=self.device,
            deterministic=True,  # 评估时使用确定性策略
            name="CurrentModel"
        )
        
        # 创建对手
        if opponent is None:
            opponent = RandomAgent(name="RandomOpponent")
        
        wins = 0
        draws = 0
        losses = 0
        
        game_iterator = range(num_games)
        if not silent:
            game_iterator = tqdm(game_iterator, desc="Evaluating")
        
        for _ in game_iterator:
            # 随机决定谁先手
            if random.random() < 0.5:
                black_agent, white_agent = agent, opponent
                agent_color = 1
            else:
                black_agent, white_agent = opponent, agent
                agent_color = 2
            
            # 重置环境和智能体
            env.reset()
            agent.reset()
            if hasattr(opponent, 'reset'):
                opponent.reset()
            
            done = False
            
            # 游戏循环
            while not done:
                current_agent = black_agent if env.current_player == 1 else white_agent
                
                # 使用智能体选择动作
                action = current_agent.select_action(env)
                
                # 执行动作
                _, _, done, info = env.step(action)
            
            # 游戏结束后判断结果
            winner = info.get("winner", 0)
            
            if winner == agent_color:
                wins += 1
            elif winner == 0:
                draws += 1
            else:
                losses += 1
        
        win_rate = wins / num_games
        self.metrics["win_rates"].append(win_rate)
        
        # 输出评估结果
        if not silent:
            logger.info(f"Evaluation: wins={wins}, draws={draws}, losses={losses}, win_rate={win_rate:.4f}")
        
        return win_rate
    
    def update_best_model(self, win_rate, threshold=0.55):
        """如果当前模型足够好，则更新最佳模型
        
        Args:
            win_rate: 当前模型的胜率
            threshold: 更新阈值
            
        Returns:
            是否更新了最佳模型
        """
        if win_rate > max(self.best_score, threshold):
            logger.info(f"Updating best model: {self.best_score:.4f} -> {win_rate:.4f}")
            self.best_model = copy.deepcopy(self.model)
            self.best_score = win_rate
            return True
        return False
    
    def save_checkpoint(self, filename=None):
        """保存检查点
        
        Args:
            filename: 文件名，如果为None则使用当前epoch
        """
        if filename is None:
            filename = f"model_epoch_{self.epochs}.pt"
        
        path = os.path.join(self.checkpoint_dir, filename)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'best_model_state_dict': self.best_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epochs': self.epochs,
            'episodes': self.episodes,
            'total_steps': self.total_steps,
            'best_score': self.best_score,
            'metrics': self.metrics
        }, path)
        
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, filepath):
        """加载检查点
        
        Args:
            filepath: 检查点文件路径
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_model.load_state_dict(checkpoint['best_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.epochs = checkpoint['epochs']
        self.episodes = checkpoint['episodes']
        self.total_steps = checkpoint['total_steps']
        self.best_score = checkpoint['best_score']
        self.metrics = checkpoint['metrics']
        
        logger.info(f"Checkpoint loaded from {filepath}")
    
    def train(self, 
              num_epochs=1000, 
              episodes_per_epoch=10,
              updates_per_epoch=10,
              batch_size=128,
              mini_batch_size=32,
              mini_epochs=4,
              num_threads=4,
              eval_interval=5,
              save_interval=10,
              log_interval=1):
        """训练模型，支持批处理和并行计算
        
        Args:
            num_epochs: 总训练周期数
            episodes_per_epoch: 每个周期的自对弈局数
            updates_per_epoch: 每个周期的策略更新次数
            batch_size: 批处理大小
            mini_batch_size: 小批量大小，为0则使用整个batch
            mini_epochs: 每个mini-batch的训练轮数
            num_threads: 并行线程数
            eval_interval: 评估间隔（周期数）
            save_interval: 保存间隔（周期数）
            log_interval: 日志间隔（周期数）
        """
        logger.info("Starting training...")
        logger.info(f"Device: {self.device}")
        logger.info(f"Configuration: board_size={self.board_size}, threads={num_threads}, "
                   f"episodes_per_epoch={episodes_per_epoch}, batch_size={batch_size}, "
                   f"mini_batch_size={mini_batch_size}, mini_epochs={mini_epochs}")
        
        # 初始评估
        logger.info("Initial evaluation...")
        initial_win_rate = self.evaluate(num_games=10)
        self.best_score = initial_win_rate
        
        # 主训练循环
        for epoch in range(1, num_epochs + 1):
            self.epochs += 1
            start_time = time.time()
            
            # 收集自对弈数据
            logger.info(f"Epoch {epoch}/{num_epochs}: Collecting experience...")
            total_collected = self.collect_experience(
                num_episodes=episodes_per_epoch,
                num_threads=num_threads
            )
            
            # 更新策略
            logger.info(f"Epoch {epoch}/{num_epochs}: Updating policy...")
            avg_policy_loss, avg_value_loss, avg_entropy = self.update_policy(
                batch_size=batch_size,
                num_updates=updates_per_epoch,
                mini_batch_size=mini_batch_size,
                num_epochs=mini_epochs
            )
            
            # 计算当前学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            self.metrics["learning_rates"].append(current_lr)
            
            # 记录时间
            epoch_time = time.time() - start_time
            
            # 记录TensorBoard指标
            self.writer.add_scalar('Loss/policy', avg_policy_loss, self.epochs)
            self.writer.add_scalar('Loss/value', avg_value_loss, self.epochs)
            self.writer.add_scalar('Metrics/entropy', avg_entropy, self.epochs)
            self.writer.add_scalar('Metrics/learning_rate', current_lr, self.epochs)
            
            # 计算缓冲区大小和利用率
            buffer_size = len(self.replay_buffer)
            buffer_util = buffer_size / self.replay_buffer.buffer.maxlen * 100
            self.writer.add_scalar('Metrics/buffer_size', buffer_size, self.epochs)
            self.writer.add_scalar('Metrics/buffer_utilization', buffer_util, self.epochs)
            
            # 定期评估
            if epoch % eval_interval == 0:
                logger.info(f"Epoch {epoch}/{num_epochs}: Evaluating...")
                win_rate = self.evaluate(num_games=20)
                
                # 记录TensorBoard指标
                self.writer.add_scalar('Metrics/win_rate', win_rate, self.epochs)
                
                # 更新最佳模型
                updated = self.update_best_model(win_rate)
                if updated:
                    self.save_checkpoint(filename="best_model.pt")
            
            # 记录日志
            if epoch % log_interval == 0:
                logger.info(f"Epoch {epoch}/{num_epochs}: "
                            f"Time={epoch_time:.2f}s, "
                            f"Episodes={self.episodes}, "
                            f"PolicyLoss={avg_policy_loss:.6f}, "
                            f"ValueLoss={avg_value_loss:.6f}, "
                            f"Entropy={avg_entropy:.6f}, "
                            f"ExperienceCollected={total_collected}, "
                            f"BufferSize={buffer_size}/{self.replay_buffer.buffer.maxlen} ({buffer_util:.1f}%)")
                
                # 打印当前策略参数
                if self.metrics["alpha_values"] and self.metrics["beta_values"]:
                    alpha = self.metrics["alpha_values"][-1]
                    beta = self.metrics["beta_values"][-1]
                    logger.info(f"Avg Strategy Params: alpha={alpha:.4f} (field), beta={beta:.4f} (urgency)")
            
            # 保存检查点
            if epoch % save_interval == 0:
                self.save_checkpoint()
        
        # 训练结束，保存最终模型
        self.save_checkpoint(filename="final_model.pt")
        
        # 输出训练统计信息
        final_win_rate = self.evaluate(num_games=50)
        logger.info(f"Training completed: final win rate = {final_win_rate:.4f}")
        
        return self.metrics

    def plot_training_metrics(self, save_path=None):
        """绘制训练指标图表
        
        Args:
            save_path: 图表保存路径，如果为None则显示图表
        """
        plt.figure(figsize=(20, 12))
        
        # 1. 胜率
        plt.subplot(2, 3, 1)
        plt.plot(self.metrics["win_rates"])
        plt.title("Win Rate")
        plt.xlabel("Evaluation")
        plt.ylabel("Win Rate")
        plt.grid(True)
        
        # 2. 策略损失和价值损失
        plt.subplot(2, 3, 2)
        plt.plot(self.metrics["policy_losses"], label="Policy Loss")
        plt.plot(self.metrics["value_losses"], label="Value Loss")
        plt.title("Losses")
        plt.xlabel("Update")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        
        # 3. 熵
        plt.subplot(2, 3, 3)
        plt.plot(self.metrics["entropy"])
        plt.title("Entropy")
        plt.xlabel("Update")
        plt.ylabel("Entropy")
        plt.grid(True)
        
        # 4. 奖励
        plt.subplot(2, 3, 4)
        rewards = np.array(self.metrics["episode_rewards"])
        window_size = min(100, len(rewards))
        if window_size > 0:
            smoothed_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
            plt.plot(smoothed_rewards)
            plt.title(f"Episode Rewards (Smoothed over {window_size} episodes)")
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.grid(True)
        
        # 5. 策略参数
        plt.subplot(2, 3, 5)
        plt.plot(self.metrics["alpha_values"], label="Alpha (Field)")
        plt.plot(self.metrics["beta_values"], label="Beta (Urgency)")
        plt.title("Strategy Parameters")
        plt.xlabel("Update")
        plt.ylabel("Parameter Value")
        plt.legend()
        plt.grid(True)
        
        # 6. 学习率
        plt.subplot(2, 3, 6)
        plt.plot(self.metrics["learning_rates"])
        plt.title("Learning Rate")
        plt.xlabel("Update")
        plt.ylabel("Learning Rate")
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Training metrics plot saved to {save_path}")
        else:
            plt.show()

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Train Gomoku AI model")
    
    # 训练参数
    parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs")
    parser.add_argument("--episodes_per_epoch", type=int, default=10, help="Episodes per epoch")
    parser.add_argument("--updates_per_epoch", type=int, default=10, help="Policy updates per epoch")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for experience sampling")
    parser.add_argument("--mini_batch_size", type=int, default=32, help="Mini-batch size for training (0=use full batch)")
    parser.add_argument("--mini_epochs", type=int, default=4, help="Number of mini-epochs per update")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--threads", type=int, default=4, help="Number of threads for experience collection")
    
    # 模型参数
    parser.add_argument("--board_size", type=int, default=9, help="Board size (9 or 15)")
    parser.add_argument("--res_blocks", type=int, default=4, help="Number of residual blocks")
    
    # 扩散参数
    parser.add_argument("--diffusion_rate", type=float, default=0.9, help="Reward diffusion rate")
    parser.add_argument("--diffusion_horizon", type=int, default=10, help="Reward diffusion horizon")
    
    # 评估和保存参数
    parser.add_argument("--eval_interval", type=int, default=5, help="Evaluate model every N epochs")
    parser.add_argument("--save_interval", type=int, default=10, help="Save checkpoint every N epochs")
    parser.add_argument("--log_interval", type=int, default=1, help="Log metrics every N epochs")
    
    # 其他参数
    parser.add_argument("--device", type=str, default="auto", 
                      help="Device to use (cuda/cpu/auto)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Load checkpoint from file")
    parser.add_argument("--logdir", type=str, default="runs", help="TensorBoard log directory")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint directory")
    
    return parser.parse_args()

def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 创建模型
    model = GomokuModel(
        board_size=args.board_size,
        num_res_blocks=args.res_blocks
    )
    
    # 根据环境选择设备
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # 如果使用CUDA，启用cudnn优化
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        logger.info(f"CUDA available: {torch.cuda.is_available()}, "
                   f"Device count: {torch.cuda.device_count()}, "
                   f"Current device: {torch.cuda.current_device()}, "
                   f"Device name: {torch.cuda.get_device_name(0)}")
    
    # 创建训练器
    trainer = PPOTrainer(
        model=model,
        board_size=args.board_size,
        lr=args.lr,
        gamma=args.gamma,
        diffusion_rate=args.diffusion_rate,
        diffusion_horizon=args.diffusion_horizon,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        tb_log_dir=args.logdir
    )
    
    # 如果指定了检查点，加载之
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
    
    # 训练
    trainer.train(
        num_epochs=args.epochs,
        episodes_per_epoch=args.episodes_per_epoch,
        batch_size=args.batch_size,
        updates_per_epoch=args.updates_per_epoch,
        mini_batch_size=args.mini_batch_size,
        mini_epochs=args.mini_epochs,
        num_threads=args.threads,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval
    )
    
    # 绘制训练指标
    trainer.plot_training_metrics(save_path="training_metrics.png")
    
    # 保存最终指标
    with open("training_metrics.json", "w") as f:
        json.dump(trainer.metrics, f)
    
    logger.info("Training completed. Final metrics saved.")

if __name__ == "__main__":
    main()