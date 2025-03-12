import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional

class ResidualBlock(nn.Module):
    """残差块，用于深度残差网络"""
    def __init__(self, channels: int):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x

class DirectionalFieldModule(nn.Module):
    """方向信息场模块，严格按照信息场理论实现滑向机制
    
    F_{i,j} = sum_{d ∈ D} φ(S_d) · ω_d(i,j)
    
    直观理解：计算每个空位在各个方向上的"吸引力"
    """
    def __init__(self, channels: int):
        super(DirectionalFieldModule, self).__init__()
        self.channels_per_dir = channels // 4
        
        # 四个方向的卷积：水平、垂直、主对角线、副对角线
        self.horizontal_conv = nn.Conv2d(channels, self.channels_per_dir, kernel_size=(1, 5), padding=(0, 2))
        self.vertical_conv = nn.Conv2d(channels, self.channels_per_dir, kernel_size=(5, 1), padding=(2, 0))
        self.diag1_conv = nn.Conv2d(channels, self.channels_per_dir, kernel_size=5, padding=2)
        self.diag2_conv = nn.Conv2d(channels, self.channels_per_dir, kernel_size=5, padding=2)
        
        # 每个方向的权重（可学习）- ω_d(i,j)
        self.direction_weights = nn.Parameter(torch.ones(4) / 4)
        
        # 棋型强度整合
        self.field_integrator = nn.Conv2d(self.channels_per_dir, 1, kernel_size=1)
        
    def forward(self, x):
        # 计算四个方向的场 - φ(S_d)
        h_field = self.horizontal_conv(x)
        v_field = self.vertical_conv(x)
        d1_field = self.diag1_conv(x)
        d2_field = self.diag2_conv(x)
        
        # 方向权重归一化 - 确保权重和为1
        dir_weights = F.softmax(self.direction_weights, dim=0)
        
        # 综合各方向场（带权重）- sum_{d ∈ D} φ(S_d) · ω_d(i,j)
        h_field = self.field_integrator(h_field) * dir_weights[0]
        v_field = self.field_integrator(v_field) * dir_weights[1]
        d1_field = self.field_integrator(d1_field) * dir_weights[2]
        d2_field = self.field_integrator(d2_field) * dir_weights[3]
        
        # 合并所有方向
        combined_field = h_field + v_field + d1_field + d2_field
        
        return combined_field

class TemporalGradientModule(nn.Module):
    """时间梯度模块，基于理论公式实现急迫度机制
    
    T_{i,j} = sum_{t=0}^{H} γ^t [V(S_{t+1}^{i,j}) - V(S_t)]
    
    直观理解：评估每个落子点能带来的价值增量，优先选择能更快带来高价值的位置
    """
    def __init__(self, channels: int):
        super(TemporalGradientModule, self).__init__()
        # 当前状态价值估计器
        self.value_estimator = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1)
        )
        
        # 模拟落子后状态的价值估计器
        self.next_value_estimator = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1)
        )
        
        # 可学习的时间折扣因子
        self.gamma = nn.Parameter(torch.tensor(0.9))
        
    def forward(self, x):
        # 估计当前状态价值 V(S_t)
        current_value = self.value_estimator(x)
        
        # 估计每个位置落子后的状态价值 V(S_{t+1}^{i,j})
        # 这里使用卷积来近似模拟落子效果
        next_value = self.next_value_estimator(x)
        
        # 计算时间梯度（价值增量）
        temporal_gradient = next_value - current_value
        
        # 应用可学习的时间折扣因子
        # 正值表示该位置落子会带来价值提升，值越大表示提升越多
        return torch.sigmoid(self.gamma) * temporal_gradient

class StrategicParameterModule(nn.Module):
    """策略参数模块，基于历史序列输出整合参数
    
    直观理解：根据局势动态调整信息场和时间梯度的权重(α和β)
    """
    def __init__(self, channels: int, seq_length: int = 8):
        super(StrategicParameterModule, self).__init__()
        self.seq_length = seq_length
        
        # 位置编码层
        self.position_embedding = nn.Parameter(
            torch.zeros(1, seq_length, channels)
        )
        
        # 时序特征提取
        self.sequence_encoder = nn.GRU(
            input_size=channels,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        
        # 参数输出头 - 生成α和β参数
        self.param_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # 输出两个策略参数
        )
        
    def forward(self, sequence_features):
        """
        处理动作序列，输出策略参数
        
        Args:
            sequence_features: 形状为 [batch_size, seq_length, channels]
        
        Returns:
            alpha, beta: 策略参数，用于组合信息场和时间梯度
        """
        batch_size = sequence_features.size(0)
        
        # 应用位置编码
        sequence_features = sequence_features + self.position_embedding[:, :sequence_features.size(1), :]
        
        # 通过序列编码器
        _, hidden = self.sequence_encoder(sequence_features)
        hidden = hidden.squeeze(0)
        
        # 输出策略参数
        params = self.param_head(hidden)
        
        # 使用Sigmoid激活确保参数在[0,1]范围内
        alpha = torch.sigmoid(params[:, 0])
        beta = torch.sigmoid(params[:, 1])
        
        return alpha, beta

class GomokuModel(nn.Module):
    """
    基于理论公式的五子棋模型，融合视觉、信息场、时间梯度和策略参数
    """
    def __init__(self, 
                 board_size: int = 15, 
                 in_channels: int = 3,
                 base_channels: int = 64, 
                 num_res_blocks: int = 10,
                 seq_length: int = 8):
        super(GomokuModel, self).__init__()
        
        self.board_size = board_size
        self.in_channels = in_channels
        self.seq_length = seq_length
        
        # 视觉基础层：初始卷积和批归一化
        self.conv_input = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(base_channels)
        
        # 残差层
        self.res_blocks = nn.ModuleList([
            ResidualBlock(base_channels) for _ in range(num_res_blocks)
        ])
        
        # 方向信息场模块（滑向机制）- 按理论公式实现
        self.information_field = DirectionalFieldModule(base_channels)
        
        # 时间梯度模块（急迫度）- 按理论公式实现
        self.temporal_gradient = TemporalGradientModule(base_channels)
        
        # 策略参数模块 - 输出alpha和beta
        self.strategic_params = StrategicParameterModule(base_channels, seq_length)
        
        # 特征提取器
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(base_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU()
        )
        
        # 策略网络（输出动作概率）
        self.policy_head = nn.Sequential(
            nn.Conv2d(base_channels, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * board_size * board_size, board_size * board_size)
        )
        
        # 价值网络（输出局面评估）
        self.value_head = nn.Sequential(
            nn.Conv2d(base_channels, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * board_size * board_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()  # 输出范围为[-1, 1]
        )
        
    def extract_features(self, state):
        """处理状态并提取特征"""
        x = F.relu(self.bn_input(self.conv_input(state)))
        
        # 应用残差块
        for block in self.res_blocks:
            x = block(x)
            
        return x
    
    def forward(self, 
               state: torch.Tensor, 
               state_sequence: Optional[torch.Tensor] = None,
               valid_moves_mask: Optional[torch.Tensor] = None):
        """
        前向传播，严格按照理论公式整合各组件
        
        Args:
            state: 当前棋盘状态 [batch_size, in_channels, board_size, board_size]
            state_sequence: 历史状态序列特征 [batch_size, seq_length, feature_dim]
            valid_moves_mask: 有效动作掩码 [batch_size, board_size*board_size]
            
        Returns:
            policy_logits: 动作概率对数 [batch_size, board_size*board_size]
            value: 状态价值估计 [batch_size, 1]
        """
        batch_size = state.size(0)
        
        # 1. 处理当前状态
        x = self.extract_features(state)
        
        # 2. 应用特征提取器
        enhanced_features = self.feature_extractor(x)
        
        # 3. 计算各方向信息场 F_{i,j}
        information_field = self.information_field(enhanced_features)
        
        # 4. 计算时间梯度（急迫度）T_{i,j}
        temporal_gradient = self.temporal_gradient(enhanced_features)
        
        # 5. 获取策略参数α和β
        if state_sequence is None:
            # 默认参数 - 等权重
            alpha = torch.ones(batch_size, device=state.device) * 0.5
            beta = torch.ones(batch_size, device=state.device) * 0.5
        else:
            # 使用策略思考模块计算参数
            alpha, beta = self.strategic_params(state_sequence)
        
        # 调整参数形状以便于广播
        alpha = alpha.view(batch_size, 1, 1, 1)
        beta = beta.view(batch_size, 1, 1, 1)
        
        # 6. 整合注意力场 A_{i,j} = exp(α·F_{i,j} + β·T_{i,j}) / sum_{k,l}exp(...)
        attention_field = alpha * information_field + beta * temporal_gradient
        
        # 7. 应用softmax获得注意力权重
        attention = F.softmax(attention_field.view(batch_size, -1), dim=1).view_as(attention_field)
        
        # 8. 应用注意力到特征
        attended_features = enhanced_features * (1.0 + attention)  # 残差连接增强原始信息
        
        # 9. 计算策略（动作概率）
        policy_logits = self.policy_head(attended_features)
        
        # 10. 应用有效动作掩码
        if valid_moves_mask is not None:
            policy_logits = policy_logits.masked_fill(valid_moves_mask == 0, float('-inf'))
        
        # 11. 计算价值估计
        value = self.value_head(attended_features)
        
        return policy_logits, value
    
    def get_action(self, 
                  state: torch.Tensor, 
                  state_sequence: Optional[torch.Tensor] = None,
                  valid_moves_mask: Optional[torch.Tensor] = None,
                  deterministic: bool = False):
        """
        获取模型预测的动作
        
        Args:
            state: 当前状态
            state_sequence: 状态序列特征（可选）
            valid_moves_mask: 有效动作掩码（可选）
            deterministic: 是否确定性选择（为True时选择概率最高的动作）
            
        Returns:
            action: 选择的动作索引
            prob: 动作的概率
            value: 状态价值估计
        """
        with torch.no_grad():
            policy_logits, value = self(state, state_sequence, valid_moves_mask)
            
            # 转换为概率
            policy = F.softmax(policy_logits, dim=1)
            
            if deterministic:
                # 选择概率最高的动作
                action = torch.argmax(policy, dim=1)
            else:
                # 根据概率采样动作
                dist = torch.distributions.Categorical(policy)
                action = dist.sample()
            
            prob = torch.gather(policy, 1, action.unsqueeze(1)).squeeze(1)
            
            return action, prob, value


class GomokuAgent:
    """
    基于GomokuModel的智能体，实现与环境交互
    """
    def __init__(self, 
                model: GomokuModel, 
                device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                sequence_length: int = 8,
                feature_dim: int = 64,
                deterministic: bool = False,
                name: str = "NeuralAgent"):
        """
        初始化智能体
        
        Args:
            model: 训练好的GomokuModel
            device: 计算设备
            sequence_length: 历史序列长度
            feature_dim: 状态特征维度
            deterministic: 是否确定性选择动作
            name: 智能体名称
        """
        self.model = model.to(device)
        self.model.eval()  # 设置为评估模式
        self.device = device
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.deterministic = deterministic
        self.name = name
        
        # 保存历史状态和特征
        self.state_history = []
        self.feature_history = []
    
    def select_action(self, env):
        """
        选择动作接口，兼容GomokuEnv
        
        Args:
            env: GomokuEnv实例
            
        Returns:
            action: 选择的动作坐标 (x, y)
        """
        # 获取当前状态
        state = env.get_state()
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # 保存当前状态到历史
        self.state_history.append(state)
        if len(self.state_history) > self.sequence_length:
            self.state_history.pop(0)
        
        # 提取当前状态特征
        with torch.no_grad():
            current_features = self.model.extract_features(state_tensor)
        
        # 将特征添加到历史中
        flattened_features = current_features.mean((2, 3)).cpu().numpy()
        self.feature_history.append(flattened_features)
        if len(self.feature_history) > self.sequence_length:
            self.feature_history.pop(0)
        
        # 准备状态特征序列
        if len(self.feature_history) > 1:
            sequence_features = torch.FloatTensor(
                np.concatenate(self.feature_history, axis=0)
            ).unsqueeze(0).to(self.device)
            
            # 重塑为[batch, seq_len, feature_dim]
            sequence_features = sequence_features.view(1, len(self.feature_history), -1)
        else:
            sequence_features = None
        
        # 获取有效动作掩码
        valid_moves_mask = torch.FloatTensor(
            env.get_valid_moves_mask()
        ).unsqueeze(0).to(self.device)
        
        # 使用模型获取动作
        action_idx, _, _ = self.model.get_action(
            state_tensor, 
            sequence_features,
            valid_moves_mask,
            self.deterministic
        )
        
        # 将一维动作索引转换为坐标
        action_idx = action_idx.item()
        x, y = divmod(action_idx, env.board_size)
        
        return (x, y)
    
    def reset(self):
        """重置智能体状态"""
        self.state_history = []
        self.feature_history = []


# 测试智能体（如果需要）
if __name__ == "__main__":
    from gomokuenv import GomokuEnv, ai_vs_ai, RandomAgent
    
    # 创建小棋盘进行测试
    board_size = 9
    model = GomokuModel(board_size=board_size, num_res_blocks=3)
    
    # 创建智能体
    agent = GomokuAgent(model, name="神经网络智能体")
    
    # 与随机智能体对战
    random_agent = RandomAgent(name="随机智能体")
    result = ai_vs_ai(agent, random_agent, board_size=board_size)
    
    print(f"游戏结果: {result}")