import numpy as np
from typing import Tuple, List, Dict, Optional, Union, Callable
import random

class GomokuEnv:
    """
    五子棋环境，提供游戏规则和状态管理
    """
    def __init__(self, board_size: int = 15):
        """
        初始化五子棋环境
        
        Args:
            board_size: 棋盘大小，默认为15x15
        """
        self.board_size = board_size
        self.reset()
    
    def reset(self) -> np.ndarray:
        """
        重置游戏状态
        
        Returns:
            当前状态表示 (3, board_size, board_size)
        """
        # 0表示空位，1表示黑子，2表示白子
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.current_player = 1  # 黑子先行
        self.last_move = None
        self.history = []  # 动作历史 [(x, y, player), ...]
        self.done = False
        self.winner = 0
        
        return self.get_state()
    
    def get_state(self) -> np.ndarray:
        """
        获取当前状态表示
        
        Returns:
            状态表示 (3, board_size, board_size)
            - channel 0: 当前玩家的棋子位置
            - channel 1: 对手的棋子位置
            - channel 2: 可行动位置
        """
        state = np.zeros((3, self.board_size, self.board_size), dtype=np.float32)
        
        # 当前玩家的棋子
        state[0] = (self.board == self.current_player).astype(np.float32)
        # 对手的棋子
        state[1] = (self.board == 3 - self.current_player).astype(np.float32)
        # 可行动位置
        state[2] = (self.board == 0).astype(np.float32)
        
        return state
    
    def step(self, action: Union[int, Tuple[int, int]]) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        执行一步动作
        
        Args:
            action: 动作，可以是一维索引(0~board_size^2-1)或二维坐标(x,y)
        
        Returns:
            (新状态, 奖励, 游戏是否结束, 信息字典)
        """
        # 将一维动作转换为坐标
        if isinstance(action, int):
            x, y = divmod(action, self.board_size)
        else:
            x, y = action
        
        # 检查动作有效性
        if not self.is_valid_move(x, y):
            return self.get_state(), -1.0, self.done, {"error": "Invalid move"}
        
        # 执行动作
        self.board[x, y] = self.current_player
        self.last_move = (x, y)
        self.history.append((x, y, self.current_player))
        
        # 检查游戏是否结束
        win = self.check_win(x, y)
        if win:
            self.done = True
            self.winner = self.current_player
            reward = 1.0
        elif np.sum(self.board == 0) == 0:  # 棋盘已满，平局
            self.done = True
            reward = 0.0
        else:
            reward = 0.0
        
        # 切换玩家
        self.current_player = 3 - self.current_player  # 1->2, 2->1
        
        return self.get_state(), reward, self.done, {
            "last_move": self.last_move,
            "current_player": self.current_player,
            "winner": self.winner
        }
    
    def is_valid_move(self, x: int, y: int) -> bool:
        """
        检查动作是否有效
        
        Args:
            x: x坐标
            y: y坐标
        
        Returns:
            动作是否有效
        """
        # 检查坐标是否在棋盘内
        if not (0 <= x < self.board_size and 0 <= y < self.board_size):
            return False
        # 检查位置是否已有棋子
        if self.board[x, y] != 0:
            return False
        # 检查游戏是否已结束
        if self.done:
            return False
        
        return True
    
    def get_valid_moves(self) -> List[Tuple[int, int]]:
        """
        获取所有有效动作的坐标
        
        Returns:
            有效动作坐标列表 [(x, y), ...]
        """
        if self.done:
            return []
        
        return [(x, y) for x in range(self.board_size) for y in range(self.board_size) 
                if self.board[x, y] == 0]
    
    def get_valid_moves_mask(self) -> np.ndarray:
        """
        获取有效动作掩码
        
        Returns:
            一维掩码数组，1表示有效，0表示无效
        """
        mask = np.zeros(self.board_size * self.board_size, dtype=np.int8)
        
        if not self.done:
            for x, y in self.get_valid_moves():
                mask[x * self.board_size + y] = 1
        
        return mask
    
    def check_win(self, x: int, y: int) -> bool:
        """
        检查最后一步是否导致胜利
        
        Args:
            x: 最后一步的x坐标
            y: 最后一步的y坐标
        
        Returns:
            是否获胜
        """
        player = self.board[x, y]
        if player == 0:
            return False
        
        # 检查四个方向：水平、垂直、主对角线、副对角线
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        
        for dx, dy in directions:
            count = 1  # 当前位置算1个
            
            # 沿着方向检查
            for i in range(1, 5):
                nx, ny = x + dx * i, y + dy * i
                if 0 <= nx < self.board_size and 0 <= ny < self.board_size and self.board[nx, ny] == player:
                    count += 1
                else:
                    break
            
            # 沿着反方向检查
            for i in range(1, 5):
                nx, ny = x - dx * i, y - dy * i
                if 0 <= nx < self.board_size and 0 <= ny < self.board_size and self.board[nx, ny] == player:
                    count += 1
                else:
                    break
            
            # 如果达到五子连线
            if count >= 5:
                return True
        
        return False
    
    def render(self) -> str:
        """
        渲染棋盘为字符串表示
        
        Returns:
            棋盘的字符串表示
        """
        # 棋盘顶部坐标
        board_str = "   " + "".join([f"{i:2d}" for i in range(self.board_size)]) + "\n"
        board_str += "  +" + "-" * (self.board_size * 2) + "-+\n"
        
        # 棋盘内容
        for i in range(self.board_size):
            board_str += f"{i:2d}|"
            for j in range(self.board_size):
                if self.board[i, j] == 0:
                    board_str += " ·"
                elif self.board[i, j] == 1:
                    board_str += " ●"  # 黑子
                else:
                    board_str += " ○"  # 白子
            board_str += " |\n"
        
        # 棋盘底部
        board_str += "  +" + "-" * (self.board_size * 2) + "-+\n"
        
        # 显示当前玩家和最后一步
        player_str = "黑方(●)" if self.current_player == 1 else "白方(○)"
        board_str += f"当前玩家: {player_str}"
        
        if self.last_move:
            board_str += f", 最后落子: {self.last_move}"
        
        if self.done:
            if self.winner != 0:
                winner_str = "黑方(○)" if self.winner == 1 else "白方(●)"
                board_str += f"\n游戏结束! 胜者: {winner_str}"
            else:
                board_str += "\n游戏结束! 平局!"
        
        return board_str


class Agent:
    """
    智能体基类，所有玩家（AI或人类）都应实现此接口
    """
    def __init__(self, name: str = "Agent"):
        """
        初始化智能体
        
        Args:
            name: 智能体名称
        """
        self.name = name
    
    def select_action(self, env: GomokuEnv) -> Tuple[int, int]:
        """根据环境选择动作"""
        raise NotImplementedError("Subclasses must implement select_action")
    
    def reset(self) -> None:
        """
        重置智能体状态（如果需要）
        """
        pass

class RandomAgent(Agent):
    """
    随机智能体，随机选择有效动作
    """
    def __init__(self, name: str = "RandomAgent"):
        super().__init__(name)
    
    def select_action(self, env: GomokuEnv) -> Tuple[int, int]:
        """随机选择有效动作"""
        valid_moves = env.get_valid_moves()
        if not valid_moves:
            return (-1, -1)
        
        return random.choice(valid_moves)

class HumanAgent(Agent):
    """
    人类玩家智能体
    """
    def __init__(self, name: str = "Human"):
        super().__init__(name)
    
    def select_action(self, env: GomokuEnv) -> Tuple[int, int]:
        """
        让人类选择动作
        
        Args:
            env: 游戏环境
        
        Returns:
            选择的坐标 (x, y)
        """
        board_size = env.board_size
        valid_moves = env.get_valid_moves()
        
        while True:
            try:
                prompt = f"\n请输入动作坐标 (行 列), 范围 0-{board_size-1}: "
                action_str = input(prompt)
                
                # 支持退出
                if action_str.lower() in ('q', 'quit', 'exit'):
                    print("玩家选择退出游戏")
                    return (-1, -1)
                
                # 支持显示有效动作
                if action_str.lower() in ('h', 'help', '?'):
                    print(f"有效动作: {valid_moves}")
                    continue
                
                # 解析坐标
                parts = action_str.split()
                if len(parts) != 2:
                    print("请输入两个数字，以空格分隔")
                    continue
                    
                x, y = map(int, parts)
                
                # 验证有效性
                if env.is_valid_move(x, y):
                    return (x, y)
                else:
                    print(f"位置 ({x}, {y}) 无效，请重新选择。")
                    print(f"可能是坐标超出范围或该位置已有棋子")
            
            except ValueError:
                print("输入格式错误，请输入两个用空格分隔的整数。")
            except Exception as e:
                print(f"发生错误: {str(e)}")

def play_game(
    env: GomokuEnv,
    black_agent: Agent,
    white_agent: Optional[Agent] = None,
    render: bool = True,
    delay: float = 0,
    callback: Optional[Callable] = None
) -> Dict:
    """
    进行一局对弈
    
    Args:
        env: 游戏环境
        black_agent: 黑方智能体
        white_agent: 白方智能体，如果为None则使用black_agent
        render: 是否显示棋盘
        delay: 每步延迟时间（秒）
        callback: 每步回调函数 callback(env, state, action, agent)
    
    Returns:
        游戏结果信息
    """
    import time
    
    # 如果没有提供白方，使用黑方
    if white_agent is None:
        white_agent = black_agent
    
    # 获取智能体名称
    black_name = getattr(black_agent, 'name', '黑方')
    white_name = getattr(white_agent, 'name', '白方')
    
    # 重置环境和智能体
    state = env.reset()
    black_agent.reset()
    white_agent.reset()
    
    # 显示初始棋盘
    if render:
        print(f"\n{black_name} (●) vs {white_name} (○)")
        print(env.render())
    
    # 游戏循环
    while not env.done:
        # 获取当前智能体
        current_agent = black_agent if env.current_player == 1 else white_agent
        agent_name = black_name if env.current_player == 1 else white_name
        
        # 显示当前玩家
        if render:
            print(f"\n{agent_name} 思考中...")
        
        # 智能体选择动作
        action = current_agent.select_action(env)
        
        # 检查特殊动作（如退出）
        if action == (-1, -1):
            print(f"{agent_name} 选择退出游戏。")
            break
        
        # 执行动作
        state, reward, done, info = env.step(action)
        
        # 显示动作
        if render:
            print(f"{agent_name} 落子于 {action}")
            print(env.render())
        
        # 延迟
        if delay > 0:
            time.sleep(delay)
        
        # 回调
        if callback:
            callback(env, state, action, current_agent)
    
    # 显示最终结果
    if render and env.done:
        if env.winner != 0:
            winner_name = black_name if env.winner == 1 else white_name
            print(f"\n游戏结束! {winner_name} 获胜!")
        else:
            print("\n游戏结束! 平局!")
    
    # 返回游戏结果
    return {
        "winner": env.winner,
        "black_agent": black_name,
        "white_agent": white_name,
        "moves": len(env.history),
        "history": env.history.copy()
    }

def human_vs_ai(
    ai_agent: Agent, 
    board_size: int = 15, 
    human_first: bool = True,
    render: bool = True
) -> Dict:
    """
    人机对战
    
    Args:
        ai_agent: AI智能体
        board_size: 棋盘大小
        human_first: 人类是否先手
        render: 是否渲染棋盘
    
    Returns:
        对局结果
    """
    env = GomokuEnv(board_size=board_size)
    human = HumanAgent(name="玩家")
    
    if human_first:
        result = play_game(env, human, ai_agent, render=render)
    else:
        result = play_game(env, ai_agent, human, render=render)
    
    return result

def ai_vs_ai(
    black_agent: Agent,
    white_agent: Agent,
    board_size: int = 15,
    render: bool = True,
    delay: float = 0.5
) -> Dict:
    """
    AI之间的对战
    
    Args:
        black_agent: 黑方智能体
        white_agent: 白方智能体
        board_size: 棋盘大小
        render: 是否渲染棋盘
        delay: 每步延迟时间（秒）
    
    Returns:
        对局结果
    """
    env = GomokuEnv(board_size=board_size)
    result = play_game(env, black_agent, white_agent, render=render, delay=delay)
    return result

# 示例用法
if __name__ == "__main__":
    # 创建环境和智能体
    env = GomokuEnv(board_size=9)  # 使用9x9棋盘便于测试
    random_agent = RandomAgent(name="随机智能体")
    human_agent = HumanAgent(name="人类玩家")
    
    # 人机对战
    print("\n===== 人机对战 =====")
    human_vs_ai(random_agent, board_size=9)
    
    # AI自我对弈
    print("\n===== AI自我对弈 =====")
    ai_vs_ai(random_agent, RandomAgent(name="随机智能体2"), board_size=9)