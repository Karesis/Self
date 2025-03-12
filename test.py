import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

from model import GomokuModel, GomokuAgent
from gomokuenv import GomokuEnv, human_vs_ai, ai_vs_ai, RandomAgent, HumanAgent

def load_model(model_path=None, board_size=9, num_res_blocks=4):
    """加载训练好的模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 如果未指定模型路径，尝试找到最佳模型或最新模型
    if model_path is None:
        if os.path.exists("checkpoints/best_model.pt"):
            model_path = "checkpoints/best_model.pt"
        elif os.path.exists("checkpoints/final_model.pt"):
            model_path = "checkpoints/final_model.pt"
        else:
            # 尝试找到最新的检查点
            checkpoints = [f for f in os.listdir("checkpoints") if f.endswith(".pt")]
            if checkpoints:
                model_path = os.path.join("checkpoints", sorted(checkpoints)[-1])
            else:
                raise FileNotFoundError("找不到模型文件，请确保checkpoints目录中有.pt文件")
    
    # 创建模型
    model = GomokuModel(
        board_size=board_size,
        num_res_blocks=num_res_blocks
    )
    
    # 加载模型权重
    print(f"正在加载模型: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()  # 设置为评估模式
    
    # 创建智能体
    ai_agent = GomokuAgent(
        model=model,
        device=device,
        deterministic=True,  # 测试时用确定性策略
        name="NeuralAgent"
    )
    
    print(f"模型已加载，参数: board_size={board_size}, num_res_blocks={num_res_blocks}")
    return model, ai_agent

def vs_random_test(ai_agent, board_size=9, num_games=10, render=True, delay=0.5):
    """测试AI与随机智能体的对战表现"""
    print(f"\n=== AI vs 随机智能体 ({num_games}局) ===")
    
    # 创建随机智能体
    random_agent = RandomAgent(name="随机智能体")
    
    # 进行AI对战
    black_wins = 0
    white_wins = 0
    draws = 0
    
    for i in range(num_games):
        # 交替先后手
        if i % 2 == 0:
            print(f"\n第 {i+1}/{num_games} 局: AI(黑) vs 随机(白)")
            result = ai_vs_ai(ai_agent, random_agent, board_size=board_size, render=render, delay=delay)
            if result["winner"] == 1:
                black_wins += 1
                print("结果: AI(黑)胜利")
            elif result["winner"] == 2:
                white_wins += 1
                print("结果: 随机(白)胜利")
            else:
                draws += 1
                print("结果: 平局")
        else:
            print(f"\n第 {i+1}/{num_games} 局: 随机(黑) vs AI(白)")
            result = ai_vs_ai(random_agent, ai_agent, board_size=board_size, render=render, delay=delay)
            if result["winner"] == 1:
                white_wins += 1
                print("结果: 随机(黑)胜利")
            elif result["winner"] == 2:
                black_wins += 1
                print("结果: AI(白)胜利")
            else:
                draws += 1
                print("结果: 平局")
    
    ai_wins = black_wins + white_wins
    print(f"\n对战结果：AI胜利 {ai_wins}/{num_games} (胜率 {ai_wins/num_games:.2%})")
    print(f"AI执黑胜利: {black_wins}/黑方总局数, AI执白胜利: {white_wins}/白方总局数, 平局: {draws}")
    
    return ai_wins / num_games

def self_play_test(ai_agent, board_size=9, num_games=5, render=True, delay=1.0):
    """测试AI自我对弈"""
    print(f"\n=== AI自我对弈 ({num_games}局) ===")
    
    # 创建第二个相同智能体
    ai_agent2 = GomokuAgent(
        model=ai_agent.model,
        device=ai_agent.device,
        deterministic=True,
        name="NeuralAgent2"
    )
    
    black_wins = 0
    white_wins = 0
    draws = 0
    
    for i in range(num_games):
        print(f"\n第 {i+1}/{num_games} 局自我对弈")
        result = ai_vs_ai(ai_agent, ai_agent2, board_size=board_size, render=render, delay=delay)
        
        if result["winner"] == 1:
            black_wins += 1
            print("结果: 黑方胜")
        elif result["winner"] == 2:
            white_wins += 1
            print("结果: 白方胜")
        else:
            draws += 1
            print("结果: 平局")
    
    print(f"\n自我对弈结果: 黑方胜: {black_wins}, 白方胜: {white_wins}, 平局: {draws}")
    
    return black_wins, white_wins, draws

def human_play_test(ai_agent, board_size=9):
    """人机对战测试"""
    print("\n=== 人机对战 ===")
    print("游戏规则：输入格式为 '行 列'，例如 '3 4' 表示第3行第4列")
    print("输入 'q' 退出游戏, 输入 'h' 显示有效动作\n")
    
    while True:
        # 选择先后手
        first = input("您想要执黑先手吗？(y/n): ").lower().startswith('y')
        
        if first:
            print("\n您执黑先手，AI执白后手")
        else:
            print("\n您执白后手，AI执黑先手")
        
        result = human_vs_ai(ai_agent, board_size=board_size, human_first=first)
        
        if result["winner"] == 0:
            print("游戏结束：平局！")
        elif (result["winner"] == 1 and first) or (result["winner"] == 2 and not first):
            print("游戏结束：您赢了！")
        else:
            print("游戏结束：AI赢了！")
        
        # 是否继续
        if not input("\n还想再玩一局吗？(y/n): ").lower().startswith('y'):
            break

def test_specific_position(ai_agent, position_setup, next_player=1, board_size=9):
    """测试AI在特定局面下的决策"""
    env = GomokuEnv(board_size=board_size)
    env.reset()
    
    # 设置特定局面
    for (x, y, player) in position_setup:
        env.board[x, y] = player
    env.current_player = next_player
    
    # 显示局面
    print("\n当前局面：")
    print(env.render())
    
    # 让AI决策
    ai_agent.reset()
    action = ai_agent.select_action(env)
    
    print(f"AI选择落子在: {action}")
    
    # 执行动作
    _, _, done, info = env.step(action)
    print("\n落子后局面：")
    print(env.render())
    
    if done:
        if info["winner"] == 0:
            print("结果: 平局")
        else:
            print(f"结果: {'黑方' if info['winner'] == 1 else '白方'}胜利")
    
    return action

def specific_test(ai_agent, board_size=9):
    """测试特定局面"""
    print("\n=== 测试特定局面 ===")
    
    while True:
        print("\n选择测试局面类型:")
        print("1. 进攻局面（AI是黑方，有连三）")
        print("2. 防守局面（AI是白方，黑方有连三）")
        print("3. 双方对杀局面")
        print("4. 自定义局面")
        print("0. 返回上一级菜单")
        
        choice = input("请选择(0-4): ")
        
        if choice == '1':
            # 测试进攻局面
            attack_position = [
                (3, 3, 1), (3, 4, 1), (3, 5, 1),  # 黑方三连
                (4, 3, 2), (4, 4, 2)              # 白方两子
            ]
            test_specific_position(ai_agent, attack_position, next_player=1, board_size=board_size)
        
        elif choice == '2':
            # 测试防守局面
            defense_position = [
                (3, 3, 1), (3, 4, 1), (3, 5, 1),  # 黑方三连
                (4, 3, 2), (4, 4, 2)              # 白方两子
            ]
            test_specific_position(ai_agent, defense_position, next_player=2, board_size=board_size)
        
        elif choice == '3':
            # 双方对杀局面
            complex_position = [
                (3, 3, 1), (3, 4, 1), (3, 5, 1),  # 黑方三连
                (4, 2, 2), (4, 3, 2), (4, 4, 2)   # 白方三连
            ]
            
            # 黑方先手
            player = 1
            while True:
                test_specific_position(ai_agent, complex_position, next_player=player, board_size=board_size)
                # 切换玩家
                player = 3 - player  # 1->2, 2->1
                
                if input("\n继续对杀？(y/n): ").lower() != 'y':
                    break
                    
        elif choice == '4':
            # 自定义局面
            custom_position = []
            print("\n请输入自定义局面，格式为 '行 列 玩家'，玩家为1(黑)或2(白)，输入'done'完成")
            
            while True:
                pos_input = input("输入位置 (行 列 玩家) 或 'done': ")
                if pos_input.lower() == 'done':
                    break
                    
                try:
                    x, y, player = map(int, pos_input.split())
                    if 0 <= x < board_size and 0 <= y < board_size and player in [1, 2]:
                        custom_position.append((x, y, player))
                    else:
                        print(f"无效输入，坐标范围: 0-{board_size-1}, 玩家: 1或2")
                except:
                    print("输入格式错误，请按 '行 列 玩家' 格式输入")
            
            if custom_position:
                next_player = int(input("AI使用哪方? (1=黑, 2=白): "))
                if next_player in [1, 2]:
                    test_specific_position(ai_agent, custom_position, next_player=next_player, board_size=board_size)
                else:
                    print("无效的玩家选择")
            else:
                print("未设置任何棋子，取消测试")
        
        elif choice == '0':
            break
        
        else:
            print("无效选项")

def visualize_policy(env, ai_agent):
    """可视化当前局面下AI的动作概率分布"""
    device = ai_agent.device
    state = env.get_state()
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    
    # 获取动作掩码
    valid_moves_mask = torch.FloatTensor(
        env.get_valid_moves_mask()
    ).unsqueeze(0).to(device)
    
    # 获取历史序列特征
    seq_features = None
    if len(ai_agent.feature_history) > 1:
        seq_features = torch.FloatTensor(
            np.concatenate(ai_agent.feature_history, axis=0)
        ).unsqueeze(0).to(device)
        seq_features = seq_features.view(1, len(ai_agent.feature_history), -1)
    
    # 获取策略和评估值
    with torch.no_grad():
        policy_logits, value = ai_agent.model(state_tensor, seq_features, valid_moves_mask)
        policy = F.softmax(policy_logits, dim=1).cpu().numpy()[0]
    
    # 重塑为棋盘形状
    policy_2d = policy.reshape(env.board_size, env.board_size)
    
    # 创建图形
    plt.figure(figsize=(10, 8))
    
    # 显示棋盘
    plt.subplot(1, 2, 1)
    board_img = np.zeros((env.board_size, env.board_size, 3))
    for i in range(env.board_size):
        for j in range(env.board_size):
            if env.board[i, j] == 1:  # 黑子
                board_img[i, j] = [0, 0, 0]
            elif env.board[i, j] == 2:  # 白子
                board_img[i, j] = [1, 1, 1]
            else:
                board_img[i, j] = [0.8, 0.8, 0.6]  # 棋盘底色
    
    plt.imshow(board_img)
    for i in range(env.board_size):
        for j in range(env.board_size):
            if env.board[i, j] != 0:
                plt.text(j, i, '●', ha='center', va='center', 
                         color='white' if env.board[i, j] == 2 else 'black', 
                         fontsize=15)
    
    plt.title(f"当前棋局 (估值: {value.item():.2f})")
    plt.grid(color='black')
    
    # 显示策略热图
    plt.subplot(1, 2, 2)
    im = plt.imshow(policy_2d, cmap='hot', interpolation='nearest')
    plt.colorbar(im, label='落子概率')
    
    # 添加概率文本
    for i in range(env.board_size):
        for j in range(env.board_size):
            if env.board[i, j] == 0:  # 只显示空位的概率
                plt.text(j, i, f'{policy_2d[i, j]:.2f}', ha='center', va='center', 
                         color='white' if policy_2d[i, j] > 0.3 else 'black', 
                         fontsize=8)
    
    plt.title("AI策略分布")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def visualization_test(ai_agent, board_size=9):
    """策略可视化测试"""
    print("\n=== 策略可视化测试 ===")
    
    while True:
        print("\n选择局面类型:")
        print("1. 进攻局面")
        print("2. 防守局面")
        print("3. 复杂局面")
        print("4. 自定义局面")
        print("0. 返回上一级菜单")
        
        choice = input("请选择(0-4): ")
        
        test_env = GomokuEnv(board_size=board_size)
        test_env.reset()
        
        if choice == '1':
            # 进攻局面
            position = [
                (3, 3, 1), (3, 4, 1),                # 黑方两连
                (4, 3, 2), (4, 4, 2), (4, 5, 2)      # 白方三连
            ]
            player = 1  # 黑方走
        
        elif choice == '2':
            # 防守局面
            position = [
                (3, 3, 1), (3, 4, 1), (3, 5, 1),     # 黑方三连
                (4, 3, 2), (4, 4, 2)                 # 白方两连
            ]
            player = 2  # 白方走
        
        elif choice == '3':
            # 复杂局面
            position = [
                (2, 2, 1), (2, 3, 1), (3, 2, 1), (4, 3, 1), (3, 4, 1),  # 黑子
                (3, 3, 2), (4, 4, 2), (2, 4, 2), (4, 2, 2), (5, 5, 2)   # 白子
            ]
            player = 1  # 黑方走
        
        elif choice == '4':
            # 自定义局面
            position = []
            print("\n请输入自定义局面，格式为 '行 列 玩家'，玩家为1(黑)或2(白)，输入'done'完成")
            
            while True:
                pos_input = input("输入位置 (行 列 玩家) 或 'done': ")
                if pos_input.lower() == 'done':
                    break
                    
                try:
                    x, y, p = map(int, pos_input.split())
                    if 0 <= x < board_size and 0 <= y < board_size and p in [1, 2]:
                        position.append((x, y, p))
                    else:
                        print(f"无效输入，坐标范围: 0-{board_size-1}, 玩家: 1或2")
                except:
                    print("输入格式错误，请按 '行 列 玩家' 格式输入")
            
            if position:
                player = int(input("谁来走棋? (1=黑, 2=白): "))
                if player not in [1, 2]:
                    print("无效的玩家选择，默认黑方走")
                    player = 1
            else:
                print("未设置任何棋子，使用空棋盘")
                player = 1  # 默认黑方先走
                
        elif choice == '0':
            break
        
        else:
            print("无效选项")
            continue
        
        # 设置棋盘
        for (x, y, p) in position:
            test_env.board[x, y] = p
        test_env.current_player = player
        
        # 显示局面
        print("\n当前局面：")
        print(test_env.render())
        
        # 重置智能体并获取一次动作以初始化历史
        ai_agent.reset()
        action = ai_agent.select_action(test_env)
        print(f"AI建议落子在: {action}")
        
        # 可视化策略
        visualize_policy(test_env, ai_agent)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="测试五子棋AI模型")
    parser.add_argument("--model", type=str, default=None, help="模型文件路径")
    parser.add_argument("--board_size", type=int, default=9, help="棋盘大小 (9 或 15)")
    parser.add_argument("--res_blocks", type=int, default=4, help="残差块数量")
    parser.add_argument("--test", type=str, default=None, 
                        choices=["random", "self", "human", "specific", "visual"], 
                        help="直接运行特定测试")
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    # 加载模型
    try:
        model, ai_agent = load_model(
            model_path=args.model, 
            board_size=args.board_size, 
            num_res_blocks=args.res_blocks
        )
    except Exception as e:
        print(f"模型加载失败: {str(e)}")
        return
    
    # 如果指定了特定测试
    if args.test:
        if args.test == "random":
            vs_random_test(ai_agent, board_size=args.board_size)
            return
        elif args.test == "self":
            self_play_test(ai_agent, board_size=args.board_size)
            return
        elif args.test == "human":
            human_play_test(ai_agent, board_size=args.board_size)
            return
        elif args.test == "specific":
            specific_test(ai_agent, board_size=args.board_size)
            return
        elif args.test == "visual":
            visualization_test(ai_agent, board_size=args.board_size)
            return
    
    # 主菜单
    while True:
        print("\n=== 五子棋AI测试工具 ===")
        print("1. 与随机智能体对战")
        print("2. AI自我对弈")
        print("3. 人机对战")
        print("4. 测试特定局面")
        print("5. 可视化决策过程")
        print("0. 退出")
        
        choice = input("请输入选项(0-5): ")
        
        if choice == '1':
            vs_random_test(ai_agent, board_size=args.board_size)
        elif choice == '2':
            self_play_test(ai_agent, board_size=args.board_size)
        elif choice == '3':
            human_play_test(ai_agent, board_size=args.board_size)
        elif choice == '4':
            specific_test(ai_agent, board_size=args.board_size)
        elif choice == '5':
            visualization_test(ai_agent, board_size=args.board_size)
        elif choice == '0':
            print("感谢使用!")
            break
        else:
            print("无效选项，请重新选择")

if __name__ == "__main__":
    main()