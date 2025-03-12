# 基于视觉+三因素融合的五子棋AI

一个基于深度学习的五子棋（Gomoku）AI项目，使用视觉特征提取和三因素融合机制（滑向、急迫度、策略思考）。

> **注意：** 此项目正在积极开发中，功能和代码可能会有较大变动。

## 简介

本项目基于纯神经网络实现五子棋AI，区别于传统的蒙特卡洛树搜索(MCTS)方法。核心思想是将视觉特征与三种因素相融合：
- 滑向机制：引导AI在相关棋子周围落子
- 急迫度：优先考虑能快速带来评估值提升的位置
- 策略思考：基于历史动态调整策略权重

## 安装

```bash
# 克隆项目
git clone https://github.com/Karesis/Self.git

# 安装依赖
pip install -r requirements.txt
```

## 使用方法

### 与AI对战

```bash
python test.py --test human --board_size 15
```

### 训练模型

```bash
python train.py --board_size 15 --epochs 1000
```

### 参数说明

训练和测试脚本支持多种参数配置，详情可查看各脚本的帮助信息：

```bash
python train.py --help
python test.py --help
```

## 技术文档

相关技术文档和图表位于同级目录下，主要文件:

- `gomoku-tech-doc.md` - 技术设计文档
- 模型架构图和训练过程图 (生成于代码分析)

## 主要文件

- `gomokuenv.py` - 五子棋环境，实现游戏规则和状态管理
- `model.py` - 神经网络模型定义
- `train.py` - 训练脚本
- `test.py` - 测试和评估脚本

## 计划改进

- 简化模型架构，提高训练稳定性
- 优化训练流程，提高效率
- 改进评估方法

## 许可证

Apache License 2.0
