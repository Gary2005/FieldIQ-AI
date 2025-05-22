import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
import numpy as np
import random

def get_random_color(seed_value):
    """基于种子生成随机颜色，确保相同名字颜色一致"""
    random.seed(seed_value)
    return (random.random(), random.random(), random.random())

def get_pitch(positions, running_direction, save_dir = 'soccer_pitch.png'):
    # 场地尺寸（米）
    field_length = 105
    field_width = 68

    # 中圈
    center_circle_radius = 9.15

    # 禁区尺寸
    penalty_area_width = 40.32
    penalty_area_length = 16.5

    # 小禁区尺寸
    goal_area_width = 18.32
    goal_area_length = 5.5

    # 罚球点距离球门
    penalty_spot_distance = 11

    # 罚球弧半径
    penalty_arc_radius = 9.15

    # 球门尺寸
    goal_width = 7.32
    goal_post_depth = 0.12  # 12厘米

    # 角球弧半径
    corner_arc_radius = 1

    # 创建画布
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(-field_length / 2 - 20, field_length / 2 + 20)  # 扩展边缘，放文字
    ax.set_ylim(-field_width / 2 - 5, field_width / 2 + 5)
    ax.set_aspect('equal')
    ax.axis('off')

    # --- 画场地外框 ---
    outer_rect = patches.Rectangle((-field_length / 2, -field_width / 2),
                                   field_length, field_width,
                                   fill=False, color='black', lw=2)
    ax.add_patch(outer_rect)

    # --- 中线 ---
    ax.plot([0, 0], [-field_width / 2, field_width / 2], color='black', lw=2)

    # --- 中圈 ---
    center_circle = patches.Circle((0, 0), 9.15,
                                   fill=False, color='black', lw=2)
    ax.add_patch(center_circle)

    # --- 中点 ---
    ax.plot(0, 0, 'ko', markersize=2)

    # --- 绘制球员和运动方向 ---
    Boundary = []
    for player, pos in positions.items():

        if "-" not in player:
            Boundary.append(pos)
            continue

        color = get_random_color(hash(player))
        
        # 1️⃣ 绘制球员位置
        ax.plot(pos[0], pos[1], 'o', color=color, markersize=5)

        # 2️⃣ 根据左右半场，名字写到外面
        if pos[0] < 0:  # 左半场
            text_x = -field_length / 2 - 15
            alignment = 'right'
        else:  # 右半场
            text_x = field_length / 2 + 15
            alignment = 'left'

        # # # 3️⃣ 绘制名字
        # ax.text(text_x, pos[1], player, fontsize=10, color=color,
        #         ha=alignment, va='center',
        #         bbox=dict(facecolor='white', edgecolor=color, boxstyle='round,pad=0.3'))

        # # # 4️⃣ 绘制连接线
        # line = lines.Line2D([pos[0], text_x], [pos[1], pos[1]], color=color, lw=0.8, linestyle='--')
        # ax.add_line(line)

    # 绘制摄像区域(Boundary)
    assert len(Boundary) == 4, "Boundary should have 4 points"
    Boundary = np.array(Boundary)

    # Boundary 不能超过球场范围
    Boundary[Boundary[:, 0] < -field_length / 2, 0] = -field_length / 2
    Boundary[Boundary[:, 0] > field_length / 2, 0] = field_length / 2
    Boundary[Boundary[:, 1] < -field_width / 2, 1] = -field_width / 2
    Boundary[Boundary[:, 1] > field_width / 2, 1] = field_width / 2
    

    ax.plot([Boundary[0][0], Boundary[1][0]], [Boundary[0][1], Boundary[1][1]], 'r--', lw=2)
    ax.plot([Boundary[2][0], Boundary[3][0]], [Boundary[2][1], Boundary[3][1]], 'r--', lw=2)
    ax.plot([Boundary[0][0], Boundary[2][0]], [Boundary[0][1], Boundary[2][1]], 'r--', lw=2)
    ax.plot([Boundary[1][0], Boundary[3][0]], [Boundary[1][1], Boundary[3][1]], 'r--', lw=2)

    # --- 绘制运动方向 ---
    for player, direction in running_direction.items():
        color = get_random_color(hash(player))
        ax.arrow(positions[player][0], positions[player][1],
                 direction[0] * 3, direction[1] * 3, head_width=1, head_length=1,
                 fc=color, ec=color, lw=2, length_includes_head=True)
        
    # --- 左右两端的罚球区、小禁区、球门、罚球弧 ---
    for side in [-1, 1]:  # -1代表左边，1代表右边
        # 罚球区
        penalty_area = patches.Rectangle(
            (side * field_length / 2 - side * penalty_area_length, -penalty_area_width / 2),
            penalty_area_length * side, penalty_area_width,
            fill=False, color='black', lw=2)
        ax.add_patch(penalty_area)

        # 小禁区
        goal_area = patches.Rectangle(
            (side * field_length / 2 - side * goal_area_length, -goal_area_width / 2),
            goal_area_length * side, goal_area_width,
            fill=False, color='black', lw=2)
        ax.add_patch(goal_area)

        # 罚球点
        penalty_spot = (side * field_length / 2 - side * penalty_spot_distance, 0)
        ax.plot(*penalty_spot, 'ko', markersize=2)

        # 罚球弧
        arc = patches.Arc(
            penalty_spot, width=2 * penalty_arc_radius, height=2 * penalty_arc_radius,
            angle=0,
            theta1=308 if side == -1 else 128,  # 角度控制弧的起止范围，右门左门不同
            theta2=52 if side == -1 else 232,
            color='black', lw=2)
        ax.add_patch(arc)

        # 球门
        goal = patches.Rectangle(
            (side * field_length / 2, -goal_width / 2),
            side * goal_post_depth, goal_width,
            fill=False, color='black', lw=2)
        ax.add_patch(goal)

    # --- 保存图片 ---
    filename = save_dir
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved visualization as: {filename}")

def get_pitch_from_pt(features, value=None, best_position = None):
    """
    根据输入特征绘制足球场
    :param features: Tensor of shape (mx_len, 3), 每一行包含 (x, y, team_id)
    :param value: 可选参数，如果不是 None，则是一个 (mx_len, 1) 的 Tensor，写在每个球员头上的值
    :param best_position: 可选参数，表示最佳位置的坐标, 是一个(mx_len, 2)的 Tensor
    :return: Matplotlib Figure 对象
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    plt.close('all')
    
    # 场地尺寸（米）
    field_length, field_width = 105, 68

    # 区域和标志的尺寸
    penalty_area_width, penalty_area_length = 40.32, 16.5
    goal_area_width, goal_area_length = 18.32, 5.5
    penalty_spot_distance = 11
    penalty_arc_radius = 9.15
    goal_width = 7.32
    goal_post_depth = 0.12

    # 颜色映射
    color_map = {0: 'red', 1: 'blue', -1: 'yellow'}

    # 创建画布
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(-field_length / 2, field_length / 2)
    ax.set_ylim(-field_width / 2, field_width / 2)
    ax.set_aspect('equal')
    ax.axis('off')

    # === 1️⃣ 绘制球场外框、中心线、中心圆 ===
    outer_rect = patches.Rectangle((-field_length / 2, -field_width / 2), field_length, field_width, 
                                   fill=False, color='black', lw=2)
    ax.add_patch(outer_rect)

    # 中线
    ax.plot([0, 0], [-field_width / 2, field_width / 2], color='black', lw=2)

    # 中圈
    center_circle = patches.Circle((0, 0), 9.15, fill=False, color='black', lw=2)
    ax.add_patch(center_circle)

    # 中点
    ax.plot(0, 0, 'ko', markersize=2)

    # === 2️⃣ 绘制左右两端的罚球区、小禁区、球门、罚球弧 ===
    for side in [-1, 1]:  # -1代表左边，1代表右边
        # 罚球区
        penalty_area = patches.Rectangle(
            (side * field_length / 2 - side * penalty_area_length, -penalty_area_width / 2),
            penalty_area_length * side, penalty_area_width,
            fill=False, color='black', lw=2)
        ax.add_patch(penalty_area)

        # 小禁区
        goal_area = patches.Rectangle(
            (side * field_length / 2 - side * goal_area_length, -goal_area_width / 2),
            goal_area_length * side, goal_area_width,
            fill=False, color='black', lw=2)
        ax.add_patch(goal_area)

        # 罚球点
        penalty_spot = (side * field_length / 2 - side * penalty_spot_distance, 0)
        ax.plot(*penalty_spot, 'ko', markersize=2)

        # 罚球弧
        arc = patches.Arc(
            penalty_spot, width=2 * penalty_arc_radius, height=2 * penalty_arc_radius,
            angle=0,
            theta1=308 if side == -1 else 128,
            theta2=52 if side == -1 else 232,
            color='black', lw=2)
        ax.add_patch(arc)

        # 球门
        goal = patches.Rectangle(
            (side * field_length / 2, -goal_width / 2),
            side * goal_post_depth, goal_width,
            fill=False, color='black', lw=2)
        ax.add_patch(goal)
    import numpy as np
    from matplotlib import cm
    import matplotlib.colors as mcolors

    # 固定的颜色列表（可扩展）
    line_colors = [
        'red', 'orange', 'green', 'cyan', 'blue', 'purple',
        'brown', 'magenta', 'olive', 'teal', 'pink', 'gray'
    ]

    # 筛选有效球员并获取分数
    valid_players = []
    for idx, player in enumerate(features):
        x, y, team_id = player.tolist()
        x_field = x
        y_field = y

        if team_id == -1:
            ax.plot(x_field, y_field, 'o', color='green', markersize=10)

        if (x == 0 and y == 0):
            continue
        if int(team_id) not in color_map:
            continue
        if value is None or idx >= len(value) or value[idx] == 1000:
            continue

        score = value[idx].item()
        valid_players.append((idx, x_field, y_field, int(team_id), score))
    
    if best_position is not None:
        for (idx, x, y, team_id, score) in valid_players:
            if idx >= best_position.shape[0]:
                continue
            best_x, best_y = best_position[idx].tolist()
            best_x_field = x + best_x
            best_y_field = y + best_y
            color = color_map[team_id]
            ax.annotate("",
                        xy=(best_x_field, best_y_field),
                        xytext=(x, y),
                        arrowprops=dict(arrowstyle='->',
                                        linestyle='dashed',
                                        color=color,
                                        lw=1.5))

    # 按分数排序（高到低）
    valid_players.sort(key=lambda x: -x[4])
    # 分组
    team0_players = [p for p in valid_players if p[3] == 0]
    team1_players = [p for p in valid_players if p[3] == 1]

    # 左右文字区域的横坐标
    label_x_left = -field_length / 2 - 5
    label_x_right = field_length / 2 + 5

    # 生成左右的纵坐标（从上到下排）
    label_ys_left = np.linspace(field_width / 2 - 5, -field_width / 2 + 5, len(team0_players))
    label_ys_right = np.linspace(field_width / 2 - 5, -field_width / 2 + 5, len(team1_players))

    # 左边（team 0）
    for label_y, (idx, x, y, team_id, score) in zip(label_ys_left, team0_players):
        color = color_map[team_id]
        line_color = line_colors[idx % len(line_colors)]
        ax.plot(x, y, 'o', color=color, markersize=8)
        ax.text(label_x_left, label_y, f'{score:.2f}', fontsize=12, ha='right', va='center', color=line_color)
        ax.plot([x, label_x_left + 1], [y, label_y], color=line_color, lw=1, linestyle='--')

    # 右边（team 1）
    for label_y, (idx, x, y, team_id, score) in zip(label_ys_right, team1_players):
        color = color_map[team_id]
        line_color = line_colors[idx % len(line_colors)]
        ax.plot(x, y, 'o', color=color, markersize=8)
        ax.text(label_x_right, label_y, f'{score:.2f}', fontsize=12, ha='left', va='center', color=line_color)
        ax.plot([x, label_x_right - 1], [y, label_y], color=line_color, lw=1, linestyle='--')
    return fig

if __name__ == "__main__":
    
    from torch.utils.data import DataLoader
    import torch.optim as optim
    from dataset import SoccerDataset
    from model import SoccerTransformer
    import torch
    import torch.nn as nn
    from tqdm import tqdm
    import wandb
    import json
    import os
    import random

    def process_data(json_paths):
        data = []
        for json_path in json_paths:
            with open(json_path, "r") as f:
                match_data = json.load(f)
                data.extend(match_data)

        # shuffle 后选择 512 条数据作为测试集
        random.shuffle(data)
        test_data = data[:512]
        data = data[512:]

        return data, test_data
        
    config = {
        "batch_size": 64,
        "learning_rate": 1e-3,
        "epochs": 10,
        "d_model": 16,
        "nhead": 4,
        "num_layers": 2,
        "max_len": 20,
        "valid_step": 100,
    }

    json_paths = ["game_example/cleaned_data.json"]
    data, test_data = process_data(json_paths)
    train_dataset = SoccerDataset(data, mx_len=config["max_len"])
    test_dataset = SoccerDataset(test_data, mx_len=config["max_len"])
    dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    dataloader_test = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    test_features, test_target = test_dataset[0]
    # print(test_features.shape, test_target.shape)
    # print(test_features)
    # print(test_target)
    image = get_pitch_from_pt(test_features)
    # save the image 
    image.savefig("test.png", dpi=300, bbox_inches='tight')