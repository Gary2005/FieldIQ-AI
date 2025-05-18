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

if __name__ == "__main__":
    pass