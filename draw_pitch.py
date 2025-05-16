import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# 创建画布和坐标轴
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_aspect('equal')
ax.set_facecolor('#3b7843')  # 草绿色
ax.set_xlim(-60, 60)
ax.set_ylim(-40, 40)
ax.axis('off')

# 球场边界
ax.plot([-52.5, -52.5], [-34, 34], color='white', linewidth=2)  # 左边线
ax.plot([52.5, 52.5], [-34, 34], color='white', linewidth=2)     # 右边线
ax.plot([-52.5, 52.5], [34, 34], color='white', linewidth=2)    # 上边线
ax.plot([-52.5, 52.5], [-34, -34], color='white', linewidth=2)  # 下边线

# 中线
ax.plot([0, 0], [-34, 34], color='white', linewidth=2)

# 中圈
centre_circle = plt.Circle((0,0), 9.15, color='white', fill=False, linewidth=2)
ax.add_patch(centre_circle)

# 球门区（小禁区）和大禁区（罚球区）参数计算
goal_post = 3.66  # 门柱半距（7.32m/2）

# 下半场禁区
def draw_penalty_areas(y_pos, direction):
    # 小禁区
    small_box_left = -goal_post - 5.5
    small_box_right = goal_post + 5.5
    small_box_depth = y_pos + direction*5.5
    
    ax.plot([small_box_left, small_box_left], [y_pos, small_box_depth], color='white', linewidth=2)
    ax.plot([small_box_right, small_box_right], [y_pos, small_box_depth], color='white', linewidth=2)
    ax.plot([small_box_left, small_box_right], [small_box_depth, small_box_depth], color='white', linewidth=2)
    
    # 大禁区
    big_box_left = -goal_post - 16.5
    big_box_right = goal_post + 16.5
    big_box_depth = y_pos + direction*16.5
    
    ax.plot([big_box_left, big_box_left], [y_pos, big_box_depth], color='white', linewidth=2)
    ax.plot([big_box_right, big_box_right], [y_pos, big_box_depth], color='white', linewidth=2)
    ax.plot([big_box_left, big_box_right], [big_box_depth, big_box_depth], color='white', linewidth=2)
    
    # 罚球点
    penalty_spot_y = y_pos + direction*11
    ax.plot(0, penalty_spot_y, 'wo', markersize=6)
    
    # 罚球弧
    theta = np.linspace(np.pi/2, 3*np.pi/2, 100) if direction == 1 else np.linspace(-np.pi/2, np.pi/2, 100)
    x_arc = 0 + 9.15 * np.cos(theta)
    y_arc = penalty_spot_y + 9.15 * np.sin(theta)
    ax.plot(x_arc, y_arc, color='white', linewidth=2)

# 绘制双方禁区
draw_penalty_areas(-34, 1)  # 下半场
draw_penalty_areas(34, -1)  # 上半场

# 角球区
corner_radius = 1
corners = [(52.5,34), (52.5,-34), (-52.5,34), (-52.5,-34)]
angles = [(270,360), (0,90), (180,270), (90,180)]

for (x,y), (t1,t2) in zip(corners, angles):
    arc = mpatches.Arc((x,y), 2*corner_radius, 2*corner_radius,
                       angle=0, theta1=t1, theta2=t2, color='white', linewidth=2)
    ax.add_patch(arc)

# 球门绘制
for y in [-34, 34]:
    ax.plot([-goal_post, goal_post], [y, y], color='white', linewidth=4)

# show the plot
plt.title("Football Pitch", fontsize=20, color='white')
plt.show()
