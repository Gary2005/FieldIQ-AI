import json

path = "game_example"

label_path = f"{path}/Labels-v2.json"
data_path = f"{path}/data.json"

labels = json.load(open(label_path, "r"))
frame_informations = json.load(open(data_path, "r"))

# 翻转下半场的left,right
new_frame_informations = []
for element in frame_informations:
    new_element = element.copy()
    left_position = {}
    left_direction = {}
    right_position = {}
    right_direction = {}
    ball_direction = {}
    ball_position = {}
    for key, value in element["positions"].items():
        role, index = key.split("-")
        index = int(index)
        if element["half"] == "second":
            if role == "left":
                role = "right"
            elif role == "right":
                role = "left"
        if role == "left":
            left_position[index] = value
            left_direction[index] = element["directions"][key]
        elif role == "right":
            right_position[index] = value
            right_direction[index] = element["directions"][key]
        elif role == "ball":
            ball_position[index] = value
            ball_direction[index] = element["directions"][key]
    
    new_element["positions"] = {}
    new_element["directions"] = {}
    if len(ball_position) > 1:
        # keep the one with max bbox_confs
        max_bbox_conf_index = -1
        for index in ball_position.keys():
            if max_bbox_conf_index == -1 or element["bbox_confs"][index] > element["bbox_confs"][max_bbox_conf_index]:
                max_bbox_conf_index = index
        ball_position = {max_bbox_conf_index: ball_position[max_bbox_conf_index]}
        ball_direction = {max_bbox_conf_index: ball_direction[max_bbox_conf_index]}
    for index, value in left_position.items():
        if value[1] < -34:
            # seen as substituted
            continue
        new_element["positions"][f"left-{index}"] = value
        new_element["directions"][f"left-{index}"] = left_direction[index]
    for index, value in right_position.items():
        if value[1] < -34:
            # seen as substituted
            continue
        new_element["positions"][f"right-{index}"] = value
        new_element["directions"][f"right-{index}"] = right_direction[index]
    for index, value in ball_position.items():
        new_element["positions"][f"ball-{index}"] = value
        new_element["directions"][f"ball-{index}"] = ball_direction[index]

    new_frame_informations.append(new_element)
frame_informations = new_frame_informations

to_lr = {}

if labels["LeftFirstHalf"] == "away":
    to_lr["first"] = {
        "away": "left",
        "home": "right"
    }
    to_lr["second"] = {
        "away": "right",
        "home": "left"
    }
else:
    assert labels["LeftFirstHalf"] == "home", f"Error: {labels['LeftFirstHalf']} is not in [home, away]"
    to_lr["first"] = {
        "home": "left",
        "away": "right"
    }
    to_lr["second"] = {
        "home": "right",
        "away": "left"
    }

rewards = {
    "Throw-in": [1, -1],
    "Foul": [-2, 2],
    "Indirect free-kick": [2, -2],
    "Clearance": [1, -1],
    "Shots on target": [7, -7],
    "Shots off target": [4, -4],
    "Corner": [3, -3],
    "Direct free-kick": [3, -3],
    "Yellow card": [-3, 3],
    "Goal": [20, -20],
    "Penalty": [15, -15],
    "Red card": [-10, 10],
    "Yellow->red card": [-10, 10],
    "Offside": [-1, 1],
}

import numpy as np

ball_position_reward = np.zeros((2, 50*60))
sparse_rewards = np.zeros((2, 50*60))
player_position_reward = np.zeros((2, 50*60))

# notin = set()

# for label in labels["annotations"]:
#     if label["label"] not in rewards:
#         notin.add(label["label"])

# print("notin:", notin)

import re


for label in labels["annotations"]:
    if label["label"] in rewards:
        text = label["gameTime"]
        match = re.match(r"(\d+)\s*-\s*(\d{2}):(\d{2})", text)
        if match:
            index = int(match.group(1))
            minutes = int(match.group(2))
            seconds = int(match.group(3))
        else:
            raise ValueError("字符串格式不正确")
        
        half = "first" if index == 1 else "second"

        reward_for_left = rewards[label["label"]][0] if to_lr[half][label["team"]] == "left" else rewards[label["label"]][1]
        
        sparse_rewards[index-1][minutes*60 + seconds] += reward_for_left

for data in frame_informations:
    text = data["time"]
    index = 1 if data["half"] == "first" else 2

    match = re.match(r"(\d+)\s*\((\d+):(\d+)\)", text)
    if match:
        seconds = int(match.group(1))
    else:
        raise ValueError("字符串格式不正确")
    
    for item, pos in data["positions"].items():
        if "ball" in item:
            ball_position_reward[index-1][seconds] = pos[0]/52.5
    
    _x = []
    for item, pos in data["positions"].items():
        if "left" in item or "right" in item:
            _x.append(pos[0])

    if len(_x) == 0:
        continue
    player_position_reward[index-1][seconds] = np.mean(_x) / 52.5

# we want position reward to be continuous
def make_continuous(position_reward):
    """
    将离散的 position reward 变为连续值，使用相邻非零值的线性插值。
    """

    if np.isnan(position_reward).any():
        raise ValueError("position_reward 中存在 NaN 值")

    continuous_reward = np.zeros_like(position_reward)
    for i in range(position_reward.shape[0]):
        # 找到所有非零值的索引
        non_zero_indices = np.nonzero(position_reward[i])[0]

        if len(non_zero_indices) <= 1:
            continuous_reward[i] = position_reward[i]
            continue
        
        valid_times = non_zero_indices
        valid_values = position_reward[i][valid_times]
        
        continuous_reward[i] = np.interp(
            np.arange(len(position_reward[i])),
            valid_times,
            valid_values
        )
    # 检查是否有nan
    if np.isnan(continuous_reward).any():
        raise ValueError("continuous_reward 中存在 NaN 值")
    return continuous_reward

print("###")

ball_position_reward = make_continuous(ball_position_reward)
print("##")
player_position_reward = make_continuous(player_position_reward)

print("---")


alpha = 2
beta = 2

rewards = alpha * ball_position_reward + beta * player_position_reward + sparse_rewards
print(sparse_rewards)
print(rewards)

import matplotlib.pyplot as plt
import numpy as np

def plot_and_save(data, title, filename):
    plt.figure(figsize=(15, 8))
    
    # 绘制 Half 1
    plt.subplot(2, 1, 1)  # (行, 列, 索引)
    plt.plot(data[0], color='blue')
    plt.title(f"{title} - Half 1")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Reward")
    plt.grid(True, linestyle='--', alpha=0.7)

    # 绘制 Half 2
    plt.subplot(2, 1, 2)
    plt.plot(data[1], color='orange')
    plt.title(f"{title} - Half 2")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Reward")
    plt.grid(True, linestyle='--', alpha=0.7)

    # 调整布局并保存图片
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"{filename} saved.")

import os

if os.path.exists("plot"):
    pass
else:
    os.makedirs("plot")

plot_and_save(ball_position_reward, "Ball Position Reward", "plot/ball_position_reward.png")
plot_and_save(player_position_reward, "Player Position Reward", "plot/player_position_reward.png")
plot_and_save(sparse_rewards, "Sparse Rewards", "plot/sparse_rewards.png")
plot_and_save(rewards, "Total Reward", "plot/total_reward.png")

slide_window_size = 20
ld = 1.1
new_rewards = np.zeros((2, 50*60))


for i in range(2):
    for j in range(50*60):
        for k in range(j, min(j+slide_window_size, 50*60)):
            new_rewards[i][j] += rewards[i][k] / (ld**(k-j))

new_rewards /= 20


plot_and_save(new_rewards, "New Total Reward", "plot/new_total_reward.png")

query_idx = 14 * 60 + 29
queyr_index = 1

left = query_idx - 30
right = query_idx + 30

import matplotlib.ticker as ticker

def format_seconds(x, _):
    """格式化时间轴，秒数转为 MM:SS 格式"""
    minutes = int(x // 60)
    seconds = int(x % 60)
    return f"{minutes:02d}:{seconds:02d}"

def plot_windowed_reward(reward_data, left, right, index, title, filename):
    """
    绘制指定窗口范围内的奖励曲线并保存到文件。
    
    参数:
    - reward_data: np.ndarray，奖励数据，形状为 (2, 时间步)。
    - left: int，窗口的左边界（时间步，单位：秒）。
    - right: int，窗口的右边界（时间步，单位：秒）。
    - index: int，1 表示上半场，2 表示下半场。
    - title: str，图像标题。
    - filename: str，保存的文件名。
    """
    if index not in [1, 2]:
        raise ValueError("index 必须是 1（上半场） 或 2（下半场）")
    
    if left < 0 or right > reward_data.shape[1]:
        raise ValueError("时间窗口越界")

    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(left, right), reward_data[index - 1][left:right], color='purple')
    plt.title(f"{title} - Half {index} ({left} to {right} seconds)")
    plt.xlabel("Time (MM:SS)")
    plt.ylabel("Reward")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 设置横坐标格式为 MM:SS
    ax = plt.gca()
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_seconds))

    # 显示更密集的时间刻度
    plt.xticks(np.arange(left, right, step=10))

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"{filename} saved.")

plot_windowed_reward(rewards, left, right, queyr_index, "Total Reward", "plot/total_reward_window.png")
plot_windowed_reward(new_rewards, left, right, queyr_index, "New Total Reward", "plot/new_total_reward_window.png")


cleaned_data = []

for data in frame_informations:
    text = data["time"]
    index = 1 if data["half"] == "first" else 2

    match = re.match(r"(\d+)\s*\((\d+):(\d+)\)", text)
    if match:
        seconds = int(match.group(1))
    else:
        raise ValueError("字符串格式不正确")
    
    target = new_rewards[index-1][seconds]
    players_info = []
    for item, pos in data["positions"].items():
        if "ball" in item:
            team_id = -1
        elif "left" in item:
            team_id = 0
        elif "right" in item:
            team_id = 1
        else:
            continue
        players_info.append({
            "x": pos[0],
            "y": pos[1],
            "vx": data["directions"][item][0],
            "vy": data["directions"][item][1],
            "team_id": team_id
        })
    cleaned_data.append((players_info, target))
with open("game_example/cleaned_data.json", "w") as f:
    json.dump(cleaned_data, f)
print("cleaned_data.json saved.")