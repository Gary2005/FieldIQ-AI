import json

path = "game_example"

label_path = f"{path}/Labels-v2.json"
data_path = f"{path}/data.json"

labels = json.load(open(label_path, "r"))
frame_informations = json.load(open(data_path, "r"))

if labels["LeftFirstHalf"] == "away":
    to_lr = {
        "away": "left",
        "home": "right"
    }
else:
    to_lr = {
        "away": "right",
        "home": "left"
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
        reward_for_left = rewards[label["label"]][0] if to_lr[label["team"]] == "left" else rewards[label["label"]][1]
        text = label["gameTime"]
        match = re.match(r"(\d+)\s*-\s*(\d{2}):(\d{2})", text)
        if match:
            index = int(match.group(1))
            minutes = int(match.group(2))
            seconds = int(match.group(3))
        else:
            raise ValueError("字符串格式不正确")
        
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
    
    left_x = []
    right_x = []
    for item, pos in data["positions"].items():
        if "left" in item:
            left_x.append(pos[0])
        elif "right" in item:
            right_x.append(pos[0])

    mean_left_x = np.mean(left_x) / 52.5
    mean_right_x = np.mean(right_x) / 52.5

    player_position_reward[index-1][seconds] = (mean_left_x + mean_right_x) / 2

# we want position reward to be continuous
def make_continuous(position_reward):
    """
    将离散的 position reward 变为连续值，使用相邻非零值的线性插值。
    """
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
    
    return continuous_reward
ball_position_reward = make_continuous(ball_position_reward)
player_position_reward = make_continuous(player_position_reward)


alpha = 2
beta = 2

rewards = alpha * ball_position_reward + beta * player_position_reward + sparse_rewards

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

plot_and_save(ball_position_reward, "Ball Position Reward", "plot/ball_position_reward.png")
plot_and_save(player_position_reward, "Player Position Reward", "plot/player_position_reward.png")
plot_and_save(sparse_rewards, "Sparse Rewards", "plot/sparse_rewards.png")
plot_and_save(rewards, "Total Reward", "plot/total_reward.png")