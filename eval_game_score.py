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
from pitch import get_pitch_from_pt
import numpy as np
pth_see_all = "final_checkpoints/see_all/model_latest.pth"
config = {
    "name": "see_all",
    "batch_size": 128,
    "learning_rate": 1e-4,
    "epochs": 30,
    "d_model": 128,
    "nhead": 8,
    "num_layers": 4,
    "max_len": 23,
    "valid_step": 100,
    "visualize_sample": 8,
    "weight_decay": 1e-4,
}
device = "cuda:7"

data_path = "2016-09-16 - 22-00 Chelsea 1 - 2 Liverpool/cleaned_data.json"

def process_data(json_path):
    data = []
    with open(json_path, "r") as f:
        match_data = json.load(f)
        data.extend(match_data)
    return data


values = np.zeros((2, 45*60))
values_target = np.zeros((2, 45*60))

data = process_data(data_path)

model_see_all = SoccerTransformer(
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_layers=config["num_layers"],
        max_len=config["max_len"]
    ).to(device)

model_see_all.load_state_dict(torch.load(pth_see_all, map_location=device))
model_see_all.eval()

diff = []

for i in tqdm(range(len(data))):
    if len(data[i]) == 0:
        continue
    if len(data[i][0]) == 0:
        continue
    ele = data[i][0][0]
    target = data[i][1]
    # print(ele)
    if int(ele["time"]) >= 45*60:
        continue
    if values[ele["half"]][ele["time"]] == 0:
        data_ = [data[i]]
        dataset = SoccerDataset(data_, mx_len=config["max_len"], data_augmentation=False)
        assert len(dataset) <= 1
        if len(dataset) == 0:
            continue
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        for batch in dataloader:
            players_features, target_tensor, padding_mask = batch
            players_features = players_features.to(device)
            padding_mask = padding_mask.to(device)
            values[ele["half"]][ele["time"]] = model_see_all(players_features, padding_mask)[0].item()
            # print(values[ele["half"]][ele["time"]])
            values_target[ele["half"]][ele["time"]] = target
            diff.append((values[ele["half"]][ele["time"]] - target)**2)

print("MSE: ", np.mean(diff))


import matplotlib.pyplot as plt
import numpy as np

# 时间坐标轴
time_axis = np.arange(45 * 60)

# 掩码去除未填充的时间点（预测值和目标值都为 0 的点）
mask_first_half = values[0] != 0
mask_second_half = values[1] != 0

# 创建一个包含两个子图的图像
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# 上半场图
axes[0].plot(time_axis[mask_first_half], values[0][mask_first_half], color="blue", label="Predicted")
axes[0].plot(time_axis[mask_first_half], values_target[0][mask_first_half], color="orange", linestyle="--", label="Target")
axes[0].set_title("First Half Reward Over Time")
axes[0].set_ylabel("Reward")
axes[0].legend()
axes[0].grid(True)

# 下半场图
axes[1].plot(time_axis[mask_second_half], values[1][mask_second_half], color="red", label="Predicted")
axes[1].plot(time_axis[mask_second_half], values_target[1][mask_second_half], color="green", linestyle="--", label="Target")
axes[1].set_title("Second Half Reward Over Time")
axes[1].set_xlabel("Time (seconds)")
axes[1].set_ylabel("Reward")
axes[1].legend()
axes[1].grid(True)

# 调整布局并保存
plt.tight_layout()
plt.savefig("reward_curve.png")