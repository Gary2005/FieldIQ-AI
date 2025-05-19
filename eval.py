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
import cv2

# ===============================
# 配置超参数
# ===============================


pth_see_all = "checkpoints/see_all/model_epoch_31_step_0_loss_0.0252.pth"
pth_mask_right = "checkpoints/mask_one_right/model_epoch_34_step_300_loss_0.0233.pth"
pth_mask_left = "checkpoints/mask_one_left/model_epoch_52_step_100_loss_0.0218.pth"
output_path = "plot"
video_path_first = "game_example/1_720p.mkv"
video_path_second = "game_example/2_720p.mkv"
json_path = "game_example/data.json"

config = {
    "name": "see_all",
    "batch_size": 64,
    "learning_rate": 1e-4,
    "epochs": 100,
    "d_model": 16,
    "nhead": 4,
    "num_layers": 2,
    "max_len": 20,
    "valid_step": 100,
    "visualize_sample": 8
}
device = "cuda:7"


def sample_data(frame_idx, half):

    video_path = video_path_first if half == "first" else video_path_second

    cap = cv2.VideoCapture(video_path)
    
    data_json = json.load(open(json_path, "r"))
    element = None
    for item in data_json:
        if item["frame"] == frame_idx and item["half"] == half:
            element = item
            break
    if element is None:
        raise ValueError(f"Frame {frame_idx} not found in {half} half")
    
    # 读取视频帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        raise ValueError(f"Frame {frame_idx} not found in {video_path}")
    # 保存这一帧
    cv2.imwrite(os.path.join(output_path, f"3d.jpg"), frame)


    players_info = []
    for item, pos in element["positions"].items():
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
            "vx": element["directions"][item][0],
            "vy": element["directions"][item][1],
            "team_id": team_id
        })

    
    data = SoccerDataset([(players_info, 0)], mx_len=config["max_len"])
    if len(data) == 0:
        raise ValueError("Unexpected data length")
    players, target, mask = data[0]

    return players, target, mask

if __name__ == "__main__":
    model_see_all = SoccerTransformer(
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_layers=config["num_layers"],
        max_len=config["max_len"]
    ).to(device)

    model_mask_left = SoccerTransformer(
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_layers=config["num_layers"],
        max_len=config["max_len"]
    ).to(device)

    model_mask_right = SoccerTransformer(
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_layers=config["num_layers"],
        max_len=config["max_len"]
    ).to(device)

    model_see_all.load_state_dict(torch.load(pth_see_all, map_location=device))
    model_mask_left.load_state_dict(torch.load(pth_mask_left, map_location=device))
    model_mask_right.load_state_dict(torch.load(pth_mask_right, map_location=device))

    model_see_all.eval()
    model_mask_left.eval()
    model_mask_right.eval()

    player_feature, target, mask = sample_data(21475, "first")

    print(f"player_feature: {player_feature}")
    print(f"target: {target}")
    print(f"mask: {mask}")
    with torch.no_grad():
        predict = model_see_all(torch.tensor(player_feature).unsqueeze(0).to(device), mask=mask.unsqueeze(0).to(device))[0].item()
    target = target.item()
    print(f"predict: {predict}, target: {target}")

    values = torch.zeros((len(player_feature)))

    print(values)

    img = get_pitch_from_pt(player_feature, values)

    # save the image to output_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    img.savefig(os.path.join(output_path, "2d.png"))