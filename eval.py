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


pth_see_all = "checkpoints/see_all/model_epoch_1_step_1600_loss_0.1740.pth"
pth_mask_right = "checkpoints/mask_one_right/model_epoch_1_step_9100_loss_0.1739.pth"
pth_mask_left = "checkpoints/mask_one_left/model_epoch_1_step_6900_loss_0.1449.pth"
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
    "max_len": 23,
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

    return players_info

def customize_data(data_dict):
    data = SoccerDataset([(data_dict, 0)], mx_len=config["max_len"], data_augmentation=False)
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

    # player_feature, target, mask = sample_data(21475, "first")
    # dict_ = sample_data(23600, "first")
    dict_ =[
{'x': 22, 'y': -12, 'vx': 0, 'vy': 0, 'team_id': 0},
{'x': 22, 'y': -22, 'vx': 0, 'vy': 0, 'team_id': 1},
{'x': 36, 'y': -22, 'vx': 0, 'vy': 0, 'team_id': 1},
{'x': 38, 'y': -23, 'vx': 0, 'vy': 0, 'team_id': 0},
{'x': 42, 'y': -20, 'vx': 0, 'vy': 0, 'team_id': 1},
{'x': 42, 'y': -22, 'vx': 0, 'vy': 0, 'team_id': 0},
{'x': 42.3, 'y': -21.5, 'vx': 0, 'vy': 0, 'team_id': -1},
{'x': 40, 'y': -16, 'vx': 0, 'vy': 0, 'team_id': 1},
{'x': 42, 'y': -2, 'vx': 0, 'vy': 0, 'team_id': 0},
{'x': 43, 'y': -1, 'vx': 0, 'vy': 0, 'team_id': 1},
{'x': 36, 'y': 0, 'vx': 0, 'vy': 0, 'team_id': 0},
{'x': 37, 'y': 1, 'vx': 0, 'vy': 0, 'team_id': 1},
]

    print("[")
    for item in dict_:
        print(f"{{'x': {item['x']}, 'y': {item['y']}, 'vx': {item['vx']}, 'vy': {item['vy']}, 'team_id': {item['team_id']}}},")
    print("]")

    player_feature, target, mask = customize_data(dict_)

    print(f"player_feature: {player_feature}")
    print(f"target: {target}")
    print(f"mask: {mask}")
    with torch.no_grad():
        predict = model_see_all(player_feature.unsqueeze(0).to(device), mask=mask.unsqueeze(0).to(device))[0].item()
    target = target.item()
    print(f"predict: {predict}, target: {target}")

    values = torch.zeros((len(player_feature)))
    
    for i in range(len(values)):
        if i< len(dict_):
            values[i] = dict_[i]["x"]
        if i< len(dict_):

            if dict_[i]["team_id"] == -1:   
                values[i] = 1000
                continue

            assert mask[i] == 0

            new_mask = mask.clone().detach()

            model = model_mask_left if dict_[i]["team_id"] == 0 else model_mask_right
            new_mask[i] = 1
            with torch.no_grad():
                predict2 = model(player_feature.unsqueeze(0).to(device), mask=new_mask.unsqueeze(0).to(device))[0].item()

            values[i] = predict2
            if dict_[i]["team_id"] == 1:
                values[i] = -values[i]
        else:
            assert mask[i] == 1
            values[i] = 1000

    values_0 = [values[i] for i in range(len(dict_)) if dict_[i]["team_id"] == 0]
    values_1 = [values[i] for i in range(len(dict_)) if dict_[i]["team_id"] == 1]
    values_0 = np.mean(values_0)
    values_1 = np.mean(values_1)
    for i in range(len(values)):
        if values[i] != 1000:
            if dict_[i]["team_id"] == 0:
                values[i] = np.mean(values_0) - values[i]
            else:
                assert dict_[i]["team_id"] == 1
                values[i] = np.mean(values_1) - values[i]

    print(values)

    best_position = []
    for i in range(len(values)):
        if i< len(dict_):
            if dict_[i]["team_id"] == -1:
                best_position.append([0, 0])
                continue
            best_value = -1e18
            flag = 1
            if dict_[i]["team_id"] == 1:
                flag = -1

            dx_ = None
            dy_ = None
            for d_x in range(-2, 3):
                for d_y in range(-2, 3):
                    dx =d_x/( 105/2)
                    dy=d_y/( 68/2)
                    temp_player_feature = player_feature.clone().detach()
                    temp_player_feature[i][0] += dx
                    temp_player_feature[i][1] += dy
                    temp_predict = model_see_all(temp_player_feature.unsqueeze(0).to(device), mask=mask.unsqueeze(0).to(device))[0].item()
                    if temp_predict * flag > best_value:
                        best_value = temp_predict * flag
                        dx_ = dx
                        dy_ = dy
            print(f"best_value: {best_value}, dx: {dx_ * (105/2)}, dy: {dy_ * (68/2)}")
            best_position.append([dx_, dy_])
            # values[i] = predict * flag - best_value
                    
        else:
            best_position.append([0, 0])
    best_position = np.array(best_position)

    print(values)


    img = get_pitch_from_pt(player_feature, values, None)

    # save the image to output_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    img.savefig(os.path.join(output_path, "2d.png"))