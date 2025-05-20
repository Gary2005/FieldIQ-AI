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


pth_see_all = "checkpoints/see_all/model_epoch_19_step_997_loss_0.0146.pth"
pth_mask_right = "checkpoints/mask_one_right/model_epoch_17_step_2900_loss_0.0122.pth"
pth_mask_left = "checkpoints/mask_one_left/model_epoch_18_step_100_loss_0.0149.pth"
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
    # dict_ = sample_data(21475, "first")
    dict_ = [
{'x': -12.648116144101763, 'y': -8.139046497704577, 'vx': -1.9654739361401496, 'vy': -1.3262491554302258, 'team_id': 0},
{'x': -8.438931398614654, 'y': -30.12683753997186, 'vx': -3.1936985573828736, 'vy': 1.8112681907215489, 'team_id': 0},
{'x': -12.333323749561744, 'y': -22.164005930687416, 'vx': -2.4148030765080986, 'vy': 0.3632563121543697, 'team_id': 0},
{'x': -11, 'y': -23.5, 'vx': -2.4148030765080986, 'vy': 0.3632563121543697, 'team_id': 1},
{'x': 9.239549925139237, 'y': -26.232393321427384, 'vx': -1.6668847777242757, 'vy': 1.2909691709088733, 'team_id': 1},
{'x': -0.6149116078294568, 'y': -29.969823645552697, 'vx': -1.4057139103501286, 'vy': -0.9958415202281579, 'team_id': 1},
{'x': 10.888981035455927, 'y': 2.311794527693397, 'vx': -1.3010191129655624, 'vy': 0.19917491142373844, 'team_id': 1},
{'x': 1.035940929714013, 'y': -14.838461614527528, 'vx': -2.9504722498432665, 'vy': -0.036165495814710624, 'team_id': 0},
{'x': -1.682838726066301, 'y': -20.39575485399618, 'vx': -2.5403609929358364, 'vy': 1.4428561117928673, 'team_id': 1},
{'x': 12.170476891885594, 'y': -9.389198972816118, 'vx': -3.04346523892689, 'vy': 0.6021077597432978, 'team_id': 0},
{'x': 10.868120202614069, 'y': -17.39982712919572, 'vx': -3.407396886184255, 'vy': -0.231002013763959, 'team_id': 0},
{'x': -3.8026378116594195, 'y': 6.027900316550121, 'vx': -3.3055259024701966, 'vy': -3.973277983768919, 'team_id': 0},
{'x': 12.83106568106772, 'y': -19.62184632929877, 'vx': 1.5999415306360731, 'vy': 1.6130285478492823, 'team_id': 1},
{'x': -1.8, 'y': -20.39575485399618, 'vx': -2.5403609929358364, 'vy': 1.4428561117928673, 'team_id': -1},
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
        predict = model_see_all(torch.tensor(player_feature).unsqueeze(0).to(device), mask=mask.unsqueeze(0).to(device))[0].item()
    target = target.item()
    print(f"predict: {predict}, target: {target}")

    values = torch.zeros((len(player_feature)))
    
    for i in range(len(values)):
        if i< len(dict_):
            values[i] = dict_[i]["x"]
        else:
            values[i] = 100

    print(values)

    img = get_pitch_from_pt(player_feature, values)

    # save the image to output_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    img.savefig(os.path.join(output_path, "2d.png"))