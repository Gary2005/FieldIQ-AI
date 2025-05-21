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


pth_see_all = "checkpoints/see_all/latest/model_latest.pth"
pth_mask_right = "checkpoints/mask_one_right/latest/model_latest.pth"
pth_mask_left = "checkpoints/mask_one_left/latest/model_latest.pth"
output_path = "plot"
video_path_first = "game_example/1_720p.mkv"
video_path_second = "game_example/2_720p.mkv"
json_path = "game_example/data.json"

config = {
    "name": "mask_one_left",
    "batch_size": 128,
    "learning_rate": 1e-5,
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
{'x': -14.316568900640467, 'y': -9.783667077827891, 'vx': -4.557202273868111, 'vy': 3.531992335658307, 'team_id': 0},
{'x': -16.12526326025321, 'y': 1.4792967568161055, 'vx': -4.777759023919526, 'vy': 5.375112968657553, 'team_id': 0},
{'x': -14.276554933798739, 'y': 14.727766403174268, 'vx': -4.738958236427937, 'vy': 4.705132353296282, 'team_id': 0},
{'x': -7.206119969380019, 'y': 3.9929892579050255, 'vx': -4.293126413366388, 'vy': 3.0733821797480743, 'team_id': 0},
{'x': -1.4397939423980124, 'y': 9.307832360176183, 'vx': -2.012896328088065, 'vy': 1.6797217783507623, 'team_id': 1},
{'x': -1.3259927316192313, 'y': -12.729802012021594, 'vx': -3.3779267401997606, 'vy': 0.6574615367628933, 'team_id': 1},
{'x': -4.534249277782724, 'y': 4.594560995763602, 'vx': -4.203857594629246, 'vy': 2.1974530734902853, 'team_id': 1},
{'x': -6.685481852390099, 'y': 9.474649597406298, 'vx': -5.088842665815108, 'vy': 5.044995246996953, 'team_id': 0},
{'x': -6.306445028782631, 'y': 22.352032977186973, 'vx': -4.5775775021133835, 'vy': 3.7004519110953815, 'team_id': 1},
{'x': -14.802488080643963, 'y': 2.787400285096983, 'vx': -4.4052729008707825, 'vy': 2.413813197191672, 'team_id': 1},
{'x': 7.036515938003775, 'y': 14.57116654808866, 'vx': -1.321232243387871, 'vy': -0.6112222602602024, 'team_id': 0},
{'x': 5.161560505689675, 'y': 5.450439408554166, 'vx': -2.5033096532446786, 'vy': -0.45584892329553206, 'team_id': 0},
{'x': -16.43171705854705, 'y': 8.602372599250419, 'vx': -4.590090410251957, 'vy': 0.21086738692113194, 'team_id': 0},
{'x': -8.184618006720921, 'y': 10.607565830206047, 'vx': -3.810884218647459, 'vy': 4.142518202433942, 'team_id': 1},
{'x': -8.184618006720921 - 0.5, 'y': 10.607565830206047 + 0.5, 'vx': -3.810884218647459, 'vy': 4.142518202433942, 'team_id': -1},
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


    img = get_pitch_from_pt(player_feature, values, best_position=best_position)

    # save the image to output_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    img.savefig(os.path.join(output_path, "2d.png"))