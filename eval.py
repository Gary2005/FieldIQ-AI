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


pth_see_all = "final_checkpoints/see_all/model_latest.pth"
pth_mask_right = "final_checkpoints/mask_one_right/model_latest.pth"
pth_mask_left = "final_checkpoints/mask_one_left/model_latest.pth"
output_path = "plot"
video_path_first = "game_example/1_720p.mkv"
video_path_second = "game_example/2_720p.mkv"
json_path = "game_example/data.json"

config = {
    "name": "mask_one_left",
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
device = "cuda:0"


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
    dict_ = [
{'x': -33.81569163348405, 'y': 10.372437193516635, 'vx': 0, 'vy': 0, 'team_id': 0},
{'x': -32.668128504989305, 'y': 0.5046291993137364, 'vx': 0, 'vy': 0, 'team_id': 0},
{'x': -32.300635941902364, 'y': -5.469187022042354, 'vx': 0, 'vy': 0, 'team_id': 0},
{'x': -32.07621227085955, 'y': -18.2438333181472, 'vx': 0, 'vy': 0, 'team_id': 0},
{'x': -25.865428535676223, 'y': -7.398215326094615, 'vx': 0, 'vy': 0, 'team_id': 0},
{'x': -21.032354698945614, 'y': -19.638708175486197, 'vx': 0, 'vy': 0, 'team_id': 0},
{'x': -19.629123936812693, 'y': -6.803074833103667, 'vx': 0, 'vy': 0, 'team_id': 0},
{'x': -22.354523824193617, 'y': 0.8104035124556981, 'vx': 0, 'vy': 0, 'team_id': 0},
{'x': -15.591689151368888, 'y': -4.0422610464207, 'vx': 0, 'vy': 0, 'team_id': 0},
{'x': -3.6849537415391715, 'y': -14.172569423032208, 'vx': 0, 'vy': 0, 'team_id': 0},
{'x': -32.091195922280036, 'y': 22.247216440384424, 'vx': 0, 'vy': 0, 'team_id': 1},
{'x': -33.693326354737216, 'y': -7.826269342147122, 'vx': 0, 'vy': 0, 'team_id': 1},
{'x': -28.19132198793557, 'y': -23.04750873047852, 'vx': 0, 'vy': 0, 'team_id': 1},
{'x': -23.6085811836369, 'y': -14.421442069768338, 'vx': 0, 'vy': 0, 'team_id': 1},
{'x': -20.623954999992772, 'y': -10.156327635428928, 'vx': 0, 'vy': 0, 'team_id': 1},
{'x': -15.7913931213346, 'y': 5.743510501065166, 'vx': 0, 'vy': 0, 'team_id': 1},
{'x': -16.32212095564353, 'y': 15.184504040563082, 'vx': 0, 'vy': 0, 'team_id': 1},
{'x': -10.811492470858468, 'y': -23.587532561108222, 'vx': 0, 'vy': 0, 'team_id': 1},
{'x': -7.1571725484272966, 'y': -10.230408907707034, 'vx': 0, 'vy': 0, 'team_id': 1},
{'x': -2.196378942317624, 'y': -0.40206191775466354, 'vx': 0, 'vy': 0, 'team_id': 1},
{'x': -19.328544937208044, 'y': -6.802200007868615, 'vx': 0, 'vy': 0, 'team_id': -1},
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
                values[i] = values_0 - values[i]
            else:
                assert dict_[i]["team_id"] == 1
                values[i] = values_1 - values[i]

    print(values)

    # best_position = []
    # for i in range(len(values)):
    #     if i< len(dict_):
    #         if dict_[i]["team_id"] == -1:
    #             best_position.append([0, 0])
    #             continue
    #         best_value = -1e18
    #         flag = 1
    #         if dict_[i]["team_id"] == 1:
    #             flag = -1

    #         dx_ = None
    #         dy_ = None
    #         for d_x in range(-5, 6):
    #             for d_y in range(-5, 6):
    #                 dx =d_x
    #                 dy=d_y
    #                 temp_player_feature = player_feature.clone().detach()
    #                 temp_player_feature[i][0] += dx
    #                 temp_player_feature[i][1] += dy
    #                 temp_predict = model_see_all(temp_player_feature.unsqueeze(0).to(device), mask=mask.unsqueeze(0).to(device))[0].item()
    #                 if temp_predict * flag > best_value:
    #                     best_value = temp_predict * flag
    #                     dx_ = dx
    #                     dy_ = dy
    #         print(f"best_value: {best_value - flag*predict}, dx: {dx_}, dy: {dy_}")
    #         best_position.append([dx_, dy_])
    #         values[i] = best_value - flag * predict
    #         # values[i] = predict * flag - best_value
                    
    #     else:
    #         best_position.append([0, 0])
    # best_position = np.array(best_position)

    # print(values)


    img = get_pitch_from_pt(player_feature, values, best_position=None)

    # save the image to output_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    img.savefig(os.path.join(output_path, "2d.png"))