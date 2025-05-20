path = "2015-09-26 - 17-00 Liverpool 3 - 2 Aston Villa"
label_path = f"{path}/Labels-v2.json"
frame_informati_path = f"{path}/output.json"
first_half_video_path = f"{path}/1_720p.mkv"
second_half_video_path = f"{path}/2_720p.mkv"


import json
import os
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, List, Tuple
import cv2
import numpy as np


label_json = json.load(open(label_path, "r"))
frame_information_json = json.load(open(frame_informati_path, "r"))

from getH import project_point
from pitch import get_pitch

def get_position_and_direction(bbox, flow, H, width, height, fps = 25, constant=1):
    fx,fy = flow

    x,y,w,h = bbox
    x_img,y_img = x + w/2, y + h # this is the center of the bottom of the bbox
    x_img = x_img/width
    y_img = y_img/height

    fx = fx/width
    fy = fy/height

    x_real,y_real = project_point(H, (x_img, y_img))
    # print(f"{(x_img, y_img)} -> {(x_real, y_real)}")
    # print(project_point(H, (0.6458333333333334, 0.37407407407407406)))
    x_real_nx, y_real_nx = project_point(H, (x_img + fx, y_img + fy))

    fx = x_real_nx - x_real
    fy = y_real_nx - y_real
    fx *= constant * fps
    fy *= constant * fps
    return (x_real, y_real), (fx, fy)


def process_frame_json():
    new_frame_json = []
    pre_t = -1
    half = "first"
    for element in frame_information_json["matrix"]:
        if element["frame"] < pre_t:
            assert half == "first", f"Error: {element['frame']} < {pre_t} in {half} half"
            half = "second"
        pre_t = element["frame"]
        new_element = element.copy()
        new_element["half"] = half

        # 检查optical_flows是否有NaN
        has_nan = False
        if new_element["optical_flows"] is None:
            has_nan = True
        else:
            for i in range(len(new_element["optical_flows"])):
                if np.isnan(new_element["optical_flows"][i][0]) or np.isnan(new_element["optical_flows"][i][1]):
                    has_nan = True
                    break
        if has_nan:
            continue
        
        if new_element["error"] > 0.5:
            continue

        if new_element["matrix"] is not None:
            new_frame_json.append(new_element)

    return new_frame_json

cap_first = cv2.VideoCapture(first_half_video_path)
cap_second = cv2.VideoCapture(second_half_video_path)

def visulize_one_frame(frame_idx, half, frame_info):
    info = None
    for frame in frame_info:
        # print(frame)
        if frame["frame"] == frame_idx and frame["half"] == half:
            info = frame
            break
    if info is None:
        print(f"Error: frame {frame_idx} not found in {half} half")
        return
    
    cap = cap_first if half == "first" else cap_second

    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_number = frame_idx

    # print(info["time"], frame_number)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    if not ret:
        return
    
    # draw the bboxes and optical flow
    # for i in range(len(info["bboxes_ltwh"])):
    #     x,y,w,h = info["bboxes_ltwh"][i]
    #     categories = info["categories"][i]
    #     role_detections = info["role_detections"][i]
    #     team = info["team_detections"][i]
    #     flow_x, flow_y = info["optical_flows"][i]
    #     flow_x *= 50
    #     flow_y *= 50
    #     cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)
    #     cv2.putText(frame, f"{categories} {role_detections} {team}", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    #     cv2.arrowedLine(frame, (int(x+w/2), int(y+h/2)), (int(x+w/2+flow_x), int(y+h/2+flow_y)), (255, 0, 0), 1)
        
    # save to f"{path}/output/{ht}_{half}.jpg"
    # if not os.path.exists(f"{path}/output/{frame_idx}-{half}"):
    #     os.makedirs(f"{path}/output/{frame_idx}-{half}")
    # cv2.imwrite(f"{path}/output/{frame_idx}-{half}/3d.jpg", frame)
    
    H = info["matrix"]
    H = np.array(H)

    positions = []
    directions = []
    for i in range(len(info["bboxes_ltwh"])):
        x,y,w,h = info["bboxes_ltwh"][i]
        flow_x, flow_y = info["optical_flows"][i]
        pos, dir = get_position_and_direction((x,y,w,h), (flow_x, flow_y), H, frame.shape[1], frame.shape[0])
        positions.append(pos)
        directions.append(dir)

    positions_dict = {}
    directions_dict = {}
    for i in range(len(info["bboxes_ltwh"])):
        x,y,w,h = info["bboxes_ltwh"][i]
        categories = info["categories"][i]
        # role_detections = info["role_detections"][i]
        # team = info["team_detections"][i]
        pos = positions[i]
        dir = directions[i]
        positions_dict[f"{categories}-{i}"] = pos
        directions_dict[f"{categories}-{i}"] = dir
    
    return {"time": info["time"], "half": half, "frame": frame_idx, "positions": positions_dict, "directions": directions_dict, "bbox_confs": info["bbox_confs"]}


if __name__ == "__main__":
    new_frame = process_frame_json()

    print("Total frames: ", len(new_frame))


    # save to f"{path}/new_output.json" 
    with open(f"{path}/new_output.json", "w") as f:
        json.dump(new_frame, f, indent=4)

    all_time_half = []
    for frame in new_frame:
        all_time_half.append((frame["frame"], frame["half"]))
        

    results = []

    for ht, half in tqdm(all_time_half):
        results.append(visulize_one_frame(ht, half, new_frame))

    # save to f"{path}/data.json"
    with open(f"{path}/data.json", "w") as f:
        json.dump(results, f, indent=4)