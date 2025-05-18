import cv2
import numpy as np
from sklearn.metrics import pairwise_distances
def remove_grass_background(image):
    """去除草地背景并返回掩码"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 定义绿色的范围
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    
    # 生成掩码，识别出草地部分
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # 反转掩码，保留非绿色区域
    mask_inv = cv2.bitwise_not(mask)
    
    # 应用掩码到原图
    filtered_image = cv2.bitwise_and(image, image, mask=mask_inv)
    
    return filtered_image, mask_inv

def extract_color_histogram(image, bins=(8, 8, 8)):
    """计算颜色直方图，排除被掩码区域；若全被掩盖，使用均匀直方图"""
    filtered_image, mask_inv = remove_grass_background(image)
    
    # 转换为 HSV 颜色空间
    hsv = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2HSV)

    # 计算带掩码的直方图
    hist = cv2.calcHist([hsv], [0, 1, 2], mask_inv, bins, [0, 180, 0, 256, 0, 256])
    
    # 如果整张图片被掩盖，返回均匀直方图
    if np.sum(mask_inv) == 0:
        print("全被掩盖，使用均匀直方图")
        # 均匀直方图：所有值相等，且总和为 1
        hist = np.ones(bins[0] * bins[1] * bins[2], dtype=np.float32)
        hist /= np.sum(hist)  # 归一化

    else:
        cv2.normalize(hist, hist)

    return hist.flatten()

team_db = {
    "left": [],
    "right": [],
}

# samples = [
#     ("left_player_1.jpg", [50, 100, 200, 300], "left"),
#     ("right_player_1.jpg", [30, 60, 180, 250], "right")
# ]


def process_db(samples):
    for img_path, bbox, team in samples:
        image = cv2.imread(img_path)
        x1, y1, x2, y2 = bbox
        player_crop = image[y1:y2, x1:x2]
        feature = extract_color_histogram(player_crop)
        team_db[team].append(feature)


def chi_square_distance(x, y, eps=1e-10):
    return 0.5 * np.sum(((x - y) ** 2) / (x + y + eps))

def reduce_distance(distances, k = 64):
    # 从小到大排序
    sorted_indices = np.argsort(distances)
    # 选择前 k 个最小距离
    selected_indices = sorted_indices[:k]
    # 计算平均距离
    mean_distance = np.mean(distances[selected_indices])
    return mean_distance

# 新图片推理
def predict_team(image, bbox):
    x1, y1, x2, y2 = bbox
    player_crop = image[y1:y2, x1:x2]
    feature = extract_color_histogram(player_crop)
    
    # 计算与数据库中每一类的距离
    distances_left = pairwise_distances([feature], team_db["left"], metric=chi_square_distance).flatten()
    distances_right = pairwise_distances([feature], team_db["right"], metric=chi_square_distance).flatten()


    distance_left = reduce_distance(distances_left)
    distance_right = reduce_distance(distances_right)

    
    # 选择最小距离的类别
    if len(distances_left) > 0 and len(distances_right) > 0:
        if distance_left < distance_right:
            return "left", distance_left, distance_right
        else:
            return "right", distance_left, distance_right
    return "unknown"


if __name__ == "__main__":
    samples = []
    name = "game_example"
    output_path = "classification_output"

    import os
    import json
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    video_path = f"{name}/1_720p.mkv"
    json_path = f"{name}/new_output.json"

    # get image
    cap = cv2.VideoCapture(video_path)
    frames = json.load(open(json_path, "r"))

    # for test_frame in range(1,300):

    #     frame_info = None

    #     for frame in frames:
    #         if frame["frame"] == test_frame:
    #             frame_info = frame
    #             break
    #         if frame["frame"] > test_frame:
    #             break
    #     if frame_info is None:
    #         continue
        
    #     cap.set(cv2.CAP_PROP_POS_FRAMES, test_frame)

    #     ret, image = cap.read()
    #     if not ret:
    #         print(f"Error: frame {test_frame} not found in {video_path}")
    #         exit(1)
        
    #     # save the image
    #     cv2.imwrite(f"{output_path}/frame_{test_frame}.jpg", image)

    #     n = len(frame_info["bboxes_ltwh"])

    #     for i in range(n):
    #         if frame_info["categories"][i] == "player" and frame_info["team_detections"][i] in ["left", "right"]:
    #             x, y, w, h = frame_info["bboxes_ltwh"][i]
    #             x2 = x + w
    #             y2 = y + h
    #             x = int(x)
    #             y = int(y)
    #             x2 = int(x2)
    #             y2 = int(y2)

    #             samples.append((f"{output_path}/frame_{test_frame}.jpg", [x, y, x2, y2], frame_info["team_detections"][i]))

    #             # player_crop = image[y:y2, x:x2]
    #             # cv2.imwrite(f"{output_path}/player_{frame_info['team_detections'][i]}_{i}.jpg", player_crop)

    #     process_db(samples)
    #     print("Team database: ", len(team_db["left"]), len(team_db["right"]), test_frame)

    # print("Team database processed.")

    # # save the db
    # team_db_serializable = {
    #     "left": [feature.tolist() for feature in team_db["left"]],
    #     "right": [feature.tolist() for feature in team_db["right"]]
    # }
    # with open(f"{output_path}/team_db.json", "w") as f:
    #     json.dump(team_db_serializable, f)

    # load the db
    with open(f"{output_path}/team_db.json", "r") as f:
        team_db_serializable = json.load(f)
        team_db["left"] = [np.array(feature) for feature in team_db_serializable["left"]]
        team_db["right"] = [np.array(feature) for feature in team_db_serializable["right"]]


    for check_frame in range(45525, 45525+30):
        frame_info = None

        for frame in frames:
            if frame["frame"] == check_frame:
                frame_info = frame
                break
            if frame["frame"] > check_frame:
                break
        if frame_info is None:
            continue
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, check_frame)

        ret, image = cap.read()
        if not ret:
            print(f"Error: frame {check_frame} not found in {video_path}")
            exit(1)
        
        n = len(frame_info["bboxes_ltwh"])
        for i in range(n):
            x, y, w, h = frame_info["bboxes_ltwh"][i]
            x2 = x + w
            y2 = y + h
            x = int(x)
            y = int(y)
            x2 = int(x2)
            y2 = int(y2)
            print(f"Old: detections: {i}-{check_frame}-{frame_info['categories'][i]}-{frame_info['team_detections'][i]}, New detections: {predict_team(image, [x, y, x2, y2])}")
            # player_crop = image[y:y2, x:x2]
            # cv2.imwrite(f"{output_path}/{i}.jpg", player_crop)
            # filtered_image = remove_grass_background(player_crop)
            # cv2.imwrite(f"{output_path}/filtered_{i}.jpg", filtered_image)

    # for i in range(n):
    #     if frame_info["categories"][i] == "player":
    #         x, y, w, h = frame_info["bboxes_ltwh"][i]
    #         x2 = x + w
    #         y2 = y + h
    #         x = int(x)
    #         y = int(y)
    #         x2 = int(x2)
    #         y2 = int(y2)

    #         print("Old team: ", frame_info["team_detections"][i])

    #         team = predict_team(image, [x, y, x2, y2])
    #         print(f"Predicted team for player {i}: {team}")