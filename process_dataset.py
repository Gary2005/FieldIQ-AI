train_len = 16463
test_len = 3141
valid_len = 3212

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import json
from getH import calc_H, project_point
import numpy

def visualize_annotations(image_path, json_path, output_path):
    # 打开图片并获取尺寸
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    width, height = image.size

    # 读取JSON标注数据
    with open(json_path, 'r') as f:
        annotations = json.load(f)

    # 预定义颜色列表和字体
    colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'cyan', 'magenta']
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    # 遍历每个标注项
    for idx, (label, points) in enumerate(annotations.items()):
        # 转换归一化坐标为实际像素坐标
        for i in range(len(points) - 1):
            p1 = (points[i]['x'] * width, points[i]['y'] * height)
            p2 = (points[i+1]['x'] * width, points[i+1]['y'] * height)
            color = colors[idx % len(colors)]

            draw.line([p1, p2], fill=color, width=3)

            # 计算标签位置（线段/矩形中心）
            mid_x = (points[i]['x'] + points[i]['x']) * width / 2
            mid_y = (points[i+1]['y'] + points[i+1]['y']) * height / 2

            # 添加文字标签（带背景框）
            text_bbox = draw.textbbox((mid_x, mid_y), label, font=font)
            draw.rectangle(text_bbox, fill="white")
            draw.text((mid_x, mid_y), label, fill=color, font=font)

    # 保存结果
    image.save(output_path)
    print(f"可视化结果已保存至 {output_path}")

def get_all_label():

    keys = set()

    for id in range(train_len):
        json_id = str(id).zfill(5)
        # print(json_id)
        json_path = f"calibration-2023/train/{json_id}.json"

        with open(json_path, 'r') as f:
            annotations = json.load(f)

        for label in annotations.keys():
            keys.add(label)
    # 排序
    keys = sorted(keys)

    print("所有标签：")
    for key in keys:
        print(key)

real_position = {
    "Big rect. left bottom":{
        "x": None,
        "y": -20.16,
        "z": 0.0
    },
    "Big rect. left main":{
        "x": -36,
        "y": None,
        "z": 0.0
    },
    "Big rect. left top":{
        "x": None,
        "y": 20.16,
        "z": 0.0
    },
    "Big rect. right bottom":{
        "x": None,
        "y": -20.16,
        "z": 0.0
    },
    "Big rect. right main":{
        "x": 36,
        "y": None,
        "z": 0.0
    },
    "Big rect. right top":{
        "x": None,
        "y": 20.16,
        "z": 0.0
    },
    "Circle central":{
        "r": 9.15,
        "x": 0.0,
        "y": 0.0,
    },
    "Circle left":{
        "r": 9.15,
        "x": -41.5,
        "y": 0.0,
    },
    "Circle right":{
        "r": 9.15,
        "x": 41.5,
        "y": 0.0,
    },
    "Goal left crossbar":{
        "x": -52.5,
        "y": None,
        "z": 2.44,
    },
    "Goal left post left":{
        "x": -52.5,
        "y": -3.66,
        "z": None,
    },
    "Goal left post right":{
        "x": -52.5,
        "y": 3.66,
        "z": None,
    },
    "Goal right crossbar":{
        "x": 52.5,
        "y": None,
        "z": 2.44,
    },
    "Goal right post left":{
        "x": 52.5,
        "y": 3.66,
        "z": None,   
    },
    "Goal right post right":{
        "x": 52.5,
        "y": -3.66,
        "z": None,
    },
    "Middle line":{
        "x": 0.0,
        "y": None,
        "z": 0.0
    },
    "Side line bottom":{
        "x": None,
        "y": -34,
        "z": 0.0
    },
    "Side line left":{
        "x": -52.5,
        "y": None,
        "z": 0.0
    },
    "Side line right":{
        "x": 52.5,
        "y": None,
        "z": 0.0
    },
    "Side line top":{
        "x": None,
        "y": 34,
        "z": 0.0
    },
    "Small rect. left bottom":{
        "x": None,
        "y": -9.16,
        "z": 0.0
    },
    "Small rect. left main":{
        "x": -47,
        "y": None,
        "z": 0.0
    },
    "Small rect. left top":{
        "x": None,
        "y": 9.16,
        "z": 0.0
    },
    "Small rect. right bottom":{
        "x": None,
        "y": -9.16,
        "z": 0.0
    },
    "Small rect. right main":{
        "x": 47,
        "y": None,
        "z": 0.0
    },
    "Small rect. right top":{
        "x": None,
        "y": 9.16,
        "z": 0.0
    },
}

import os

def draw_pitch(real_x, real_y):

    # 场地尺寸（米）
    field_length = 105
    field_width = 68

    # 中圈
    center_circle_radius = 9.15

    # 禁区尺寸
    penalty_area_width = 40.32
    penalty_area_length = 16.5

    # 小禁区尺寸
    goal_area_width = 18.32
    goal_area_length = 5.5

    # 罚球点距离球门
    penalty_spot_distance = 11

    # 罚球弧半径
    penalty_arc_radius = 9.15

    # 球门尺寸
    goal_width = 7.32
    goal_post_depth = 0.12  # 12厘米

    # 角球弧半径
    corner_arc_radius = 1

    # 创建画布
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(-field_length/2 - 5, field_length/2 + 5)
    ax.set_ylim(-field_width/2 - 5, field_width/2 + 5)
    ax.set_aspect('equal')
    ax.axis('off')

    # --- 画场地外框 ---
    outer_rect = patches.Rectangle((-field_length/2, -field_width/2),
                                    field_length, field_width,
                                    fill=False, color='black', lw=2)
    ax.add_patch(outer_rect)

    # --- 中线 ---
    ax.plot([0, 0], [-field_width/2, field_width/2], color='black', lw=2)

    # --- 中圈 ---
    center_circle = patches.Circle((0, 0), center_circle_radius,
                                    fill=False, color='black', lw=2)
    ax.add_patch(center_circle)

    # --- 中点 ---
    ax.plot(0, 0, 'ko', markersize=2)

    # --- 左右两端的罚球区、小禁区、球门、罚球弧 ---
    for side in [-1, 1]:  # -1代表左边，1代表右边
        # 罚球区
        penalty_area = patches.Rectangle(
            (side * field_length/2 - side * penalty_area_length, -penalty_area_width/2),
            penalty_area_length * side, penalty_area_width,
            fill=False, color='black', lw=2)
        ax.add_patch(penalty_area)

        # 小禁区
        goal_area = patches.Rectangle(
            (side * field_length/2 - side * goal_area_length, -goal_area_width/2),
            goal_area_length * side, goal_area_width,
            fill=False, color='black', lw=2)
        ax.add_patch(goal_area)

        # 罚球点
        penalty_spot = (side * field_length/2 - side * penalty_spot_distance, 0)
        ax.plot(*penalty_spot, 'ko', markersize=2)

        # 罚球弧
        arc = patches.Arc(
            penalty_spot, width=2*penalty_arc_radius, height=2*penalty_arc_radius,
            angle=0,
            theta1=308 if side == -1 else 128,  # 角度控制弧的起止范围，右门左门不同
            theta2=52 if side == -1 else 232,
            color='black', lw=2)
        ax.add_patch(arc)

        # 球门
        goal = patches.Rectangle(
            (side * field_length/2, -goal_width/2),
            side * goal_post_depth, goal_width,
            fill=False, color='black', lw=2)
        ax.add_patch(goal)


    # 画real_x, real_y
    # 画一个小圆圈表示位置
    ax.plot(real_x, real_y, 'ro', markersize=5)


    # save the figure
    plt.savefig('soccer_pitch.png', dpi=300, bbox_inches='tight')


def process(split, id):
    # if exists f"calibration-2023/{split}/{str(id).zfill(5)}_H.json", return 1
    # if os.path.exists(f"calibration-2023/{split}/{str(id).zfill(5)}_H.json"):
    #     return 1

    path = f"calibration-2023/{split}"
    input_json = f"{path}/{str(id).zfill(5)}.json"
    with open(input_json, 'r') as f:
        annotations = json.load(f)
    data = []
    # check_data = []
    for (label, points) in annotations.items():
        if label in real_position:
            if ("z" not in real_position[label].keys()) or (real_position[label]["z"] is not None and real_position[label]["z"] == 0.0):
                if "r" in real_position[label].keys():
                    for point in points:
                        data.append({
                            "x_img": point["x"],
                            "y_img": point["y"],
                            "circle": {
                                "a": real_position[label]["x"],
                                "b": real_position[label]["y"],
                                "R": real_position[label]["r"]
                            }
                        })
                        # data.append(check_data[-1])

                        # pass
                else:
                    for point in points:
                        if real_position[label]["x"] is not None:
                            data.append({
                                "x_img": point["x"],
                                "y_img": point["y"],
                                "x_real": real_position[label]["x"]
                            })
                        if real_position[label]["y"] is not None:
                            data.append({
                                "x_img": point["x"],
                                "y_img": point["y"],
                                "y_real": real_position[label]["y"]
                            })

    error = 1e9
    H = None

    for try_ in range(2):
        H, error = calc_H(data, np.random.rand(9))
        # print(error)
        if H is None:
            break
        if error < 3:
            break
        H = None


    # check_data
    if H is None:
        return 0
    # for data in check_data:
    #     x_img = data["x_img"]
    #     y_img = data["y_img"]
    #     circle = data["circle"]
    #     a = circle["a"]
    #     b = circle["b"]
    #     R = circle["R"]
    #     # 计算投影点
    #     proj = project_point(H, (x_img, y_img))
    #     # 计算距离
    #     distance = np.sqrt((proj[0] - a) ** 2 + (proj[1] - b) ** 2)
    #     # print(abs(distance - R))
    #     if abs(distance - R) > 2:
    #         return 0
    if H is not None:
        # write to path/{str(id).zfill(5)}_H.json
        output_json = f"{path}/{str(id).zfill(5)}_H.json"
        with open(output_json, 'w') as f:
            json.dump(H.tolist(), f)
        return 1
    return 0

def process_anno(annotations):
    data = []
    for dic in annotations:
        if dic["z_world"] == 0.0:
            data.append(dic)
    error = 1e9
    H = None

    for try_ in range(2):
        H, error = calc_H(data, np.random.rand(9))
        if H is None:
            break
        if error < 0.5:
            break
        H = None

    return H, error


def test(split, id, input_x, input_y, output_path = "./"):
    path = f"calibration-2023/{split}"
    input_h = f"{path}/{str(id).zfill(5)}_H.json"
    input_img = f"{path}/{str(id).zfill(5)}.jpg"
    with open(input_h, 'r') as f:
        H = json.load(f)
    H = numpy.array(H)
    real_x, real_y = project_point(H, (input_x, input_y))

    # draw (input_x, input_y) in the original image
    image = Image.open(input_img)
    draw = ImageDraw.Draw(image)
    img_x = int(input_x * image.size[0])
    img_y = int(input_y * image.size[1])
    draw.ellipse((img_x - 5, img_y - 5, img_x + 5, img_y + 5), fill="red", outline="red")
    draw.text((img_x + 10, img_y), f"({input_x:.2f}, {input_y:.2f})", fill="red")
    image.save(os.path.join(output_path, f"vis.jpg"))
    print(real_x, real_y)
    draw_pitch(real_x, real_y)

def clear():
    # delete all json files in calibration-2023/train
    path = f"calibration-2023/train"
    for file in os.listdir(path):
        if file.endswith("_H.json"):
            os.remove(os.path.join(path, file))
            print(f"Deleted {file}")

    # delete all json files in calibration-2023/test
    path = f"calibration-2023/test"
    for file in os.listdir(path):
        if file.endswith("_H.json"):
            os.remove(os.path.join(path, file))
            print(f"Deleted {file}")
    # delete all json files in calibration-2023/valid
    path = f"calibration-2023/valid"
    for file in os.listdir(path):
        if file.endswith("_H.json"):
            os.remove(os.path.join(path, file))
            print(f"Deleted {file}")

def check_the_shape_of_image():
    shape = set()

    path = f"calibration-2023/train"
    for file in os.listdir(path):
        if file.endswith(".jpg"):
            image = Image.open(os.path.join(path, file))
            shape.add(image.size)

    path = f"calibration-2023/test"
    for file in os.listdir(path):
        if file.endswith(".jpg"):
            image = Image.open(os.path.join(path, file))
            shape.add(image.size)

    path = f"calibration-2023/valid"
    for file in os.listdir(path):
        if file.endswith(".jpg"):
            image = Image.open(os.path.join(path, file))
            shape.add(image.size)

    print(shape)
    
def test(path, annotations, vis_pts):
    H, error = process_anno(annotations)
    print(H, error)
    
    # visualize the vis_pts
    image = Image.open(path)
    draw = ImageDraw.Draw(image)
    for pt in vis_pts:
        x, y = pt
        img_x = int(x * image.size[0])
        img_y = int(y * image.size[1])
        world_x, world_y = project_point(H, (x, y))
        draw.ellipse((img_x - 5, img_y - 5, img_x + 5, img_y + 5), fill="red", outline="red")
        draw.text((img_x + 10, img_y), f"({x:.2f}, {y:.2f})", fill="red")
        draw.text((img_x + 10, img_y + 20), f"({world_x:.2f}, {world_y:.2f})", fill="red")
        print(world_x, world_y)
    image.save("vis.jpg")

if __name__ == "__main__":

    # check_the_shape_of_image()

    # print(process("test", 3067))

    # test("test", 3067, 0.3, 0.7, "./")
    
    path = "game_example/output/0:04_first.jpg"
    annotations = [{'x_image': 0.29791666666666666, 'y_image': 0.18888888888888888, 'x_world': -52.5, 'y_world': 34.0, 'z_world': 0.0}, {'x_image': 0.19583333333333333, 'y_image': 0.22592592592592592, 'x_world': -52.5, 'y_world': 20.16, 'z_world': 0.0}, {'x_image': 0.41041666666666665, 'y_image': 0.24074074074074073, 'x_world': -36.0, 'y_world': 20.16, 'z_world': 0.0}, {'x_image': 0.0875, 'y_image': 0.26296296296296295, 'x_world': -52.5, 'y_world': 9.16, 'z_world': 0.0}, {'x_image': 0.1625, 'y_image': 0.27037037037037037, 'x_world': -47.0, 'y_world': 9.16, 'z_world': 0.0}, {'x_image': 0.3104166666666667, 'y_image': 0.2962962962962963, 'x_world': -36.0, 'y_world': 7.32, 'z_world': 0.0}, {'x_image': 0.14791666666666667, 'y_image': 0.3814814814814815, 'x_world': -36.0, 'y_world': -7.310000000000002, 'z_world': 0.0}, {'x_image': 0.3145833333333333, 'y_image': 0.3333333333333333, 'x_world': -32.510000000000005, 'y_world': 1.7100000000000009, 'z_world': 0.0}, {'x_image': 0.7708333333333334, 'y_image': 0.3814814814814815, 'x_world': -8.82, 'y_world': 2.469999999999999, 'z_world': 0.0}, {'x_image': 0.7520833333333333, 'y_image': 0.4222222222222222, 'x_world': -8.82, 'y_world': -2.460000000000001, 'z_world': 0.0}, {'x_image': 0.14375, 'y_image': 0.32222222222222224, 'x_world': -41.5, 'y_world': -0.0, 'z_world': 0.0}, {'x_image': 0.23541666666666666, 'y_image': 0.337037037037037, 'x_world': -36.0, 'y_world': -0.0, 'z_world': 0.0}, {'x_image': 0.29791666666666666, 'y_image': 0.34444444444444444, 'x_world': -32.35, 'y_world': -0.0, 'z_world': 0.0}, {'x_image': 0.8375, 'y_image': 0.3592592592592593, 'x_world': -6.469999999999999, 'y_world': 6.469999999999999, 'z_world': 0.0}, {'x_image': 0.75625, 'y_image': 0.3962962962962963, 'x_world': -9.149999999999999, 'y_world': -0.0, 'z_world': 0.0}, {'x_image': 0.7979166666666667, 'y_image': 0.4703703703703704, 'x_world': -6.469999999999999, 'y_world': -6.469999999999999, 'z_world': 0.0}]
    vis_pts = [(0.3,0.3), (0.5,0.9), (0.9,0.5)]
    test(path, annotations, vis_pts)

    # input_id = "03067"
    # image_path = f"calibration-2023/test/{input_id}.jpg"
    # json_path = f"calibration-2023/test/{input_id}.json"
    # output_path = "visualization2.jpg"
    # visualize_annotations(image_path, json_path, output_path)

    # get_all_label()


    RUN_ALL = False
    if RUN_ALL:
        clear()
        from tqdm import tqdm

        # # # for i in tqdm(range(train_len)):
        # # #     process("train", i)
        # # # for i in range(test_len):
        # # #     process("test", i)
        # # # for i in range(valid_len):
        # # #     process("valid", i)

        # # 多线程
        from concurrent.futures import ThreadPoolExecutor
        def process_wrapper_train(id):
            result = process("train", id)
            return result
        with ThreadPoolExecutor(max_workers=16) as executor:
            results = list(tqdm(executor.map(process_wrapper_train, range(train_len)), total=train_len))
        # results = numpy.zeros(train_len)
        # for i in tqdm(range(train_len)):
        #     result = process("train", i)
        #     results[i] = result
        print(f"train done, {sum(results)}/ {len(results)}")
        def process_wrapper_test(id):
            result = process("test", id)
            return result
        with ThreadPoolExecutor(max_workers=16) as executor:
            results = list(tqdm(executor.map(process_wrapper_test, range(test_len)), total=test_len))
        # results = numpy.zeros(test_len)
        # for i in tqdm(range(test_len)):
        #     result = process("test", i)
        #     results[i] = result
        print(f"test done, {sum(results)}/ {len(results)}")
        def process_wrapper_valid(id):
            result = process("valid", id)
            return result
        with ThreadPoolExecutor(max_workers=16) as executor:
            results = list(tqdm(executor.map(process_wrapper_valid, range(valid_len)), total=valid_len))
        # results = numpy.zeros(valid_len)
        # for i in tqdm(range(valid_len)):
        #     result = process("valid", i)
        #     results[i] = result
        print(f"valid done, {sum(results)}/ {len(results)}")
