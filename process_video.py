video_path = "test_video"

first_half_path = f"{video_path}/1_720p.mkv"
second_half_path = f"{video_path}/2_720p.mkv"
label_camera_path = f"{video_path}/Labels-cameras.json"

output_path = f"{video_path}/output.json"

compute_segments = []

import json

label_camera_json = json.load(open(label_camera_path, "r"))

for index, label in enumerate(label_camera_json["annotations"]):
    if label["replay"] == "real-time" and label["label"].startswith("Main"):
        pre_time = None
        if index > 0:
            pre_time = label_camera_json["annotations"][index - 1]["gameTime"]
        else:
            pre_time = "1 - 00:00"
        now_time = label_camera_json["annotations"][index]["gameTime"]
        if pre_time[0] != now_time[0]:
            pre_time = "2 - 00:00"
            assert pre_time[0] == now_time[0], f"pre_time: {pre_time}, now_time: {now_time}"
        compute_segments.append((pre_time, now_time))

total_time = 0
for segment in compute_segments:
    start_time = segment[0].split(" - ")[1]
    end_time = segment[1].split(" - ")[1]
    start_time = int(start_time.split(":")[0]) * 60 + int(start_time.split(":")[1])
    end_time = int(end_time.split(":")[0]) * 60 + int(end_time.split(":")[1])
    total_time += end_time - start_time

print(f"Total time: {total_time} seconds")


import cv2
import cv2

from get_annotation import run_demo
from process_dataset import process_anno

def process(frames):
    images = [image for image, _ in frames]
    annotations = run_demo(images)
    results = []
    for i, (image, t) in enumerate(frames):
        annotation = annotations[i]
        if annotation is None:
            continue
        matrix, error = process_anno(annotation)
        result = {
            "time": t,
            "htime": f"{int(t // 60)}:{int(t % 60):02d}",
            "annotation": annotation,
            "matrix": (matrix.tolist() if matrix is not None else None),
            "error": error,
        }
        results.append(result)
    return results


def get_sampled_frames_from_segment(video_path, count, segment, sample_interval=1):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_str = segment[0].split(" - ")[1]
    end_str   = segment[1].split(" - ")[1]
    
    start_seconds = int(start_str.split(":")[0]) * 60 + int(start_str.split(":")[1])
    end_seconds   = int(end_str.split(":")[0])   * 60 + int(end_str.split(":")[1])
    
    t = start_seconds
    t = max(1, t)

    results = []

    frames = []

    BATCH_SIZE = 8

    frame_idx = int(t * fps - 1) # -1 to get the previous frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    prev_frame = None
    
    while t < end_seconds:

        while frame_idx < int(t * fps):
            ret, frame = cap.read()
            frame_idx += 1
            prev_frame = frame
            if not ret:
                break

        ret, frame = cap.read()
        frame_idx += 1
        if not ret:
            break

        # print(t, start_seconds, end_seconds, frame_idx, fps)
        
        frames.append((frame, t))
        count += 1
        frame_filename = f"{video_path}/main_camera/Image/img1/{count:06d}.jpg"
        cv2.imwrite(frame_filename, frame)

        if len(frames) == BATCH_SIZE:
            results.extend(process(frames))
            frames = []

        t += sample_interval

    if len(frames) > 0:
        results.extend(process(frames))

    cap.release()
    return results, count


picture_detail = {}
picture_detail["matrix"] = []


from tqdm import tqdm
import subprocess 
import os
image_dir = f"{video_path}/main_camera/Image/img1"
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

count = 0

bar = tqdm(total=total_time, desc="Processing video segments", unit="s")

for segment in compute_segments:
    video_path = first_half_path if segment[0][0] == "1" else second_half_path
    results, new_count = get_sampled_frames_from_segment(video_path, count, segment)
    count = new_count
    print(f"Running tracklab.main for segment: {segment}")
    subprocess.run([
        'python', '-m', 'tracklab.main', '-cn', 'soccernet',
        f'data_dir={video_path}', 'dataset.eval_set=main_camera'
    ])
    for result in results:
        picture_detail["matrix"].append(result)

    with open(output_path, "w") as f:
        json.dump(picture_detail, f, indent=4)

    start_str = segment[0].split(" - ")[1]
    end_str   = segment[1].split(" - ")[1]
    
    start_seconds = int(start_str.split(":")[0]) * 60 + int(start_str.split(":")[1])
    end_seconds   = int(end_str.split(":")[0])   * 60 + int(end_str.split(":")[1])

    bar.update(end_seconds - start_seconds)
bar.close()