import cv2

from get_annotation import load_models, run_demo
from process_dataset import process_anno
from tqdm import tqdm

model, model_l = load_models()


def process(frames, fps):
    images = [image for image, _ in frames]
    annotations = run_demo(images, model, model_l)
    results = {}
    for i, (image, frame_idx) in enumerate(frames):
        annotation = annotations[i]
        # print(frame_idx, len(annotation))
        if annotation is None:
            continue
        matrix, error = process_anno(annotation)
        t = int(frame_idx / fps)
        if matrix is not None:
            key = f"{t}({int(t // 60)}:{int(t % 60):02d})"
            result = {
                "frame": frame_idx,
                "matrix": (matrix.tolist() if matrix is not None else None),
                "error": error,
                "annotation": annotation,
            }
            if key not in results:
                results[key] = []
            results[key].append(result)
    return results

def merge_dict(dst, src):
    for key, value in src.items():
        if key not in dst:
            dst[key] = []
        dst[key].extend(value)
    return dst

def process_time_segment(video_path, start_time, end_time, num_frame_each_second = 5):

    batch_size = 10


    """
    Process the video segment between start_time and end_time.
    """
    # Load the video
    video = cv2.VideoCapture(video_path)
    
    # Get the frames per second (fps) of the video
    fps = video.get(cv2.CAP_PROP_FPS)
    
    # Calculate the start and end frame numbers
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    
    # Set the video to the start frame
    video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frames = []
    results = {}
    
    # Read frames until we reach the end frame

    bar = tqdm(total=end_frame - start_frame, desc="Processing frames", unit="frame")


    while True:
        ret, frame = video.read()
        # to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        current_frame_number = int(video.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        if not ret or current_frame_number > end_frame:
            break
        t = int(current_frame_number / fps)
        key = f"{t}({int(t // 60)}:{int(t % 60):02d})"
        if key in results and len(results[key]) >= num_frame_each_second:
            bar.update(1)
            continue

        frames.append((frame, current_frame_number))

        if len(frames) == batch_size:
            batch_results = process(frames, fps)
            results = merge_dict(results, batch_results)
            frames = []

        bar.update(1)

    if len(frames) > 0:
        batch_results = process(frames, fps)
        results = merge_dict(results, batch_results)
    video.release()
    bar.close()

    new_results = {}

    for key, value in results.items():
        if len(value) > num_frame_each_second:
            value.sort(key=lambda x: x["frame"])
            new_results[key] = value[:num_frame_each_second]
        else:
            new_results[key] = value

    # for key, value in new_results.items():
    #     for i in range(len(value)):
    #         print(f"key: {key}, len: {len(value)}, {value[i]['frame'], len(value[i]['annotation'])}")

    return new_results
    
if __name__ == "__main__":
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

    bar = tqdm(total=total_time, desc="Processing video segments", unit="s")

    processed_results = {}

    for segment in compute_segments:
        video_path = first_half_path if segment[0][0] == "1" else second_half_path
        
        start_str = segment[0].split(" - ")[1]
        end_str   = segment[1].split(" - ")[1]
        start_seconds = int(start_str.split(":")[0]) * 60 + int(start_str.split(":")[1])
        end_seconds   = int(end_str.split(":")[0])   * 60 + int(end_str.split(":")[1])

        processed_results[f"{segment}"] = process_time_segment(video_path, start_seconds, end_seconds)

        with open(output_path, "w") as f:
            json.dump(processed_results, f, indent=4)

        bar.update(end_seconds - start_seconds)
    bar.close()