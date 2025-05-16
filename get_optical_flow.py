import cv2
import numpy as np

def get_optical_flow(previous_frame, current_frame, bbox, lines_points):
    '''
    bbox: [[x_1, y_1, w_1, h_1], [x_2, y_2, w_2, h_2], ...]
    '''
    # Convert frames to grayscale
    previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    # Calculate optical flow
    flow = cv2.calcOpticalFlowFarneback(previous_gray, current_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    # Calculate the average flow for each bounding box

    avg_flow_of_lines = []
    for point in lines_points:
        x, y = point
        roi = flow[y-5:y+5, x-5:x+5]
        avg_flow = np.mean(roi, axis=(0, 1))
        # print("avg_flow: ", avg_flow)
        avg_flow_of_lines.append(avg_flow)
    avg_flow_of_lines = np.mean(avg_flow_of_lines, axis=0)
    # print("avg_flow_of_lines: ", avg_flow_of_lines)

    flows = []
    for box in bbox:
        x, y, w, h = box
        roi = flow[y:y+h, x:x+w]
        avg_flow = np.mean(roi, axis=(0, 1))
        flows.append(avg_flow - avg_flow_of_lines)

    return flows

if __name__ == "__main__":
    frame_idx = 25*8
    video_path = "test_video/1_720p.mkv"
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, previous_frame = cap.read()
    if not ret:
        print("Error reading frame")
        exit(1)
    ret, current_frame = cap.read()
    if not ret:
        print("Error reading frame")
        exit(1)

    # visualize previous_frame and current_frame
    # save to "test_video/previous_frame.jpg" and "test_video/current_frame.jpg"

    detected = [{'x_image': 0.29791666666666666, 'y_image': 0.18888888888888888, 'x_world': -52.5, 'y_world': 34.0, 'z_world': 0.0}, {'x_image': 0.19583333333333333, 'y_image': 0.22592592592592592, 'x_world': -52.5, 'y_world': 20.16, 'z_world': 0.0}, {'x_image': 0.41041666666666665, 'y_image': 0.24074074074074073, 'x_world': -36.0, 'y_world': 20.16, 'z_world': 0.0}, {'x_image': 0.0875, 'y_image': 0.26296296296296295, 'x_world': -52.5, 'y_world': 9.16, 'z_world': 0.0}, {'x_image': 0.1625, 'y_image': 0.27037037037037037, 'x_world': -47.0, 'y_world': 9.16, 'z_world': 0.0}, {'x_image': 0.3104166666666667, 'y_image': 0.2962962962962963, 'x_world': -36.0, 'y_world': 7.32, 'z_world': 0.0}, {'x_image': 0.14791666666666667, 'y_image': 0.3814814814814815, 'x_world': -36.0, 'y_world': -7.310000000000002, 'z_world': 0.0}, {'x_image': 0.3145833333333333, 'y_image': 0.3333333333333333, 'x_world': -32.510000000000005, 'y_world': 1.7100000000000009, 'z_world': 0.0}, {'x_image': 0.7708333333333334, 'y_image': 0.3814814814814815, 'x_world': -8.82, 'y_world': 2.469999999999999, 'z_world': 0.0}, {'x_image': 0.7520833333333333, 'y_image': 0.4222222222222222, 'x_world': -8.82, 'y_world': -2.460000000000001, 'z_world': 0.0}, {'x_image': 0.14375, 'y_image': 0.32222222222222224, 'x_world': -41.5, 'y_world': -0.0, 'z_world': 0.0}, {'x_image': 0.23541666666666666, 'y_image': 0.337037037037037, 'x_world': -36.0, 'y_world': -0.0, 'z_world': 0.0}, {'x_image': 0.29791666666666666, 'y_image': 0.34444444444444444, 'x_world': -32.35, 'y_world': -0.0, 'z_world': 0.0}, {'x_image': 0.8375, 'y_image': 0.3592592592592593, 'x_world': -6.469999999999999, 'y_world': 6.469999999999999, 'z_world': 0.0}, {'x_image': 0.75625, 'y_image': 0.3962962962962963, 'x_world': -9.149999999999999, 'y_world': -0.0, 'z_world': 0.0}, {'x_image': 0.7979166666666667, 'y_image': 0.4703703703703704, 'x_world': -6.469999999999999, 'y_world': -6.469999999999999, 'z_world': 0.0}]
    vis_pts = [(0.3,0.3), (0.5,0.9), (0.9,0.5)]

    special_points = []
    for dic in detected:
        if dic['z_world'] != 0:
            continue
        x = dic['x_image']
        y = dic['y_image']
        special_points.append((int(x * previous_frame.shape[1]), int(y * previous_frame.shape[0])))

    cv2.imwrite("test_video/previous_frame.jpg", previous_frame)
    cv2.imwrite("test_video/current_frame.jpg", current_frame)

    w = previous_frame.shape[1]
    h = previous_frame.shape[0]

    bbox = [
        [0.39 ,0.42, 0.02, 0.1],
        [0.33 ,0.55, 0.03, 0.13],
        [0.11 ,0.7, 0.03, 0.14],
    ]

    for i in range(len(bbox)):
        bbox[i][0] = int(bbox[i][0] * w)
        bbox[i][1] = int(bbox[i][1] * h)
        bbox[i][2] = int(bbox[i][2] * w)
        bbox[i][3] = int(bbox[i][3] * h)

    # visualize bbox
    # for box in bbox:
    #     x, y, w, h = box
    #     cv2.rectangle(current_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # cv2.imwrite("test_video/current_frame_bbox.jpg", current_frame)


    flow = get_optical_flow(previous_frame, current_frame, bbox, special_points)

    print("Optical flow: ", flow)

    # visualize normalized flow


    for i, flow_ in enumerate(flow):

        x = flow_[0]
        y = flow_[1]
        middle_x = bbox[i][0] + bbox[i][2] // 2
        middle_y = bbox[i][1] + bbox[i][3] // 2


        x *= 10
        y *= 10

        cv2.arrowedLine(current_frame, (middle_x, middle_y), (middle_x + int(x), middle_y + int(y)), (0, 255, 0), 2)

    cv2.imwrite("test_video/current_frame_flow.jpg", current_frame)