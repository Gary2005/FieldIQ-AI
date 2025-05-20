import cv2
import numpy as np
from getH import calc_H, project_point

def solve_H(lines_points, velocity, shape1, shape0):
    data = []
    for i, point in enumerate(lines_points):
        data.append({
            'x_image': point[0]/shape1,
            'y_image': point[1]/shape0,
            'x_world': velocity[i][0],
            'y_world': velocity[i][1]
        })
    for try_ in range(2):
        H, error = calc_H(data, np.random.rand(9))
        if H is None:
            break
        if error < 1:
            break
        H = None
    return H

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

    shape0, shape1 = previous_frame.shape[:2]

    flow_of_lines = []
    for point in lines_points:
        x, y = point
        x = int(x)
        y = int(y)
        roi = flow[y-5:y+5, x-5:x+5]
        avg_flow = np.mean(roi, axis=(0, 1))
        print("avg_flow: ", avg_flow)
        print(point, avg_flow)
        flow_of_lines.append(avg_flow)
    H = solve_H(lines_points, flow_of_lines, shape1, shape0)
    if H is None:
        return None

    flows = []
    print("H: ", H)
    print(project_point(H, (0.8395833333333333, 0.22592592592592592)))
    for box in bbox:
        x, y, w, h = box
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)
        roi = flow[y:y+h, x:x+w]
        avg_flow = np.mean(roi, axis=(0, 1))
        print(avg_flow, project_point(H, ((x + w // 2)/shape1, (y + h // 2)/shape0)), ((x + w // 2)/shape1, (y + h // 2)/shape0))
        flows.append(avg_flow - project_point(H, ((x + w // 2)/shape1, (y + h // 2)/shape0)))

    return flows

if __name__ == "__main__":
    frame_idx = 29003 - 1
    video_path = "2015-02-21 - 18-00 Swansea 2 - 1 Manchester United/1_720p.mkv"
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, previous_frame = cap.read()
    ret, current_frame = cap.read()
    
    # visualize previous_frame and current_frame
    # save to "test_video/previous_frame.jpg" and "test_video/current_frame.jpg"

    from get_annotation import load_models
    from get_annotation import run_demo
    model, model_l = load_models()
    annotations = run_demo([current_frame], model, model_l)[0]

    special_points = []
    for element in annotations:
        special_points.append((element['x_image'] * current_frame.shape[1], element['y_image'] * current_frame.shape[0]))


    cv2.imwrite("test_video/previous_frame.jpg", previous_frame)
    cv2.imwrite("test_video/current_frame.jpg", current_frame)

    w = previous_frame.shape[1]
    h = previous_frame.shape[0]

    bbox = [
                [
                    88.88481140136719,
                    325.72845458984375,
                    33.8018798828125,
                    76.56097412109375
                ],
                [
                    311.1767578125,
                    475.4122314453125,
                    42.875244140625,
                    98.8050537109375
                ],
                [
                    476.85906982421875,
                    341.86688232421875,
                    48.4859619140625,
                    80.298828125
                ],
                [
                    459.956298828125,
                    240.452880859375,
                    31.91925048828125,
                    57.975830078125
                ],
                [
                    547.2772216796875,
                    364.2423095703125,
                    42.60595703125,
                    83.26214599609375
                ],
                [
                    818.30859375,
                    201.83514404296875,
                    30.911376953125,
                    53.24871826171875
                ],
                [
                    1116.59912109375,
                    392.80596923828125,
                    60.04931640625,
                    78.0367431640625
                ],
                [
                    1138.205322265625,
                    210.09432983398438,
                    23.543212890625,
                    65.68438720703125
                ],
                [
                    116.3570785522461,
                    201.34698486328125,
                    27.26819610595703,
                    53.96807861328125
                ],
                [
                    238.66563415527344,
                    517.6160278320312,
                    67.37342834472656,
                    98.0269775390625
                ],
                [
                    248.91848754882812,
                    197.51364135742188,
                    19.34613037109375,
                    51.829132080078125
                ],
                [
                    614.188720703125,
                    151.8590850830078,
                    27.522216796875,
                    43.963623046875
                ],
                [
                    996.7164306640625,
                    461.94769287109375,
                    12.4677734375,
                    14.599853515625
                ],
                [
                    0.0,
                    227.67489624023438,
                    25.997562408447266,
                    65.462890625
                ]
            ]


    # visualize bbox
    for box in bbox:
        x = int(box[0])
        y = int(box[1])
        w = int(box[2])
        h = int(box[3])
        cv2.rectangle(current_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imwrite("test_video/current_frame_bbox.jpg", current_frame)


    flow = get_optical_flow(previous_frame, current_frame, bbox, special_points)

    print("Optical flow: ", flow)

    # visualize normalized flow


    for i, flow_ in enumerate(flow):

        x = flow_[0]
        y = flow_[1]
        middle_x = bbox[i][0] + bbox[i][2] // 2
        middle_y = bbox[i][1] + bbox[i][3] // 2
        middle_x = int(middle_x)
        middle_y = int(middle_y)


        x *= 10
        y *= 10

        cv2.arrowedLine(current_frame, (middle_x, middle_y), (middle_x + int(x), middle_y + int(y)), (0, 255, 0), 2)

    cv2.imwrite("test_video/current_frame_flow.jpg", current_frame)