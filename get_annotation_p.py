import os  
import torch  
import numpy as np  
from PIL import Image  
import matplotlib.pyplot as plt  
import torchvision.transforms as T  
from pathlib import Path  


from plugins.calibration.nbjw_calib.model.cls_hrnet import get_cls_net  
from plugins.calibration.nbjw_calib.model.cls_hrnet_l import get_cls_net as get_cls_net_l  
from plugins.calibration.nbjw_calib.utils.utils_heatmap import (  
    get_keypoints_from_heatmap_batch_maxpool,  
    get_keypoints_from_heatmap_batch_maxpool_l,  
    complete_keypoints,  
    coords_to_dict  
)  
from tracklab.utils.download import download_file  

def kp_to_realworld(keypoints):  
    # Define the world coordinates for the main keypoints (57 points)  
    keypoint_world_coords_2D = [[0., 0.], [52.5, 0.], [105., 0.], [0., 13.84], [16.5, 13.84], [88.5, 13.84],  
                               [105., 13.84], [0., 24.84], [5.5, 24.84], [99.5, 24.84], [105., 24.84],  
                               [0., 30.34], [0., 30.34], [105., 30.34], [105., 30.34], [0., 37.66],  
                               [0., 37.66], [105., 37.66], [105., 37.66], [0., 43.16], [5.5, 43.16],  
                               [99.5, 43.16], [105., 43.16], [0., 54.16], [16.5, 54.16], [88.5, 54.16],  
                               [105., 54.16], [0., 68.], [52.5, 68.], [105., 68.], [16.5, 26.68],  
                               [52.5, 24.85], [88.5, 26.68], [16.5, 41.31], [52.5, 43.15], [88.5, 41.31],  
                               [19.99, 32.29], [43.68, 31.53], [61.31, 31.53], [85., 32.29], [19.99, 35.7],  
                               [43.68, 36.46], [61.31, 36.46], [85., 35.7], [11., 34.], [16.5, 34.],  
                               [20.15, 34.], [46.03, 27.53], [58.97, 27.53], [43.35, 34.], [52.5, 34.],  
                               [61.5, 34.], [46.03, 40.47], [58.97, 40.47], [84.85, 34.], [88.5, 34.],  
                               [94., 34.]]  # 57  
      
    # Define the auxiliary world coordinates (16 additional points)  
    keypoint_aux_world_coords_2D = [[5.5, 0], [16.5, 0], [88.5, 0], [99.5, 0], [5.5, 13.84], [99.5, 13.84],   
                                   [16.5, 24.84], [88.5, 24.84], [16.5, 43.16], [88.5, 43.16], [5.5, 54.16],   
                                   [99.5, 54.16], [5.5, 68], [16.5, 68], [88.5, 68], [99.5, 68]]  
      
    # Convert coordinates to be centered at the middle of the pitch (optional, but matches the codebase)  
    keypoint_world_coords_2D = [[x - 52.5, -(y - 34)] for x, y in keypoint_world_coords_2D]  
    keypoint_aux_world_coords_2D = [[x - 52.5, -(y - 34)] for x, y in keypoint_aux_world_coords_2D]  
      
    # Create a dictionary to store the real-world coordinates  
    real_world_coords = []
      
    # Map each detected keypoint to its real-world coordinates  
    for kp_idx in keypoints.keys():  
        # Get the corresponding world coordinates  
        if kp_idx <= 57:  
            # Main keypoints (1-57)  
            xw, yw = keypoint_world_coords_2D[kp_idx - 1]  
            # Set z-coordinate to -2.44 for goal posts, 0 for other points  
            zw = -2.44 if kp_idx in [12, 15, 16, 19] else 0.0  
        elif kp_idx <= 73:  # Auxiliary keypoints (58-73)  
            xw, yw = keypoint_aux_world_coords_2D[kp_idx - 1 - 57]  
            zw = 0.0  # All auxiliary points are on the ground plane  
        else:  
            continue  # Skip invalid keypoint indices  
          
        # Add to the dictionary with both image and world coordinates  
        real_world_coords.append({  
            'x_image': keypoints[kp_idx]['x'],  
            'y_image': keypoints[kp_idx]['y'],  
            'x_world': xw,  
            'y_world': yw,  
            'z_world': zw  
        }  )
      
    return real_world_coords

def run_demo(images, model, model_l):  


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Available GPUs: {torch.cuda.device_count()}")  # 打印有几张卡
    # print(f"Model on GPUs: {model.device_ids}")

    model = torch.nn.DataParallel(model).to(device)
    model_l = torch.nn.DataParallel(model_l).to(device)
    
    # Preprocess all images and stack them into a batch
    tfms_resize = T.Compose([  
        T.Resize((540, 960)),  
        T.ToTensor()  
    ])  

    image_tensors = []
    for image in images:
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")  
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)  
        elif not isinstance(image, Image.Image):
            raise ValueError("Each image should be a file path, PIL Image, or numpy array.")  
        
        # if not RGB, convert to RGB
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        image_tensor = tfms_resize(image)
        image_tensors.append(image_tensor)
    
    # Create a batch tensor of shape [N, C, H, W]
    batch_tensor = torch.stack(image_tensors).to(device)
    
    # Run inference in a single forward pass
    with torch.no_grad():  
        heatmaps = model(batch_tensor)  
        heatmaps_l = model_l(batch_tensor)  

    annotations = []
    batch_size = batch_tensor.size(0)
    for i in range(batch_size):
        # 切出第 i 张图的 heatmap，保持 batch dim=1
        hm   = heatmaps  [i:i+1, :-1, :, :]
        hm_l = heatmaps_l[i:i+1, :-1, :, :]

        # 单图调用
        kp_coords   = get_keypoints_from_heatmap_batch_maxpool   (hm)
        line_coords = get_keypoints_from_heatmap_batch_maxpool_l (hm_l)

        # 转 dict、补全、转线
        kp_dict    = coords_to_dict(kp_coords,   threshold=0.1449)
        lines_dict = coords_to_dict(line_coords, threshold=0.2983)
        final_dict = complete_keypoints(
            kp_dict, lines_dict,
            w=batch_tensor.size(-1),
            h=batch_tensor.size(-2),
            normalize=True
        )
        lines = kp_to_realworld(final_dict[0])
        annotations.append(lines)

    return annotations
