import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from model import HomographyNet
import torch.nn as nn


class SoccerDataset(Dataset):
    def __init__(self, root, split="train", transform=None):
        """
        参数：
            root: 文件根路径，例如 "/share/guwanjun-local/DeepSoccer"
            split: 数据集划分，可选 "train", "test", "valid"
            transform: 对图像进行的预处理操作，默认为将图片转换为 Tensor
        """
        self.split = split
        self.root = os.path.join(root, "calibration-2023", split)
        self.transform = transform if transform is not None else transforms.ToTensor()
        self.items = []
        # 遍历目录下所有jpg文件，并检查是否存在对应的 _H.json 文件
        for filename in os.listdir(self.root):
            if filename.endswith(".jpg"):
                base = os.path.splitext(filename)[0]
                h_json = f"{base}_H.json"
                if os.path.exists(os.path.join(self.root, h_json)):
                    self.items.append(base)
        self.items.sort()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        base = self.items[idx]
        # 图片路径
        img_path = os.path.join(self.root, f"{base}.jpg")
        # 对应的_H.json路径
        json_path = os.path.join(self.root, f"{base}_H.json")
        # 读取图片并应用预处理
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        # 读取并转换_H.json（保存为列表格式的 3x3 矩阵）
        with open(json_path, "r") as f:
            H = json.load(f)
        H = torch.tensor(H, dtype=torch.float32)
        H = H.view(-1)
        # L2 归一化
        H = nn.functional.normalize(H, p=2, dim=0)
        return image, H

if __name__ == "__main__":
    # 示例：使用默认转换将图片转换为Tensor
    dataset = SoccerDataset(root="/share/guwanjun-local/DeepSoccer", split="test")
    max_value = 0
    count = 0
    for i in range(len(dataset)):
        img, H = dataset[i]
        if abs(H).max().item() > 1e6:
            count +=1
        max_value = max(max_value, abs(H).max().item())
    print(count)
    print("Max value in H:", max_value)
    model = HomographyNet().to("cuda")
    print(f"Dataset size: {len(dataset)}")
    img, H = dataset[0]
    print(img)
    print("Image shape:", img.shape)
    print("H matrix:", H)
    output = model(img.unsqueeze(0).to("cuda"))
    print("Model output", output)
    import torch.nn as nn
    loss = nn.MSELoss()
    target = H.view(1, -1).to("cuda")
    print("loss:", loss(output, target))
    print("Model output shape:", output.shape)