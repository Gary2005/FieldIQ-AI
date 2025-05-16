import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from model import HomographyNet
from dataset import SoccerDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from tqdm import tqdm
import wandb

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()

    bar = tqdm(dataloader, desc="Training", unit="batch")

    for images, targets in bar:
        images = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets.view(outputs.size(0), -1))
        loss.backward()
        optimizer.step()
        bar.set_postfix({"loss": loss.item()})
        wandb.log({"train_loss": loss.item()})
        print(f"train: {targets[0], outputs[0]}")

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        bar = tqdm(dataloader, desc="Evaluating", unit="batch")
        for images, targets in bar:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets.view(outputs.size(0), -1))
            running_loss += loss.item() * images.size(0)
            print(f"eval: {targets[0], outputs[0]}")
    epoch_loss = running_loss / len(dataloader.dataset)
    wandb.log({"test_loss": epoch_loss})
    return epoch_loss

def main():

    wandb.init(
        project="soccer_homography",
        config={
            "learning_rate": 1e-3,
            "batch_size": 64,
            "num_epochs": 20
        }
    )

    # 超参数设置
    num_epochs = 20
    batch_size = 64
    learning_rate = 1e-3

    # 加载数据集
    train_ds = SoccerDataset("./", "train")
    valid_ds = SoccerDataset("./", "valid")
    # 合并 train 和 valid 数据集
    train_valid_ds = ConcatDataset([train_ds, valid_ds])
    test_ds = SoccerDataset("./", "test")

    train_loader = DataLoader(train_valid_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # 创建模型、损失函数与优化器
    model = HomographyNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练过程
    for epoch in range(num_epochs):
        train_epoch(model, train_loader, criterion, optimizer, device)
        loss = evaluate(model, test_loader, criterion, device)
        # save_checkpoint
        torch.save(model.state_dict(), f"models/model_epoch_{epoch}_loss_{loss:.4f}.pth")


if __name__ == "__main__":
    main()
    