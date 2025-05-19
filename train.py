from torch.utils.data import DataLoader
import torch.optim as optim
from dataset import SoccerDataset
from model import SoccerTransformer
import torch
import torch.nn as nn
from tqdm import tqdm
import wandb
import json
import os
import random
from pitch import get_pitch_from_pt

# ===============================
# 配置超参数
# ===============================
config = {
    "batch_size": 64,
    "learning_rate": 1e-3,
    "epochs": 10,
    "d_model": 16,
    "nhead": 4,
    "num_layers": 2,
    "max_len": 20,
    "valid_step": 100,
    "visualize_sample": 8
}

# ===============================
# 初始化 wandb
# ===============================

def process_data(json_paths):
    data = []
    for json_path in json_paths:
        with open(json_path, "r") as f:
            match_data = json.load(f)
            data.extend(match_data)

    # shuffle 后选择 512 条数据作为测试集
    random.shuffle(data)
    test_data = data[:512]
    data = data[512:]

    return data, test_data

# ===============================
# 准备数据
# ===============================
json_paths = ["game_example/cleaned_data.json"]
data, test_data = process_data(json_paths)
train_dataset = SoccerDataset(data, mx_len=config["max_len"])
test_dataset = SoccerDataset(test_data, mx_len=config["max_len"])
dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
dataloader_test = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)
wandb.init(project="soccer-transformer", name="training-run-1", config=config)

# ===============================
# 初始化模型和优化器
# ===============================
model = SoccerTransformer(
    d_model=config["d_model"],
    nhead=config["nhead"],
    num_layers=config["num_layers"],
    max_len=config["max_len"]
)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

# 将模型参数和配置记录到 wandb
wandb.watch(model, log="all")

# ===============================
# 训练循环
# ===============================
for epoch in range(config["epochs"]):
    epoch_loss = 0
    with tqdm(dataloader, desc=f"Epoch {epoch + 1}", leave=False) as pbar:
        step = 0
        for player_features, target in pbar:
            optimizer.zero_grad()
            output = model(player_features)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # 更新进度条和 loss 统计
            pbar.set_postfix(loss=loss.item())
            epoch_loss += loss.item()
            
            # 记录到 wandb
            wandb.log({
                "Batch Loss": loss.item(),
                "Learning Rate": config["learning_rate"]
            })
            if step % config["valid_step"] == 0 or step == len(dataloader) - 1:
                model.eval()
                with torch.no_grad():
                    val_loss = 0
                    table = wandb.Table(columns=["Step", "Target", "Prediction", "Visualization"])
                    sampled_indices = random.sample(range(len(test_dataset)), min(config["visualize_sample"], len(test_dataset)))
                    
                    for idx in sampled_indices:
                        val_features, val_target = test_dataset[idx]
                        val_output = model(val_features.unsqueeze(0))
                        image = get_pitch_from_pt(val_features)
                        table.add_data(step, val_target.item(), val_output[0].item(), wandb.Image(image))

                    wandb.log({"Validation Samples": table})

                    for val_player_features, val_target in dataloader_test:
                        val_output = model(val_player_features)
                        val_loss += criterion(val_output, val_target).item()
                    val_loss /= len(dataloader_test)
                    wandb.log({"Validation Loss": val_loss})
                print(f"Validation Loss: {val_loss:.4f}")
                model.train()
            step += 1

    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch {epoch + 1} Completed. Average Loss = {avg_loss:.4f}")
    
    # 记录 epoch 结束时的平均 loss 到 wandb
    wandb.log({"Epoch": epoch + 1, "Average Loss": avg_loss})

# ===============================
# 结束 wandb 记录
# ===============================
wandb.finish()
