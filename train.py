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
import numpy as np
json_paths = ["2015-02-21 - 18-00 Swansea 2 - 1 Manchester United/cleaned_data.json", "2015-09-26 - 17-00 Liverpool 3 - 2 Aston Villa/cleaned_data.json", "2016-01-13 - 22-45 Chelsea 2 - 2 West Brom"]

# ===============================
# 配置超参数
# ===============================
config = {
    "name": "see_all",
    "batch_size": 128,
    "learning_rate": 1e-5,
    "epochs": 30,
    "d_model": 64,
    "nhead": 8,
    "num_layers": 4,
    "max_len": 23,
    "valid_step": 100,
    "visualize_sample": 8,
    "weight_decay": 1e-4,
}
device = "cuda:7"

# ===============================
# 初始化 wandb
# ===============================

# torch.autograd.set_detect_anomaly(True)

def process_data(json_paths):
    data = []
    for json_path in json_paths:
        with open(json_path, "r") as f:
            match_data = json.load(f)
            data.extend(match_data)
    
    # 选出B条数据作为测试集
    B = 2048
    # shuffle data
    random.shuffle(data)
    test_data = data[:B]
    data = data[B:]
    
    return data, test_data

# ===============================
# 准备数据
# ===============================
data, test_data = process_data(json_paths)
train_dataset = SoccerDataset(data, mx_len=config["max_len"])
test_dataset = SoccerDataset(test_data, mx_len=config["max_len"], data_augmentation=False)
dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
dataloader_test = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)
wandb.init(project="soccer-transformer", name="training-run-1", config=config)

# ===============================
# 初始化模型和优化器
# ===============================
# def init_weights(m):
#     if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
#         nn.init.xavier_uniform_(m.weight)
#         if m.bias is not None:
#             nn.init.zeros_(m.bias)

model = SoccerTransformer(
    d_model=config["d_model"],
    nhead=config["nhead"],
    num_layers=config["num_layers"],
    max_len=config["max_len"]
).to(device)
# model.apply(init_weights)
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)


# 将模型参数和配置记录到 wandb
wandb.watch(model, log="all")

# ===============================
# 训练循环
# ===============================
# table = wandb.Table(columns=["Epoch", "Step", "Target", "Prediction", "Visualization"])


for epoch in range(config["epochs"]):
    epoch_loss = 0
    with tqdm(dataloader, desc=f"Epoch {epoch + 1}", leave=False) as pbar:
        step = 0
        for player_features, target, mask in pbar:
            if step % config["valid_step"] == 0 or step == len(dataloader) - 1:
                model.eval()
                with torch.no_grad():
                    val_loss = 0
                    # table = wandb.Table(columns=["Epoch", "Step", "Target", "Prediction", "Visualization"])
                    # sampled_indices = random.sample(range(len(test_dataset)), min(config["visualize_sample"], len(test_dataset)))
                    
                    # for idx in sampled_indices:
                    #     val_features, val_target, val_mask = test_dataset[idx]
                    #     val_features = val_features.to(device)
                    #     val_target = val_target.to(device)
                    #     val_mask = val_mask.to(device)
                    #     # print(val_mask)
                    #     # print(val_features)
                    #     # print(val_target)
                    #     val_output = model(val_features.unsqueeze(0), val_mask.unsqueeze(0))
                    #     image = get_pitch_from_pt(val_features)
                    #     table.add_data(epoch, step, val_target.item(), val_output[0].item(), wandb.Image(image))

                    # wandb.log({f"Samples_Epoch_{epoch}_Step_{step}": table})

                    for val_player_features, val_target, val_mask in dataloader_test:
                        val_player_features = val_player_features.to(device)
                        val_target = val_target.to(device)
                        val_mask = val_mask.to(device)
                        val_output = model(val_player_features, val_mask)
                        val_loss += criterion(val_output, val_target).item()
                    val_loss /= len(dataloader_test)
                    wandb.log({"Validation Loss": val_loss, "Log Validation Loss": np.log(val_loss)})
                print(f"Validation Loss: {val_loss:.4f}")

                # save the ckpoint
                if not os.path.exists(f"checkpoints/{config['name']}"):
                    os.makedirs(f"checkpoints/{config['name']}")
                if not os.path.exists(f"checkpoints/{config['name']}/latest"):
                    os.makedirs(f"checkpoints/{config['name']}/latest")
                torch.save(model.state_dict(), f"checkpoints/{config['name']}/model_epoch_{epoch + 1}_step_{step}_loss_{val_loss:.4f}.pth")
                torch.save(model.state_dict(), f"checkpoints/{config['name']}/latest/model_latest.pth")
                print(f"Model saved at checkpoints/{config['name']}/model_epoch_{epoch + 1}_step_{step}_loss_{val_loss:.4f}.pth")
                # 如果checkpoints/{config['name']} 存在>=3个文件，删除val_loss最大的文件
                ckpt_files = [f for f in os.listdir(f"checkpoints/{config['name']}") if f.endswith(".pth")]
                if len(ckpt_files) > 3:
                    # 解析文件名中的 val_loss
                    losses = []
                    for filename in ckpt_files:
                        try:
                            loss = float(filename.split("_loss_")[1].replace(".pth", ""))
                            losses.append((loss, filename))
                        except (IndexError, ValueError):
                            continue

                    # 找到最大的 loss 文件并删除
                    if losses:
                        max_loss_file = max(losses, key=lambda x: x[0])[1]
                        os.remove(os.path.join(f"checkpoints/{config['name']}", max_loss_file))
                        print(f"Removed checkpoint: {max_loss_file}")

                model.train()
            step += 1
            player_features = player_features.to(device)
            target = target.to(device)
            mask = mask.to(device)
            optimizer.zero_grad()
            # print(player_features.shape, target.shape, mask.shape)
            output = model(player_features,mask)
            loss = criterion(output, target)
            # torch.autograd.set_detect_anomaly(True)
            loss.backward()
            # torch.autograd.set_detect_anomaly(False)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            # 更新进度条和 loss 统计
            pbar.set_postfix(loss=loss.item())
            epoch_loss += loss.item()
            
            # 记录到 wandb
            wandb.log({
                "Batch Loss": loss.item(),
                "Log Batch Loss": np.log(loss.item()),
                "Learning Rate": optimizer.param_groups[0]["lr"]
            })

    avg_loss = epoch_loss / len(dataloader)
    scheduler.step()
    print(f"Epoch {epoch + 1} Completed. Average Loss = {avg_loss:.4f}")
    
    # 记录 epoch 结束时的平均 loss 到 wandb
    wandb.log({"Epoch": epoch + 1, "Average Loss": avg_loss, "Log Average Loss": np.log(avg_loss)})

# ===============================
# 结束 wandb 记录
# ===============================
wandb.finish()
