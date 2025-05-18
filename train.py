from torch.utils.data import DataLoader
import torch.optim as optim
from dataset import SoccerDataset
from model import SoccerTransformer
import torch
import torch.nn as nn
from tqdm import tqdm
import wandb

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
    "max_len": 20
}

# ===============================
# 初始化 wandb
# ===============================
wandb.init(project="soccer-transformer", name="training-run-1", config=config)

def process_data(json_paths):
    data = 
    return data

# ===============================
# 准备数据
# ===============================
json_paths = ["game_example/cleaned_data.json"]
dataset = SoccerDataset(process_data(json_paths), mx_len=config["max_len"])
dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

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

    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch {epoch + 1} Completed. Average Loss = {avg_loss:.4f}")
    
    # 记录 epoch 结束时的平均 loss 到 wandb
    wandb.log({"Epoch": epoch + 1, "Average Loss": avg_loss})

# ===============================
# 结束 wandb 记录
# ===============================
wandb.finish()
