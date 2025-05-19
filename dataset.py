import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

class SoccerDataset(Dataset):
    def __init__(self, data, mx_len = 20):
        """
        data: List of match situations. 
              Each situation is a list of player features + target.
              # team_id: 0 left, 1 right, -1 ball
              Example:
              [
                    ([{"x": 10, "y": 30, "vx": 0.5, "vy": -0.2, "team_id": 0},
                    {"x": 50, "y": 20, "vx": 0.1, "vy": 0.3, "team_id": 1},
                    ...],
                    target=10
                    )
              ]
        """

        data_ = []
        for situation in data:
            if len(situation[0]) > 0:
                data_.append(situation)
        data = data_

        mx_target = max([(situation[1]) for situation in data])
        mn_target = min([(situation[1]) for situation in data])
        print(f"max target: {mx_target}, min target: {mn_target}, len: {len(data)}")
        self.data = data
        self.mx_len = mx_len
        self.dx = 105/2
        self.dy = 68/2
        self.dv = 10

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        players_info, target = self.data[idx]
        players_features = []
        for player in players_info:
            player_tensor = torch.tensor([player["x"]/self.dx, player["y"]/self.dy, player["vx"]/self.dv, player["vy"]/self.dv, player["team_id"]], dtype=torch.float32)
            players_features.append(player_tensor)

        padding_mask = torch.zeros((self.mx_len,), dtype=torch.bool)

        if len(players_features) == 0:
            raise ValueError(f"No player features at index {idx}")
        else:
            players_features = torch.stack(players_features)
        if len(players_features) > self.mx_len:
            players_features = players_features[:self.mx_len]
        elif len(players_features) < self.mx_len:
            padding = torch.zeros((self.mx_len - len(players_features), players_features.shape[1]), dtype=torch.float32)
            padding_mask[len(players_features):] = True
            players_features = torch.cat((players_features, padding), dim=0)
        target_tensor = torch.tensor(target, dtype=torch.float32)
        if torch.isnan(players_features).any() or torch.isinf(players_features).any():
            raise ValueError(f"Invalid player features at index {idx}: {players_features}")
        if torch.isnan(target_tensor).any() or torch.isinf(target_tensor).any():
            raise ValueError(f"Invalid target at index {idx}: {target_tensor}")
        # check padding mask 不能都是 True
        if padding_mask.all():
            raise ValueError(f"Padding mask is all True at index {idx}: {padding_mask}")
        
        return players_features, target_tensor, padding_mask
    