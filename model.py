import torch
import torch.nn as nn

class MaskedAvgPooling(nn.Module):
    def forward(self, x, mask):
        mask = mask.unsqueeze(1).repeat(1, x.size(1), 1)  # [batch_size, d_model, max_len]
        x = x.masked_fill(mask, 0)
        valid_counts = (~mask).sum(dim=-1)
        # 检查 valid_counts 是否为 0
        if (valid_counts == 0).any():
            raise ValueError("valid_counts 中存在 0 值，可能导致除以零的错误")
        return x.sum(dim=-1) / valid_counts

class SoccerTransformer(nn.Module):
    def __init__(self, d_model=36, nhead=6, num_layers=4, max_len=20, d_model_team = 8):
        super(SoccerTransformer, self).__init__()
        
        self.pos_x_embedding = nn.Embedding(105 + 1,( d_model - d_model_team) // 2)
        self.pos_y_embedding = nn.Embedding(68 + 1, (d_model - d_model_team) // 2)
        self.team_embedding = nn.Embedding(3 + 1, d_model_team)
        
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True),
            num_layers=num_layers
        )
        
        self.pooling = MaskedAvgPooling()
        self.output_layer = nn.Linear(d_model, 1)

    def forward(self, player_features, mask):
        """
        player_features: [batch_size, max_len, 3]
        """
        x_grid = torch.floor(player_features[:,:,0] + 52.5).long()
        x_grid = torch.clamp(x_grid, 0, 104)
        y_grid = torch.floor(player_features[:,:,1] + 34).long()
        y_grid = torch.clamp(y_grid, 0, 67)
        team_id = player_features[:,:,2]
        team_id = team_id.clone()
        team_id[team_id == -1] = 2
        team_token = team_id.long() # [batch_size, max_len]

        # 把mask掉的位置填充为x_grid 填充105, y_grid 填充68,team_token 填充3
        x_grid = torch.where(mask == 0, x_grid, torch.full_like(x_grid, 105))
        y_grid = torch.where(mask == 0, y_grid, torch.full_like(y_grid, 68))
        team_token = torch.where(mask == 0, team_token, torch.full_like(team_token, 3))

        # print(x_grid)
        # print(y_grid)
        # print(team_token)

        grid_x_embedding = self.pos_x_embedding(x_grid)  # [batch_size, max_len, (d_model - d_model_team) // 2]
        grid_y_embedding = self.pos_y_embedding(y_grid)
        # [batch_size, max_len, (d_model - d_model_team) // 2]
        team_embedding = self.team_embedding(team_token)
        # [batch_size, max_len, d_model_team]

        x = torch.cat([grid_x_embedding, grid_y_embedding, team_embedding], dim=-1)
        # [batch_size, max_len, d_model]
        
        x = self.transformer_encoder(x, src_key_padding_mask=mask)  # [batch_size, max_len, d_model]
        x = x.permute(0, 2, 1)  # [batch_size, d_model, max_len]
        x = self.pooling(x, mask)  # Mask-aware pooling
        reward = self.output_layer(x)  # [batch_size, 1]
        return reward.squeeze(-1)  # [batch_size]