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
    def __init__(self, d_model=16, nhead=4, num_layers=2, max_len=20):
        super(SoccerTransformer, self).__init__()
        
        self.input_proj = nn.Linear(3, d_model) 
        
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True),
            num_layers=num_layers
        )
        
        self.pooling = MaskedAvgPooling()
        self.output_layer = nn.Linear(d_model, 1)

    def forward(self, player_features, mask):
        """
        player_features: [batch_size, max_len, 5]
        """

        # only keep the 0,1,4 features(ignore vx, vy)
        player_features = player_features[:, :, [0, 1, 4]]

        x = self.input_proj(player_features)  # [batch_size, max_len, d_model]
        x = self.transformer_encoder(x, src_key_padding_mask=mask)  # [batch_size, max_len, d_model]
        x = x.permute(0, 2, 1)  # [batch_size, d_model, max_len]
        x = self.pooling(x, mask)  # Mask-aware pooling
        reward = self.output_layer(x)  # [batch_size, 1]
        return reward.squeeze(-1)  # [batch_size]