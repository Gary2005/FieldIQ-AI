import torch
import torch.nn as nn

class SoccerTransformer(nn.Module):
    def __init__(self, d_model=16, nhead=4, num_layers=2, max_len=20):
        super(SoccerTransformer, self).__init__()
        
        self.input_proj = nn.Linear(5, d_model) 
        
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead),
            num_layers=num_layers
        )
        
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.output_layer = nn.Linear(d_model, 1)

    def forward(self, player_features):
        """
        player_features: [batch_size, max_len, 5]
        """
        # 线性映射到 d_model 维度
        x = self.input_proj(player_features)  # [batch_size, max_len, d_model]
        
        # 转置: [batch, max_len, d_model] -> [max_len, batch, d_model]
        x = x.permute(1, 0, 2)
        
        # Transformer 编码
        x = self.transformer_encoder(x)
        
        # 转置回来 [max_len, batch, d_model] -> [batch, max_len, d_model]
        x = x.permute(1, 2, 0)
        
        x = self.pooling(x).squeeze(-1)  # [batch_size, d_model]
        
        reward = self.output_layer(x)  # [batch_size, 1]
        
        return reward.squeeze(-1)  # [batch_size]
