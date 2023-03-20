import math

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.embedding = Embedding(model_cfg)
        self.transformer_encoder = TransformerEncoder(model_cfg)
        self.mlp_head = MLPHead(model_cfg)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        x = x[:,0,:]
        x = self.mlp_head(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size = x.size(0)
        query = self.query(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attention = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention = torch.softmax(attention, dim=-1)
        attention = torch.matmul(attention, value).transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return self.out(attention)

class TransformerEncoder(nn.Module):
    def __init__(self, model_cfg):
        super(TransformerEncoder, self).__init__()
        num_layers = model_cfg["num_layers"]
        hid_dim = model_cfg["hidden_dim"]
        feedforward_dim = model_cfg["feedforward_dim"]
        dropout_rate = model_cfg["dropout_rate"]
        num_heads = model_cfg["num_heads"]

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer_norm1 = nn.LayerNorm(hid_dim)
            multi_head_attention = MultiHeadAttention(hid_dim, num_heads)
            layer_norm2 = nn.LayerNorm(hid_dim)
            position_wise_ffn = nn.Sequential(
                nn.Linear(hid_dim, feedforward_dim),
                nn.ReLU(),
                nn.Linear(feedforward_dim, hid_dim)
            )
            dropout = nn.Dropout(dropout_rate)

            self.layers.append(
                nn.ModuleList([layer_norm1, multi_head_attention, layer_norm2, position_wise_ffn, dropout])
            )

    def forward(self, x):
        for layer_norm1, multi_head_attention, layer_norm2, position_wise_ffn, dropout in self.layers:
            x = layer_norm1(x)
            x = x + dropout(multi_head_attention(x))
            x = layer_norm2(x)
            x = x + dropout(position_wise_ffn(x))
        return x

class Embedding(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        patch_size = model_cfg["patch_size"]
        in_channels = model_cfg["in_channels"]
        hid_dim = model_cfg["hidden_dim"]
        num_patches = model_cfg["num_patches"]
        device = torch.device('cuda' if model_cfg['device'] == 'cuda' and torch.cuda.is_available() else 'cpu')


        self.devider = Devider(model_cfg)
        self.patch_embedding = nn.Linear(
            patch_size * patch_size * in_channels,
            hid_dim,
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, hid_dim))
        self.position_embedding = self.sinusoidal_positional_embedding(
            1 + num_patches, hid_dim, device
        )

    def forward(self, x):
        x = self.devider(x)
        x = self.patch_embedding(x)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.position_embedding
        return x

    @staticmethod
    def sinusoidal_positional_embedding(seq_len, hidden_dim, device=None, dtype=None):
        position = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2, dtype=torch.float32, device=device) * -(math.log(10000.0) / hidden_dim))
        pos_emb = torch.empty(seq_len, hidden_dim, device=device, dtype=dtype)
        pos_emb[:, 0::2] = torch.sin(position * div_term)
        pos_emb[:, 1::2] = torch.cos(position * div_term)
        return pos_emb.unsqueeze(0)

class MLPHead(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        # self.pool = nn.AdaptiveAvgPool1d(1)
        hid_dim = model_cfg["hidden_dim"]
        num_classes = model_cfg['num_classes']
        self.fc = nn.Linear(hid_dim, num_classes)

    def forward(self, x):
        # x = self.pool(x.permute(0, 2, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class Devider(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.patch_size = model_cfg["patch_size"]
        self.num_patches = None

    def forward(self, x):
        B, C, H, W = x.shape
        P = self.patch_size
        self.num_patches = (H // P) * (W // P)
        x = x.unfold(2, P, P).unfold(3, P, P)
        x = x.reshape(B, self.num_patches, C * P * P)
        return x