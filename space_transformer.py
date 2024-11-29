import torch
import torch.nn as nn
import math

class AddPositionalEncoding(nn.Module):
    def __init__(self, d_model, height, width):
        super(AddPositionalEncoding, self).__init__()
        self.height = height
        self.width = width
        self.d_model = d_model

        # Create positional encodings for rows and columns separately
        self.row_encoding = torch.zeros(height, d_model)
        self.col_encoding = torch.zeros(width, d_model)

        row_position = torch.arange(0, height, dtype=torch.float).unsqueeze(1)
        col_position = torch.arange(0, width, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        self.row_encoding[:, 0::2] = torch.sin(row_position * div_term)
        self.row_encoding[:, 1::2] = torch.cos(row_position * div_term)
        self.col_encoding[:, 0::2] = torch.sin(col_position * div_term)
        self.col_encoding[:, 1::2] = torch.cos(col_position * div_term)

        self.row_encoding = self.row_encoding.unsqueeze(1)  # (height, 1, d_model)
        self.col_encoding = self.col_encoding.unsqueeze(0)  # (1, width, d_model)

    def forward(self, x):
        # x: (batch_size, height, width, d_model)
        x = x + self.row_encoding.to(x.device) + self.col_encoding.to(x.device)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_head = d_model // num_heads
        self.num_heads = num_heads
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.scale = math.sqrt(self.d_head)

    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        out = self.out(context)
        return out

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForwardNetwork, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.linear2(self.dropout(torch.relu(self.linear1(x))))
        return x

class ResidualNormalizationWrapper(nn.Module):
    def __init__(self, d_model, sublayer):
        super(ResidualNormalizationWrapper, self).__init__()
        self.sublayer = sublayer
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, *args, **kwargs):
        out = x + self.sublayer(self.norm(x), *args, **kwargs)
        return out

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = ResidualNormalizationWrapper(d_model, MultiHeadAttention(d_model, num_heads))
        self.ffn = ResidualNormalizationWrapper(d_model, FeedForwardNetwork(d_model, d_ff, dropout))

    def forward(self, x, mask=None):
        x = self.self_attn(x, mask=mask)
        x = self.ffn(x)
        return x

class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, height, width, dropout=0.1):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.pos_encoding = AddPositionalEncoding(d_model, height, width)

    def forward(self, x, mask=None):
        # x: (batch_size, height, width, d_model)
        batch_size, height, width, d_model = x.shape
        x = x.view(batch_size, height * width, d_model)  # Flatten spatial dimensions
        x = self.pos_encoding(x.view(batch_size, height, width, d_model))
        x = x.view(batch_size, height * width, d_model)  # Flatten again for attention layers
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x[:, -1, :]  # Return only the last element of the sequence

if __name__ == '__main__':
    # Example usage
    feature_vectors = torch.randn(1, 15, 9, 512)  # Example feature vectors with shape (batch_size, height, width, d_model)
    encoder = Encoder(d_model=512, num_heads=8, d_ff=2048, num_layers=6, height=15, width=9, dropout=0.1)
    encoded_output = encoder(feature_vectors)  # Output shape will be (batch_size, d_model)
    print(encoded_output.shape)
