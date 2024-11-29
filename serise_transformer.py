import torch
import torch.nn as nn
import math

class TransformerSequenceModel(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super(TransformerSequenceModel, self).__init__()
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.positional_encoding = AddPositionalEncoding(d_model, max_len=20)

    def forward(self, x, mask=None):
        # x: (batch_size, seq_len, d_model), where seq_len is 20
        x = self.positional_encoding(x)
        for layer in self.encoder_layers:
            x = layer(x, mask=mask)
        return x[:, -1, :]  # Return only the last element of the sequence

# Required Components (copied from TransformerEncoderModule for completeness)
class AddPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(AddPositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.encoding[:, :seq_len, :].to(x.device)
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
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))

class ResidualNormalizationWrapper(nn.Module):
    def __init__(self, d_model, sublayer):
        super(ResidualNormalizationWrapper, self).__init__()
        self.sublayer = sublayer
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, *args, **kwargs):
        return x + self.sublayer(self.norm(x), *args, **kwargs)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = ResidualNormalizationWrapper(d_model, MultiHeadAttention(d_model, num_heads))
        self.ffn = ResidualNormalizationWrapper(d_model, FeedForwardNetwork(d_model, d_ff, dropout))

    def forward(self, x, mask=None):
        x = self.self_attn(x, mask=mask)
        x = self.ffn(x)
        return x

if __name__ == '__main__':
    # Example usage
    feature_vectors = torch.randn(1, 20, 512)  # Example feature vectors with shape (batch_size, sequence_length=20, d_model=512)
    sequence_model = TransformerSequenceModel(d_model=512, num_heads=8, d_ff=2048, num_layers=6, dropout=0.1)
    final_output = sequence_model(feature_vectors)  # Output shape will be (batch_size, d_model)
    print(final_output.shape)
