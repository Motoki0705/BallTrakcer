import torch
import torch.nn as nn
import math

class AddCLSToken(nn.Module):
    """[CLS] トークンを追加するモジュール"""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        return torch.cat((cls_tokens, x), dim=1)

class AddPositionalEmbedding(nn.Module):
    """位置エンコーディングを追加するモジュール"""
    def __init__(self, num_frames: int, embed_dim: int):
        super().__init__()
        self.position_embedding = nn.Parameter(
            torch.randn(1, num_frames + 1, embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.position_embedding

class MultiHeadAttention(nn.Module):
    """マルチヘッド注意機構"""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (embed_dim // num_heads) ** -0.5
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.o_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, embed_dim = x.size()
        qkv = self.qkv_proj(x).chunk(3, dim=-1)
        q, k, v = [t.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2) for t in qkv]

        attn_weights = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = attn_weights.softmax(dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = (attn_weights @ v).transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
        return self.o_proj(attn_output)

class FeedForward(nn.Module):
    """位置ごとの前向きフィードフォワード層"""
    def __init__(self, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim*4, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class EncoderBlock(nn.Module):
    """Transformerエンコーダブロック"""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ff = FeedForward(embed_dim, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x

class SeriseTransformer(nn.Module):
    """Vision Transformer"""
    def __init__(self,embed_dim: int, num_heads: int, num_layers: int, output_dim: int, num_frames: int, dropout: float = 0.1):
        super().__init__()
        
        self.add_cls_token = AddCLSToken(embed_dim)
        self.add_positional_embedding = AddPositionalEmbedding(num_frames, embed_dim)
        self.encoder = nn.Sequential(
            *[EncoderBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)]
        )
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.add_cls_token(x)
        x = self.add_positional_embedding(x)
        x = self.encoder(x)
        return self.mlp_head(x[:, 0])

if __name__ == '__main__':
    #ハイパーパラメータ
    embed_dim = 512
    num_heads = 8
    num_layers = 6
    output_dim = 512
    num_frames = 20
    dropout_rate = 0.1
    
    #モデルの初期化
    sequence_model = SeriseTransformer(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        output_dim=output_dim,
        num_frames=num_frames,
        dropout=0.1
        )
    feature_vectors = torch.randn(1, num_frames, embed_dim)  # Example feature vectors with shape (batch_size, seq_len, d_model)
    final_output = sequence_model(feature_vectors)
    print(final_output.shape)
