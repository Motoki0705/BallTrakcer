import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class PatchEmbedding(nn.Module):
    """画像をパッチに分割し埋め込みを作成するモジュール"""
    def __init__(self, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.projection = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0)
        x = self.projection(x)
        return x.flatten(2).transpose(1, 2)

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
    def __init__(self, num_patches: int, embed_dim: int):
        super().__init__()
        self.position_embedding = nn.Parameter(
            torch.randn(1, num_patches + 1, embed_dim)
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
    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class EncoderBlock(nn.Module):
    """Transformerエンコーダブロック"""
    def __init__(self, embed_dim: int, num_heads: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ff = FeedForward(embed_dim, hidden_dim, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    """Vision Transformer"""
    def __init__(self, img_h: int, img_w: int, patch_size: int, in_channels: int, embed_dim: int, 
                 num_heads: int, num_layers: int, num_classes: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.patch_embedding = PatchEmbedding(patch_size, in_channels, embed_dim)
        
        grid_h = (img_h + (patch_size - img_h % patch_size) % patch_size) // patch_size
        grid_w = (img_w + (patch_size - img_w % patch_size) % patch_size) // patch_size
        num_patches = grid_h * grid_w
        
        self.add_cls_token = AddCLSToken(embed_dim)
        self.add_positional_embedding = AddPositionalEmbedding(num_patches, embed_dim)
        self.encoder = nn.Sequential(
            *[EncoderBlock(embed_dim, num_heads, hidden_dim, dropout) for _ in range(num_layers)]
        )
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embedding(x)
        x = self.add_cls_token(x)
        x = self.add_positional_embedding(x)
        x = self.encoder(x)
        return self.mlp_head(x[:, 0])
 
if __name__ == '__main__':
    import torch
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    import torch.optim as optim
    import torch.nn as nn

    # データセットの前処理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ViTに適したサイズにリサイズ
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 正規化
    ])

    # CIFAR-10データセットの読み込み
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # モデルのインスタンス化
    vit_model = VisionTransformer(
        img_h=224,
        img_w=224,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        num_heads=8,
        num_layers=12,
        num_classes=10,  # CIFAR-10は10クラス
        hidden_dim=3072,
        dropout=0.1
    )

    # 損失関数とオプティマイザ
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(vit_model.parameters(), lr=0.001)

    # 学習ループ
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vit_model.to(device)

    num_epochs = 10

    for epoch in range(num_epochs):
        vit_model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = vit_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

    # テストデータで評価
    vit_model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = vit_model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy: {100 * correct / total:.2f}%")
