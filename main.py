import torch
import torch.nn as nn
from ViT import VisionTransformer  # ViTモデル
from SeT import SeriseTransformer  # SeTモデル

class BallDetectionModel(nn.Module):
    def __init__(self, seq_len=20, device=None):
        super(BallDetectionModel, self).__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seq_len = seq_len
        
        # Vision Transformer
        self.vit = VisionTransformer(
            img_h=224,
            img_w=224,
            patch_size=16,
            in_channels=3,
            embed_dim=512,
            num_heads=8,
            num_layers=6,
            num_classes=512,  # 中間特徴次元を出力
            hidden_dim=2048,
            dropout=0.1
        ).to(self.device)
        
        # Series Transformer
        self.set = SeriseTransformer(
            embed_dim=512,
            num_heads=8,
            num_layers=6,
            output_dim=3,  # 最終分類次元
            num_frames=seq_len,
            dropout=0.1
        ).to(self.device)

        self.vector_buffer = None

    def init_buffer(self, batch_size, device):
        self.vector_buffer = torch.zeros((batch_size, 0, 512), device=device)

    def forward(self, frames):
        frames = frames.to(self.device)
        batch_size, seq_len, channels, height, width = frames.shape
        if self.vector_buffer is None:
            self.init_buffer(batch_size, frames.device)

        # フレームごとにViTで処理
        vit_features = []
        for i in range(seq_len):
            frame_features = self.vit(frames[:, i, :, :, :])  # (batch_size, 512)
            vit_features.append(frame_features)
        vit_features = torch.stack(vit_features, dim=1)  # (batch_size, seq_len, 512)

        # バッファに蓄積
        self.vector_buffer = torch.cat((self.vector_buffer, vit_features), dim=1)
        if self.vector_buffer.size(1) > self.seq_len:
            self.vector_buffer = self.vector_buffer[:, -self.seq_len :, :]

        if self.vector_buffer.size(1) < self.seq_len:
            print("[DEBUG] Vector buffer not full yet.")
            return None

        # Series Transformerでシーケンス処理
        prediction = self.set(self.vector_buffer)
        return prediction

def simulate_frame_processing(batch_size=1, seq_len=20, num_frames=100, frame_size=(224, 224)):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BallDetectionModel(seq_len=seq_len, device=device)
    model.eval()

    for frame_idx in range(num_frames):
        frames = torch.randn(batch_size, 1, 3, *frame_size).to(device)
        with torch.no_grad():
            prediction = model(frames)
        if prediction is not None:
            print(f"Frame {frame_idx}: Prediction = {prediction.cpu().numpy()}")
        else:
            print(f"Frame {frame_idx}: Buffer not full yet.")

if __name__ == "__main__":
    simulate_frame_processing()
