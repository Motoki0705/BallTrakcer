import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from patcher import VideoFramePatcher
from ImgToVec import ImgToVec
from space_transformer import Encoder as SpaceTransformer
from serise_transformer import TransformerSequenceModel
from tqdm import tqdm

import torch
import torch.nn as nn
from patcher import VideoFramePatcher
from ImgToVec import ImgToVec
from space_transformer import Encoder as SpaceTransformer
from serise_transformer import TransformerSequenceModel

class BallDetectionModel(nn.Module):
    def __init__(self, seq_len=20):
        super(BallDetectionModel, self).__init__()
        self.seq_len = seq_len
        self.patcher = VideoFramePatcher(patch_size=(128, 128))
        self.img_to_vec = ImgToVec()
        self.space_transformer = SpaceTransformer(d_model=512, num_heads=8, d_ff=2048, num_layers=6, height=15, width=9, dropout=0.1)
        self.sequence_transformer = TransformerSequenceModel(d_model=512, num_heads=8, d_ff=2048, num_layers=6, dropout=0.1)
        self.output_layer = nn.Linear(512, 3)

        # フレームバッファを初期化（各バッチで最新の20フレームを保存）
        self.frame_buffer = None

    def init_buffer(self, batch_size, device):
        # フレームバッファをバッチサイズとデバイスに合わせて初期化
        self.frame_buffer = torch.zeros((batch_size, self.seq_len, 3, 1920, 1080), device=device)

    def forward(self, frames):
        batch_size, _, channels, height, width = frames.shape

        # 初回実行時にバッファを初期化
        if self.frame_buffer is None:
            self.init_buffer(batch_size, frames.device)

        # バッファをシフトして新しいフレームを追加
        self.frame_buffer[:, :-1, :, :, :] = self.frame_buffer[:, 1:, :, :, :].clone()  # 修正箇所
        self.frame_buffer[:, -1, :, :, :] = frames[:, 0, :, :, :]

        # バッファが埋まっていない場合は処理をスキップ
        if not torch.all(self.frame_buffer[:, 0, :, :, :] != 0):
            return None

        # バッファ内のフレームを処理
        all_predictions = []
        for t in range(self.seq_len):
            patches = self.patcher(self.frame_buffer[:, t, :, :, :])  # (batch_size, 15, 9, 3, 128, 128)
            patch_features = self.img_to_vec(patches)  # (batch_size, 15, 9, 512)
            spatial_features = self.space_transformer(patch_features)  # (batch_size, 512)
            all_predictions.append(spatial_features)

        # 時系列データとしてTransformerに渡す
        sequence_features = torch.stack(all_predictions, dim=1)  # (batch_size, seq_len, 512)
        final_output = self.sequence_transformer(sequence_features)  # (batch_size, 512)

        # 最終的な予測（座標とフラグを出力）
        prediction = self.output_layer(final_output)  # (batch_size, 3)
        return prediction

if __name__ == "__main__":
    batch_size = 1
    model = BallDetectionModel(seq_len=20)
    model.eval()

    # シミュレーション: 25フレームを順次入力
    for frame_idx in range(25):
        frames = torch.randn(batch_size, 1, 3, 1920, 1080)  # 1フレームずつ入力
        prediction = model(frames)

        if prediction is not None:
            print(f"Frame {frame_idx}: Prediction shape {prediction.shape}")
        else:
            print(f"Frame {frame_idx}: Waiting for more frames...")
