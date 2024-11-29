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
import torch
import torch.nn as nn
import tracemalloc  # メモリ使用量の追跡用

# その他のインポートは省略

import torch
import torch.nn as nn
from patcher import VideoFramePatcher
from ImgToVec import ImgToVec
from space_transformer import Encoder as SpaceTransformer
from serise_transformer import TransformerSequenceModel

class BallDetectionModel(nn.Module):
    def __init__(self, seq_len=20):
        super(BallDetectionModel, self).__init__()
        self.seq_len = seq_len  # ベクトル列の長さ
        self.patcher = VideoFramePatcher(patch_size=(128, 128))
        self.img_to_vec = ImgToVec()
        self.space_transformer = SpaceTransformer(d_model=512, num_heads=8, d_ff=2048, num_layers=6, height=15, width=9, dropout=0.1)
        self.sequence_transformer = TransformerSequenceModel(d_model=512, num_heads=8, d_ff=2048, num_layers=6, dropout=0.1)
        self.output_layer = nn.Linear(512, 3)

        # ベクトル列バッファを初期化（各バッチで20個のベクトルを保持）
        self.vector_buffer = None

    def init_buffer(self, batch_size, device):
        # ベクトル列バッファを初期化
        self.vector_buffer = torch.zeros((batch_size, 0, 512), device=device)

    def forward(self, frames):
        """
        フレームごとに処理を実行し、20個のベクトルを管理しながら予測を行う。
        """
        batch_size, _, channels, height, width = frames.shape

        # 初回実行時にバッファを初期化
        if self.vector_buffer is None:
            self.init_buffer(batch_size, frames.device)

        # ステップ1: フレームをパッチに分割
        patches = self.patcher(frames[:, 0, :, :, :])  # (batch_size, 15, 9, 3, 128, 128)

        # ステップ2: 各パッチを特徴ベクトルに変換
        patch_features = self.img_to_vec(patches)  # (batch_size, 15, 9, 512)

        # ステップ3: 空間変換
        spatial_features = self.space_transformer(patch_features)  # (batch_size, 512)

        # ステップ4: ベクトル列に最新ベクトルを追加
        self.vector_buffer = torch.cat((self.vector_buffer, spatial_features.unsqueeze(1)), dim=1)  # (batch_size, n+1, 512)

        # ステップ5: ベクトル列が20個を超えたら最古のベクトルを削除
        if self.vector_buffer.size(1) > self.seq_len:
            self.vector_buffer = self.vector_buffer[:, 1:, :]  # 最古のベクトルを削除

        # ステップ6: ベクトル列が20個未満なら処理をスキップ
        if self.vector_buffer.size(1) < self.seq_len:
            print("[DEBUG] Vector buffer not full yet.")
            return None

        # ステップ7: 時系列モデルに渡して最終ベクトルを取得
        sequence_features = self.sequence_transformer(self.vector_buffer)  # (batch_size, 512)

        # ステップ8: 最終出力を計算
        prediction = self.output_layer(sequence_features)  # (batch_size, 3)
        return prediction


def simulate_frame_processing(batch_size=1, seq_len=20, num_frames=25, frame_size=(1920, 1080)):
    """
    仮のランダムなフレームデータを生成し、モデルに入力して動作を確認する。
    """
    # モデルの初期化
    model = BallDetectionModel(seq_len=seq_len)
    model.eval()  # 推論モードに設定

    # 仮のフレームデータを逐次生成して処理
    for frame_idx in range(num_frames):
        # ランダムな1フレーム分のデータを生成
        frames = torch.randn(batch_size, 1, 3, *frame_size)
        with torch.no_grad():
            # モデルにフレームを入力
            prediction = model(frames)

        # 結果を出力
        if prediction is not None:
            print(f"Frame {frame_idx}: Prediction = {prediction.detach().numpy()}")
        else:
            print(f"Frame {frame_idx}: Buffer not full yet.")

# メイン関数として実行
if __name__ == "__main__":
    simulate_frame_processing()