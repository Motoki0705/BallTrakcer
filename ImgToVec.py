import torch
import torch.nn as nn
import torchvision.models as models

class ImgToVec(nn.Module):
    def __init__(self):
        super(ImgToVec, self).__init__()
        # ResNet18をベースモデルとして利用
        self.resnet = models.resnet18(weights=None)
        self.resnet.fc = nn.Identity()  # 最終分類層を無効化して特徴ベクトルを取得

    def forward(self, patches: torch.Tensor):
        # patchesの形状は (batch_size, num_patches_h, num_patches_w, 3, 128, 128)
        batch_size, num_patches_h, num_patches_w, _, patch_h, patch_w = patches.shape

        # ResNetに渡すためにテンソルをフラット化
        patches = patches.view(-1, 3, patch_h, patch_w)  # (batch_size * num_patches_h * num_patches_w, 3, 128, 128)

        # 各パッチをResNetに通して特徴量を取得
        features = self.resnet(patches)  # 出力の形状: (batch_size * num_patches_h * num_patches_w, feature_dim)

        # 元の構造に戻す
        feature_dim = features.shape[1]  # ResNetの出力次元
        features = features.view(batch_size, num_patches_h, num_patches_w, feature_dim)  # (batch_size, num_patches_h, num_patches_w, feature_dim)
        return features

if __name__ == '__main__':
    # テスト用の例
    patches = torch.randn(3, 15, 9, 3, 128, 128)  # (batch_size, num_patches_h, num_patches_w, 3, 128, 128)

    img_to_vec = ImgToVec()
    patches_features = img_to_vec(patches)  # 出力の形状は (batch_size, num_patches_h, num_patches_w, feature_dim)
    print(patches_features.shape)
