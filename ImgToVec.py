import torch
import torch.nn as nn
import torchvision.models as models

class ImgToVec(nn.Module):
    def __init__(self):
        super(ImgToVec, self).__init__()
        # MobileNet V2を事前学習済みモデルとしてロード
        self.mobilenet = models.mobilenet_v2(pretrained=True)
        # 特徴抽出のために最終分類層を無効化
        self.mobilenet.classifier = nn.Identity()

    def forward(self, patches: torch.Tensor):
        # patchesの形状は (batch_size, num_patches_h, num_patches_w, 3, 128, 128)
        batch_size, num_patches_h, num_patches_w, _, patch_h, patch_w = patches.shape

        # MobileNetに渡すためにテンソルをフラット化
        patches = patches.view(-1, 3, patch_h, patch_w)  # (batch_size * num_patches_h * num_patches_w, 3, 128, 128)

        # 各パッチをMobileNetに通して特徴量を取得
        features = self.mobilenet(patches)  # 出力の形状: (batch_size * num_patches_h * num_patches_w, feature_dim)

        # 元の構造に戻す
        feature_dim = features.shape[1]  # MobileNetの出力次元
        features = features.view(batch_size, num_patches_h, num_patches_w, feature_dim)  # (batch_size, num_patches_h, num_patches_w, feature_dim)
        return features

if __name__ == '__main__':
    img_to_vec = ImgToVec()
    # テスト用の例
    for i in range(25):
        patches = torch.randn(1, 15, 9, 3, 128, 128)  # (batch_size, num_patches_h, num_patches_w, 3, 128, 128)
        patches_features = img_to_vec(patches)  # 出力の形状は (batch_size, num_patches_h, num_patches_w, feature_dim)
        print(patches_features.shape)
