import torch
import torch.nn as nn
import torch.nn.functional as F

class VideoFramePatcher(nn.Module):
    def __init__(self, patch_size: tuple = (128, 128)):
        super(VideoFramePatcher, self).__init__()
        self.patch_size = patch_size

    def pad_to_patch_size(self, frame: torch.Tensor):
        _, _, height, width = frame.shape
        patch_height, patch_width = self.patch_size

        pad_height = (patch_height - (height % patch_height)) % patch_height
        pad_width = (patch_width - (width % patch_width)) % patch_width

        padding = (0, pad_width, 0, pad_height)  # (left, right, top, bottom)
        padded_frame = F.pad(frame, padding, value=0)
        return padded_frame

    def split_into_patches(self, frame: torch.Tensor):
        padded_frame = self.pad_to_patch_size(frame)
        batch_size, channels, height, width = padded_frame.shape
        patch_height, patch_width = self.patch_size

        patches = padded_frame.unfold(2, patch_height, patch_height).unfold(3, patch_width, patch_width)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()  # (batch_size, num_patches_h, num_patches_w, channels, patch_height, patch_width)
        return patches

    def forward(self, frame: torch.Tensor):
        patches = self.split_into_patches(frame)
        return patches
    
if __name__ == '__main__':
    # Example usage
    frame = torch.randn(4, 3, 1920, 1128)  # A random batch of frames with shape (batch_size, 3, 1920, 1128)
    patcher = VideoFramePatcher(patch_size=(128, 128))
    patches = patcher(frame)  # Output shape will be (batch_size, 15, 9, 3, 128, 128)
    print(patches.shape)
