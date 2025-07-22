import torch
from resnet.blocks import BasicBlock, Bottleneck

# test if the BasicBlock preserves input shape (no downsampling, used in ResNet18)
def test_basic_block_shape():
    x = torch.randn(2, 64, 56, 56)
    block = BasicBlock(64, 64)
    y = block(x)
    assert y.shape == x.shape

# test if Bottleneck block expands channels correctly with downsample (used in ResNet50+)
def test_bottleneck_block_shape():
    x = torch.randn(2, 64, 56, 56)

    downsample = torch.nn.Sequential(
        torch.nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False),
        torch.nn.BatchNorm2d(256)
    )

    block = Bottleneck(64, 64, downsample=downsample)
    y = block(x)
    assert y.shape == (2, 256, 56, 56)