import torch
from resnet.resnet import resnet18, resnet50

# test if ResNet18 outputs correct shape for 10-class classification
def test_resnet18_output_shape():
    model = resnet18(num_classes=10)
    x = torch.randn(4, 3, 224, 224)
    y = model(x)
    assert y.shape == (4, 10)

# test if ResNet50 outputs correct shape for 100-class classification
def test_resnet50_output_shape():
    model = resnet50(num_classes=100)
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    assert y.shape == (1, 100)