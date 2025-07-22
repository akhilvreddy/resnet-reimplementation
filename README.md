**Please take a look at [this notebook](https://github.com/akhilvreddy/resnet-reimplementation/blob/main/notebook/resnet_demo.ipynb)** for the working model.

--- 

# ResNet reimplemented

A reimplementation of the ResNet architecture using PyTorch's low-level API. In this project, I reproduced the key ideas behind the original [ResNet paper (2015)](https://arxiv.org/abs/1512.03385), including

- BasicBlock & Bottleneck residual connections
- modular `_make_layer()` block construction
- multiple resnet architectures: ResNet-18, ResNet-34, and ResNet-50
- Weight initialization with He init
- Training on CIFAR-10

---

## Tests

All tests pass via `pytest` and the test suite checks for:

- Correct forward pass for both `BasicBlock` and `Bottleneck` modules
- Functional ResNet model end-to-end on dummy inputs
- Output shape consistency and residual connection correctness
- Compatibility with various depths (ResNet-18/34/50)

---