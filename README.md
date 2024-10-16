# VisionDiff

Using the [Differential Transformer](https://arxiv.org/abs/2410.05258) in a vision-friendly way, similar to [VisionMamba](https://github.com/kyegomez/VisionMamba).

```python
from block import VisionDiffAttn

layer1 = VisionDiffAttn(32, 64, in_channels=3)
layer2 = VisionDiffAttn(32, 64)
layer3 = VisionDiffAttn(32, 64, out_channels=3)

x = torch.zeros(1, 3, 64, 64)
x = layer1(x)
x = layer2(x)
x = layer3(x)

print(f"Output shape: {x.shape}") # [1, 3, 64, 64]
```