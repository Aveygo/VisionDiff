# VisionDiff

Using the [Differential Transformer](https://arxiv.org/abs/2410.05258) in a vision-friendly way, similar to [VisionMamba](https://github.com/kyegomez/VisionMamba).

```python
layer1 = VisionDiffAttn(dim=32, in_channels=3)
layer2 = VisionDiffAttn(dim=32)
layer3 = VisionDiffAttn(dim=32, out_channels=3)

x = torch.zeros(1, 3, 64, 64) # Example image
x = layer1(x)
x = layer2(x)
x = layer3(x)

print(f"Output shape: {x.shape}") # [1, 3, 64, 64]
```