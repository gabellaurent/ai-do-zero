import torch
checkpoint = torch.load('slm_model.pth', map_location='cpu')
print(f"Iter: {checkpoint.get('iter')}")
print(f"Loss: {checkpoint.get('loss')}")
print(f"Vocab size: {len(checkpoint.get('chars'))}")
