# debug_clip.py
import clip
import torch
from PIL import Image

print("🔄 Loading CLIP...")
device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
print("✅ CLIP loaded!")

# Test image
print("🔄 Testing with random tensor...")
test_input = torch.randn(1, 3, 224, 224).to(device)
with torch.no_grad():
    features = model.encode_image(test_input)
print(f"✅ Test passed! Output shape: {features.shape}")