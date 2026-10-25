import torch
import timm
import os
from collections import OrderedDict

# === PATHS ===
pth_path = os.environ.get("DMS_CHECKPOINT", "models/mobilenetv4.pth")
pt_path = os.path.splitext(pth_path)[0] + "_scripted.pt"

# === MODEL NAME ===
# Make sure this matches your checkpoint architecture
MODEL_NAME = "mobilenetv4_conv_small.e2400_r224_in1k"

print(f"🔍 Loading model architecture: {MODEL_NAME}")
model = timm.create_model(MODEL_NAME, pretrained=False)
model.eval()

# === LOAD CHECKPOINT ===
print(f"📦 Loading checkpoint: {pth_path}")
checkpoint = torch.load(pth_path, map_location="cpu")

# Handle both raw state_dicts and wrapped checkpoints
if isinstance(checkpoint, OrderedDict):
    state_dict = checkpoint
elif "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
else:
    state_dict = checkpoint

# Clean up keys if wrapped (remove "module." prefixes)
new_state_dict = {}
for k, v in state_dict.items():
    new_k = k.replace("module.", "")
    new_state_dict[new_k] = v

# === LOAD WEIGHTS ===
missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
if missing:
    print(f"⚠️ Missing keys: {missing}")
if unexpected:
    print(f"⚠️ Unexpected keys: {unexpected}")

# === CONVERT TO TORCHSCRIPT ===
example_input = torch.randn(1, 3, 224, 224)
scripted_model = torch.jit.trace(model, example_input)

# === SAVE ===
torch.jit.save(scripted_model, pt_path)
print(f"✅ TorchScript model saved successfully at: {pt_path}")
