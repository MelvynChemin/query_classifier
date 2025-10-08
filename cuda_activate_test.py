import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version (bundled):", torch.version.cuda)
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0)) 