import torch, platform
print("Device:", "cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("GPUs:", torch.cuda.device_count())
    print("GPU name:", torch.cuda.get_device_name(0))
print("Python:", platform.python_version())
print("Torch:", torch.__version__)