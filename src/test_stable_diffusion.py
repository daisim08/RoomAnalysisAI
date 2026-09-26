import torch
from diffusers import StableDiffusionPipeline

print("=" * 60)
print("STABLE DIFFUSION TEST")
print("=" * 60)

print("\nPyTorch version:", torch.__version__)

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)

if device == "cpu":
    print("\n⚠️ GPU is not available.")
    print("Stable Diffusion will run on CPU and may be slow.")

print("\nStable Diffusion library imported successfully!")
print("Environment is ready for the renovation module.")

print("=" * 60)