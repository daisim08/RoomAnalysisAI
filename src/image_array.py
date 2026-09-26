import numpy as np

# Create a small 3 × 3 image
image = np.array([
    [255, 255, 255],
    [255,   0, 255],
    [255, 255, 255]
])

print("Image represented as numbers:")
print(image)

print("Shape of the image:", image.shape)