import cv2

# Read image
image = cv2.imread("dataset/test_images/room1.jpg")

if image is None:
    print("❌ Image not found!")
else:
    print("Original Data Type:", image.dtype)
    print("Original Pixel Range:", image.min(), "to", image.max())

    # Convert to float and normalize
    normalized_image = image.astype("float32") / 255.0

    print("\nAfter Normalization")
    print("Data Type:", normalized_image.dtype)
    print("Pixel Range:", normalized_image.min(), "to", normalized_image.max())
    