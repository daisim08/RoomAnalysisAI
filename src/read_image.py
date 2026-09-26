import cv2

# Read the image
image = cv2.imread("dataset/test_images/room1.jpg")

# Check if the image exists
if image is None:
    print("❌ Error: Image not found!")
else:
    print("✅ Image loaded successfully!")

    # Print image size
    print("Image Shape:", image.shape)