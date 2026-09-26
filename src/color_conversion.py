import cv2

# Read image
image = cv2.imread("dataset/test_images/room1.jpg")

if image is None:
    print("❌ Image not found!")
else:
    print("✅ Image loaded!")

    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print("Original Shape:", image.shape)
    print("RGB Shape:", rgb_image.shape)

    # Display original image
    cv2.imshow("Original (BGR)", image)

    # Display converted image
    cv2.imshow("Converted (RGB)", rgb_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()