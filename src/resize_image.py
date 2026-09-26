import cv2

# Read the image
image = cv2.imread("dataset/test_images/room1.jpg")

if image is None:
    print("❌ Image not found!")
else:
    print("✅ Original Shape:", image.shape)

    # Resize the image
    resized_image = cv2.resize(image, (224, 224))

    print("✅ Resized Shape:", resized_image.shape)

    # Display resized image
    cv2.imshow("Resized Room Image", resized_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()