import cv2

# Read the image
image = cv2.imread("dataset/test_images/room1.jpg")

# Check if the image was loaded
if image is None:
    print("❌ Error: Image not found!")
else:
    print("✅ Image loaded successfully!")

    # Display the image
    cv2.imshow("Room Image", image)

    # Wait until a key is pressed
    cv2.waitKey(0)

    # Close the image window
    cv2.destroyAllWindows()