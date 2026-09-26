import requests
import os

image_folder = r"C:\Users\Daisi M\Desktop\RoomAnalysisAI\dataset\test\crack"

image_files = [
    f for f in os.listdir(image_folder)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

if not image_files:
    print("No test image found.")
    exit()

image_path = os.path.join(
    image_folder,
    image_files[0]
)

print("Testing image:")
print(image_path)

with open(image_path, "rb") as image:
    response = requests.post(
        "http://127.0.0.1:5000/predict",
        files={
            "image": (
                os.path.basename(image_path),
                image,
                "image/jpeg"
            )
        },
        timeout=120
    )

print("\nAPI Status:", response.status_code)
print("\nAPI Response:")
print(response.text)