import requests
import json
from pathlib import Path

API_URL = "http://127.0.0.1:5000/renovation/generate"

IMAGE_PATH = Path(
    r"C:\Users\Daisi M\Desktop\RoomAnalysisAI\dataset\test\crack\crack_test_00000.jpg"
)

print("=" * 60)
print("ROOM ANALYSIS AI - RENOVATION TEST")
print("=" * 60)

print("\nTest image:")
print(IMAGE_PATH)

if not IMAGE_PATH.exists():
    print("\nERROR: Test image not found.")
    raise SystemExit(1)

print("\nSending image to local Flask backend...")
print("Cloudflare FLUX.2 Klein 4B will generate the renovation.")
print("Please wait...\n")

try:
    with open(IMAGE_PATH, "rb") as image_file:
        files = {
            "image": (
                IMAGE_PATH.name,
                image_file,
                "image/jpeg"
            )
        }

        data = {
            "style": "minimal",
            "furniture": "true",
            "budget": "50000"
        }

        response = requests.post(
            API_URL,
            files=files,
            data=data,
            timeout=300
        )

    print("HTTP Status:", response.status_code)

    print("\nAPI Response:")

    try:
        result = response.json()
        print(json.dumps(result, indent=2))
    except Exception:
        print(response.text)
        raise SystemExit(1)

    if response.status_code != 200:
        print("\nRenovation request failed.")
        raise SystemExit(1)

    if not result.get("success"):
        print("\nRenovation was not successful.")
        raise SystemExit(1)

    print("\n" + "=" * 60)
    print("SUCCESS!")
    print("=" * 60)

    print("\nRenovated image URL:")
    print(result.get("image_url"))

    print("\nCondition:")
    print(result.get("condition"))

    print("\nStyle:")
    print(result.get("style"))

    print("\nBudget:")
    print(result.get("budget"))

    print("\nThe Cloudflare renovation test completed successfully.")

except requests.exceptions.Timeout:
    print("\nERROR: Request timed out.")
    print("Cloudflare may still be processing the image.")

except requests.exceptions.ConnectionError:
    print("\nERROR: Could not connect to Flask.")
    print("Make sure main.py is still running.")

except Exception as e:
    print("\nERROR:")
    print(str(e))
