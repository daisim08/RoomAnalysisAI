import base64
import os
import requests

# ============================================================
# ROOM ANALYSIS AI - CLOUDFLARE FLUX.2 KLEIN 4B TEST
# ============================================================

ACCOUNT_ID = "357d5a78f65add6c733702ab12f19c09"

# IMPORTANT:
# Put your Cloudflare API token here.
# Do NOT send the token to ChatGPT.
API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN")

MODEL = "@cf/black-forest-labs/flux-2-klein-4b"

IMAGE_PATH = r"C:\Users\Daisi M\Desktop\RoomAnalysisAI\dataset\test\crack\crack_test_00000.jpg"

OUTPUT_PATH = r"C:\Users\Daisi M\Desktop\RoomAnalysisAI\cloudflare_renovated_room.jpg"

API_URL = (
    f"https://api.cloudflare.com/client/v4/accounts/"
    f"{ACCOUNT_ID}/ai/run/{MODEL}"
)

PROMPT = """
Photorealistic renovation of the SAME ROOM in the input image.
Preserve the exact room layout, walls, ceiling, doors, windows,
camera viewpoint and perspective.
Repair all visible wall cracks and damaged areas.
Create clean smooth repaired walls with fresh warm-white paint.
Add modern minimalist furniture: comfortable sofa, small coffee table,
light wood furniture, curtains, area rug, indoor plant and modern ceiling light.
Keep the room realistic, spacious and naturally lit.
Do not change the architecture or room structure.
"""

print("=" * 60)
print("ROOM ANALYSIS AI - CLOUDFLARE FLUX.2 KLEIN 4B TEST")
print("=" * 60)

print("\nModel:")
print(MODEL)

print("\nInput image:")
print(IMAGE_PATH)

if not os.path.exists(IMAGE_PATH):
    print("\nERROR: Input image not found.")
    print(IMAGE_PATH)
    raise SystemExit(1)

if API_TOKEN == "PASTE_YOUR_CLOUDFLARE_API_TOKEN_HERE":
    print("\nERROR: Cloudflare API token has not been added.")
    print("Open this file and replace the placeholder with your token.")
    raise SystemExit(1)

print("\nReading image...")

with open(IMAGE_PATH, "rb") as f:
    image_bytes = f.read()

print(f"Image size: {len(image_bytes):,} bytes")

# ============================================================
# MULTIPART REQUEST
# ============================================================

files = {
    "input_image": (
        os.path.basename(IMAGE_PATH),
        image_bytes,
        "image/jpeg"
    )
}

data = {
    "prompt": PROMPT,
    "width": "1024",
    "height": "576",
}

headers = {
    "Authorization": f"Bearer {API_TOKEN}"
}

print("\nSending room image to Cloudflare...")
print("FLUX.2 Klein 4B is generating the renovated room.")
print("Please wait...\n")

try:
    response = requests.post(
        API_URL,
        headers=headers,
        files=files,
        data=data,
        timeout=300
    )
except Exception as e:
    print("=" * 60)
    print("CONNECTION ERROR")
    print("=" * 60)
    print(str(e))
    raise SystemExit(1)

print(f"HTTP Status: {response.status_code}")

# ============================================================
# ERROR HANDLING
# ============================================================

if response.status_code != 200:
    print("\n" + "=" * 60)
    print("CLOUDFLARE API ERROR")
    print("=" * 60)

    try:
        print(response.json())
    except Exception:
        print(response.text)

    with open(
        r"C:\Users\Daisi M\Desktop\RoomAnalysisAI\cloudflare_response.json",
        "w",
        encoding="utf-8"
    ) as f:
        f.write(response.text)

    print("\nDebug response saved to:")
    print(r"C:\Users\Daisi M\Desktop\RoomAnalysisAI\cloudflare_response.json")

    raise SystemExit(1)

# ============================================================
# READ RESULT
# ============================================================

try:
    result_json = response.json()
except Exception:
    print("\nERROR: Cloudflare returned invalid JSON.")
    print(response.text)
    raise SystemExit(1)

if not result_json.get("success", False):
    print("\nCloudflare request was not successful.")
    print(result_json)
    raise SystemExit(1)

result = result_json.get("result", {})
image_base64 = result.get("image")

if not image_base64:
    print("\nERROR: No generated image was returned.")
    print(result_json)
    raise SystemExit(1)

# ============================================================
# SAVE GENERATED IMAGE
# ============================================================

try:
    generated_bytes = base64.b64decode(image_base64)

    with open(OUTPUT_PATH, "wb") as f:
        f.write(generated_bytes)

except Exception as e:
    print("\nERROR while decoding/saving image:")
    print(str(e))
    raise SystemExit(1)

print("\n" + "=" * 60)
print("SUCCESS!")
print("=" * 60)

print("\nRenovated image saved to:")
print(OUTPUT_PATH)

print(f"\nGenerated image size: {len(generated_bytes):,} bytes")

print("\nCloudflare FLUX.2 Klein 4B test completed.")
print("=" * 60)