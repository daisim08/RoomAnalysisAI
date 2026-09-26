import os

BASE_DIR = r"C:\Users\Daisi M\Desktop\RoomAnalysisAI\dataset"

classes = [
    "crack",
    "dampness",
    "normal",
    "peeling_paint"
]

print("=" * 60)
print("DATASET CHECK")
print("=" * 60)

for split in ["train", "test"]:

    print(f"\n{split.upper()} DATASET")
    print("-" * 40)

    split_total = 0

    for class_name in classes:

        folder = os.path.join(
            BASE_DIR,
            split,
            class_name
        )

        images = [
            file
            for file in os.listdir(folder)
            if file.lower().endswith(
                (".jpg", ".jpeg", ".png")
            )
        ]

        count = len(images)
        split_total += count

        print(
            f"{class_name:15} : {count}"
        )

    print("-" * 40)
    print(f"TOTAL           : {split_total}")

print("\n" + "=" * 60)
print("✅ DATASET CHECK COMPLETED")
print("=" * 60)