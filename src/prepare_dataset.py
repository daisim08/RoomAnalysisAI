import os
import shutil
import random

# ============================================================
# ROOM ANALYSIS AI - DATASET PREPARATION
# Randomized Train / Validation / Test Split
# ============================================================

print("=" * 60)
print("ROOM ANALYSIS AI - DATASET PREPARATION")
print("=" * 60)

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

SOURCE_DIR = r"C:\Users\Daisi M\Downloads\BD3_original_dataset\train"

PROJECT_DIR = r"C:\Users\Daisi M\Desktop\RoomAnalysisAI"

DATASET_DIR = os.path.join(PROJECT_DIR, "dataset")

TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "validation")
TEST_DIR = os.path.join(DATASET_DIR, "test")

# ------------------------------------------------------------
# RANDOM SEED
# ------------------------------------------------------------

RANDOM_SEED = 42

random.seed(RANDOM_SEED)

# ------------------------------------------------------------
# CLASS MAPPING
# ------------------------------------------------------------

CLASS_MAPPING = {
    "crack": ["major_crack", "minor_crack"],
    "peeling_paint": ["peeling"],
    "dampness": ["stain"],
    "normal": ["plain"]
}

# ------------------------------------------------------------
# CHECK SOURCE DATASET
# ------------------------------------------------------------

print("\nSource dataset:")
print(SOURCE_DIR)

print("\nProject dataset:")
print(DATASET_DIR)

if not os.path.exists(SOURCE_DIR):
    print("\n❌ ERROR: Source dataset not found!")
    print("Please check the dataset path.")
    exit()

print("\n✅ Source dataset found!")

# ------------------------------------------------------------
# REMOVE OLD DATASET
# ------------------------------------------------------------

if os.path.exists(DATASET_DIR):

    print("\nRemoving old dataset...")

    shutil.rmtree(DATASET_DIR)

    print("✅ Old dataset removed.")

# ------------------------------------------------------------
# CREATE DIRECTORIES
# ------------------------------------------------------------

for class_name in CLASS_MAPPING.keys():

    os.makedirs(os.path.join(TRAIN_DIR, class_name), exist_ok=True)
    os.makedirs(os.path.join(VAL_DIR, class_name), exist_ok=True)
    os.makedirs(os.path.join(TEST_DIR, class_name), exist_ok=True)

# ------------------------------------------------------------
# IMAGE EXTENSIONS
# ------------------------------------------------------------

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)

# ------------------------------------------------------------
# PROCESS EACH CLASS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("PROCESSING DATASET")
print("=" * 60)

for class_name, source_folders in CLASS_MAPPING.items():

    print("\n" + "-" * 60)
    print(f"Processing class: {class_name}")
    print("-" * 60)

    all_images = []

    # --------------------------------------------------------
    # COLLECT IMAGES
    # --------------------------------------------------------

    for folder in source_folders:

        folder_path = os.path.join(SOURCE_DIR, folder)

        print(f"\nLooking in:")
        print(folder_path)

        if not os.path.exists(folder_path):

            print("⚠️ Folder not found. Skipping...")

            continue

        images = [
            os.path.join(folder_path, file)
            for file in os.listdir(folder_path)
            if file.lower().endswith(IMAGE_EXTENSIONS)
        ]

        print(f"Images found: {len(images)}")

        all_images.extend(images)

    # --------------------------------------------------------
    # REMOVE DUPLICATES
    # --------------------------------------------------------

    all_images = list(dict.fromkeys(all_images))

    total_images = len(all_images)

    print(f"\nTotal images collected: {total_images}")

    if total_images == 0:

        print("❌ No images found for this class!")

        continue

    # --------------------------------------------------------
    # RANDOM SHUFFLE
    # --------------------------------------------------------

    random.shuffle(all_images)

    # --------------------------------------------------------
    # TEST SPLIT
    # --------------------------------------------------------

    test_count = round(total_images * 0.20)

    test_images = all_images[:test_count]

    remaining_images = all_images[test_count:]

    # --------------------------------------------------------
    # TRAIN / VALIDATION SPLIT
    # --------------------------------------------------------

    validation_count = round(len(remaining_images) * 0.20)

    validation_images = remaining_images[:validation_count]

    training_images = remaining_images[validation_count:]

    # --------------------------------------------------------
    # COPY TRAINING IMAGES
    # --------------------------------------------------------

    print(f"\nCopying {len(training_images)} training images...")

    for index, image_path in enumerate(training_images):

        extension = os.path.splitext(image_path)[1]

        destination = os.path.join(
            TRAIN_DIR,
            class_name,
            f"{class_name}_train_{index:05d}{extension}"
        )

        shutil.copy2(image_path, destination)

    # --------------------------------------------------------
    # COPY VALIDATION IMAGES
    # --------------------------------------------------------

    print(f"Copying {len(validation_images)} validation images...")

    for index, image_path in enumerate(validation_images):

        extension = os.path.splitext(image_path)[1]

        destination = os.path.join(
            VAL_DIR,
            class_name,
            f"{class_name}_val_{index:05d}{extension}"
        )

        shutil.copy2(image_path, destination)

    # --------------------------------------------------------
    # COPY TEST IMAGES
    # --------------------------------------------------------

    print(f"Copying {len(test_images)} testing images...")

    for index, image_path in enumerate(test_images):

        extension = os.path.splitext(image_path)[1]

        destination = os.path.join(
            TEST_DIR,
            class_name,
            f"{class_name}_test_{index:05d}{extension}"
        )

        shutil.copy2(image_path, destination)

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("\nClass summary:")
    print(f"Total images     : {total_images}")
    print(f"Training images  : {len(training_images)}")
    print(f"Validation images: {len(validation_images)}")
    print(f"Testing images   : {len(test_images)}")

# ------------------------------------------------------------
# FINAL MESSAGE
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("✅ DATASET PREPARATION COMPLETED!")
print("=" * 60)

print("\nDataset location:")
print(DATASET_DIR)

print("\nClasses:")

for index, class_name in enumerate(CLASS_MAPPING.keys(), start=1):

    print(f"{index}. {class_name}")

print("\nDataset split:")
print("80% Training/Validation pool")
print("20% Testing")

print("\nTraining pool split:")
print("80% Training")
print("20% Validation")

print("\nRandom seed:")
print(RANDOM_SEED)

print("\n🎉 Dataset is ready for CNN training!")