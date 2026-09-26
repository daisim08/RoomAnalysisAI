import os
import shutil
import random


BASE_DIR = r"C:\Users\Daisi M\Desktop\RoomAnalysisAI\dataset"

TRAIN_DIR = os.path.join(
    BASE_DIR,
    "train"
)

VALIDATION_DIR = os.path.join(
    BASE_DIR,
    "validation"
)


VALIDATION_RATIO = 0.20

random.seed(42)


CLASSES = [
    "crack",
    "dampness",
    "normal",
    "peeling_paint"
]


print("=" * 60)
print("CREATING VALIDATION DATASET")
print("=" * 60)


for class_name in CLASSES:

    source_folder = os.path.join(
        TRAIN_DIR,
        class_name
    )

    validation_folder = os.path.join(
        VALIDATION_DIR,
        class_name
    )


    os.makedirs(
        validation_folder,
        exist_ok=True
    )


    images = [
        file
        for file in os.listdir(source_folder)
        if file.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]


    random.shuffle(images)


    validation_count = int(
        len(images) * VALIDATION_RATIO
    )


    validation_images = images[
        :validation_count
    ]


    for image in validation_images:

        source_path = os.path.join(
            source_folder,
            image
        )

        destination_path = os.path.join(
            validation_folder,
            image
        )

        shutil.move(
            source_path,
            destination_path
        )


    print(
        f"{class_name:15} : "
        f"{len(validation_images)} moved to validation"
    )


print("\n" + "=" * 60)

print(
    "✅ VALIDATION DATASET CREATED"
)

print("=" * 60)