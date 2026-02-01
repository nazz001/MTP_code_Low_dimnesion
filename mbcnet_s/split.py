import os
import random
import shutil

def make_train_test_split(
    source_dir="/home/nishkal/datasets/iris_db/CASIA_v1/worldcoin_outputs_images/normalized",
    output_dir="dataset",
    train_ratio=0.7,
    seed=42,
    img_exts=(".png", ".jpg", ".bmp")
):
    """
    Create train/recog split from identity folders

    source_dir/
        ├── 001/
        │   ├── img1.png
        │   ├── img2.png
        ├── 002/
        │   ├── img1.png

    output_dir/
        ├── train/
        └── recog/
    """

    random.seed(seed)

    train_dir = os.path.join(output_dir, "train")
    test_dir  = os.path.join(output_dir, "recog")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    identities = sorted(os.listdir(source_dir))

    for identity in identities:
        id_path = os.path.join(source_dir, identity)
        if not os.path.isdir(id_path):
            continue

        images = [
            img for img in os.listdir(id_path)
            if img.lower().endswith(img_exts)
        ]

        if len(images) < 2:
            continue  # skip weak identities

        random.shuffle(images)

        split_idx = int(len(images) * train_ratio)
        train_imgs = images[:split_idx]
        test_imgs  = images[split_idx:]

        # Ensure at least 1 image in test
        if len(test_imgs) == 0:
            test_imgs.append(train_imgs.pop())

        # Create identity folders
        os.makedirs(os.path.join(train_dir, identity), exist_ok=True)
        os.makedirs(os.path.join(test_dir, identity), exist_ok=True)

        # Copy files
        for img in train_imgs:
            shutil.copy(
                os.path.join(id_path, img),
                os.path.join(train_dir, identity, img)
            )

        for img in test_imgs:
            shutil.copy(
                os.path.join(id_path, img),
                os.path.join(test_dir, identity, img)
            )

        print(f"✓ ID {identity}: train={len(train_imgs)}, test={len(test_imgs)}")

    print("\n=== TRAIN / TEST SPLIT COMPLETE ===")
    print(f"Train dir: {train_dir}")
    print(f"Test  dir: {test_dir}")

make_train_test_split()