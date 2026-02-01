# import os
# import csv
# import random
# import itertools
# # here itertools use for effcient pair genration


# def csv_file_generator_single_images(root_dir, output_csv):
#     """
#     Generate a CSV file containing SINGLE images and their class labels.

#     Folder structure:
#         root_dir/
#             ├── class_001/
#             │     ├── img1.png
#             │     ├── img2.png
#             ├── class_002/
#             │     ├── img1.png

#     Output CSV format:
#         image_path, class_label
#     """

#     image_entries = []
#     # here in image entry store csv row and letter we will dump in csv file
#     total_images = 0

#     for class_name in os.listdir(root_dir):
#         class_path = os.path.join(root_dir, class_name)

#         if not os.path.isdir(class_path):
#             continue

#         for img in os.listdir(class_path):
#             if img.lower().endswith((".png", ".jpg", ".bmp")):
#                 img_path = os.path.join(class_path, img)
#                 image_entries.append((img_path, class_name))
#                 total_images += 1
#                 # here pair gerate within and put in image entry

#     with open(output_csv, "w", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow(["image_path", "class_label"])
#         writer.writerows(image_entries)
#         # put in csv file

#     print("=" * 60)
#     print("SINGLE IMAGE CSV CREATED")
#     print("=" * 60)
#     print(f"Total images : {total_images}")
#     print(f"Total classes: {len(os.listdir(root_dir))}")
#     print(f"Saved to     : {output_csv}")


# def csv_file_generator_with_classes(
#     root_dir,
#     output_csv,
#     num_images_per_folder_for_negatives=2
# ):
#     """
#     Generate a CSV file containing GENUINE and IMPOSTER image pairs.

#     Genuine pair  : two images from SAME class
#     Imposter pair : two images from DIFFERENT classes

#     Output CSV format:
#         image_path1, class1, image_path2, class2, label

#         label = 1 → genuine
#         label = 0 → imposter
#     """

#     pairs = []
#     positive_count = 0
#     negative_count = 0


#     # --------------------------------------------------
#     # Collect images per class
# #     # {
# #   "001": [img1, img2],
# #   "002": [img1, img2]
# #     }

#     # --------------------------------------------------
#     class_to_images = {}

#     for class_name in os.listdir(root_dir):
#         class_path = os.path.join(root_dir, class_name)

#         if not os.path.isdir(class_path):
#             continue

#         images = [
#             os.path.join(class_path, img)
#             for img in os.listdir(class_path)
#             if img.lower().endswith((".png", ".jpg", ".bmp"))
#         ]

#         if len(images) >= 2:
#             class_to_images[class_name] = images

#     # --------------------------------------------------
#     # Generate GENUINE pairs (same class)
#     # --------------------------------------------------
#     for class_name, images in class_to_images.items():
#         for img1, img2 in itertools.combinations(images, 2):
#             pairs.append((img1, class_name, img2, class_name, 1))
#             positive_count += 1
#             # it create unorder pair and take this pair and it is positve store inside 

#     # --------------------------------------------------
#     # Generate IMPOSTER pairs (different classes)
#     # --------------------------------------------------
#     class_list = list(class_to_images.keys())

#     for i in range(len(class_list)):
#         class1 = class_list[i]
#         images1 = class_to_images[class1]
#         random.shuffle(images1)

#         for j in range(i + 1, len(class_list)):
#             class2 = class_list[j]
#             images2 = class_to_images[class2]
#             random.shuffle(images2)

#             # here shuffling so that bais can be metigate

#             max_pairs = min(
#                 num_images_per_folder_for_negatives,
#                 len(images1),
#                 len(images2)
#             )
#             # here writing like this so that no need to quadratic expolosion

#             for k in range(max_pairs):
#                 pairs.append(
#                     (images1[k], class1, images2[k], class2, 0)
#                 )
#                 negative_count += 1

#                 #

#     # --------------------------------------------------
#     # Write CSV
#     # --------------------------------------------------
#     with open(output_csv, "w", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow(
#             ["image_path1", "class1", "image_path2", "class2", "label"]
#         )
#         writer.writerows(pairs)

#     print("=" * 60)
#     print("PAIR CSV CREATED")
#     print("=" * 60)
#     print(f"Genuine pairs : {positive_count}")
#     print(f"Imposter pairs: {negative_count}")
#     print(f"Total pairs   : {len(pairs)}")
#     print(f"Saved to      : {output_csv}")





# # csv_file_generator_with_classes(
# #     root_dir="/path/to/iris_dataset",
# #     output_csv="iris_pairs.csv",
# #     num_images_per_folder_for_negatives=2
# # )
# # how to call this file from some other functon i will call like this





# if __name__ == "__main__":
#     """
#     Simple test entry point to verify CSV generation.

#     Usage:
#         python pair_generator.py <dataset_root> <output_csv> [mode]

#     mode:
#         single  → generate single-image CSV
#         pairs   → generate genuine/imposter pairs (default)
#     """

#     import sys

#     if len(sys.argv) < 3:
#         print("=" * 60)
#         print("CSV GENERATION TEST")
#         print("=" * 60)
#         print("\nUsage:")
#         print("  python pair_generator.py <dataset_root> <output_csv> [mode]")
#         print("\nModes:")
#         print("  single  → single image CSV")
#         print("  pairs   → genuine/imposter pairs (default)")
#         print("\nExamples:")
#         print("  python pair_generator.py /data/IITD single.csv single")
#         print("  python pair_generator.py /data/IITD pairs.csv pairs")
#         print("=" * 60)
#         sys.exit(1)

#     dataset_root = sys.argv[1]
#     output_csv = sys.argv[2]
#     mode = sys.argv[3] if len(sys.argv) > 3 else "pairs"

#     if not os.path.isdir(dataset_root):
#         print(f"\n❌ Error: Dataset path does not exist: {dataset_root}")
#         sys.exit(1)

#     if mode == "single":
#         csv_file_generator_single_images(dataset_root, output_csv)

#     elif mode == "pairs":
#         csv_file_generator_with_classes(
#             root_dir=dataset_root,
#             output_csv=output_csv,
#             num_images_per_folder_for_negatives=2
#         )

#     else:
#         print(f"\n❌ Unknown mode: {mode}")
#         print("Use 'single' or 'pairs'")



# # HERE MODE INDICATE
# # If mode ="single" then csv format produce will be
# '''

# image_path, class_label
# /data/iris/001/img1.png, 001
# /data/iris/001/img2.png, 001



# '''
# # depend on how i am using this
# # and if mode will be pairs 
# # Create a CSV where each row is a pair of images with a label telling whether they match.”
# # image_path1, class1, image_path2, class2, label




import os
import csv
import random
import itertools


def csv_file_generator_single_images(root_dir, output_csv):
    """
    Generate a CSV file containing SINGLE images and their class labels.

    Folder structure:
        root_dir/
            ├── class_001/
            │     ├── img1.png
            │     ├── img2.png
            ├── class_002/
            │     ├── img1.png

    Output CSV format:
        image_path, class_label
    """

    image_entries = []
    total_images = 0

    for class_name in os.listdir(root_dir):
        class_path = os.path.join(root_dir, class_name)

        if not os.path.isdir(class_path):
            continue

        for img in os.listdir(class_path):
            if img.lower().endswith((".png", ".jpg", ".bmp")):
                img_path = os.path.join(class_path, img)
                image_entries.append((img_path, class_name))
                total_images += 1

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "class_label"])
        writer.writerows(image_entries)

    print("=" * 60)
    print("SINGLE IMAGE CSV CREATED")
    print("=" * 60)
    print(f"Total images : {total_images}")
    print(f"Total classes: {len(os.listdir(root_dir))}")
    print(f"Saved to     : {output_csv}")


def csv_file_generator_with_classes(
    root_dir,
    output_csv,
    num_images_per_folder_for_negatives=2
):
    """
    Generate a CSV file containing GENUINE and IMPOSTER image pairs.

    Genuine pair  : two images from SAME class
    Imposter pair : two images from DIFFERENT classes

    Output CSV format:
        image_path1, class1, image_path2, class2, label

        label = 1 → genuine
        label = 0 → imposter
    """

    pairs = []
    positive_count = 0
    negative_count = 0

    # --------------------------------------------------
    # Collect images per class
    # --------------------------------------------------
    class_to_images = {}

    for class_name in os.listdir(root_dir):
        class_path = os.path.join(root_dir, class_name)

        if not os.path.isdir(class_path):
            continue

        images = [
            os.path.join(class_path, img)
            for img in os.listdir(class_path)
            if img.lower().endswith((".png", ".jpg", ".bmp"))
        ]

        if len(images) >= 2:
            class_to_images[class_name] = images

    # --------------------------------------------------
    # Generate GENUINE pairs (same class)
    # --------------------------------------------------
    for class_name, images in class_to_images.items():
        for img1, img2 in itertools.combinations(images, 2):
            pairs.append((img1, class_name, img2, class_name, 1))
            positive_count += 1

    # --------------------------------------------------
    # Generate IMPOSTER pairs (different classes)
    # --------------------------------------------------
    class_list = list(class_to_images.keys())

    for i in range(len(class_list)):
        class1 = class_list[i]
        images1 = class_to_images[class1]
        random.shuffle(images1)

        for j in range(i + 1, len(class_list)):
            class2 = class_list[j]
            images2 = class_to_images[class2]
            random.shuffle(images2)

            max_pairs = min(
                num_images_per_folder_for_negatives,
                len(images1),
                len(images2)
            )

            for k in range(max_pairs):
                pairs.append(
                    (images1[k], class1, images2[k], class2, 0)
                )
                negative_count += 1

    # --------------------------------------------------
    # Write CSV
    # --------------------------------------------------
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["image_path1", "class1", "image_path2", "class2", "label"]
        )
        writer.writerows(pairs)

    print("=" * 60)
    print("PAIR CSV CREATED")
    print("=" * 60)
    print(f"Genuine pairs : {positive_count}")
    print(f"Imposter pairs: {negative_count}")
    print(f"Total pairs   : {len(pairs)}")
    print(f"Saved to      : {output_csv}")


if __name__ == "__main__":
    """
    Simple test entry point to verify CSV generation.

    Usage:
        python preprocess.py <dataset_root> <output_csv> [mode]

    mode:
        single  → generate single-image CSV
        pairs   → generate genuine/imposter pairs (default)
    """

    import sys

    if len(sys.argv) < 3:
        print("=" * 60)
        print("CSV GENERATION TEST")
        print("=" * 60)
        print("\nUsage:")
        print("  python preprocess.py <dataset_root> <output_csv> [mode]")
        print("\nModes:")
        print("  single  → single image CSV")
        print("  pairs   → genuine/imposter pairs (default)")
        print("\nExamples:")
        print("  python preprocess.py /data/IITD single.csv single")
        print("  python preprocess.py /data/IITD pairs.csv pairs")
        print("=" * 60)
        sys.exit(1)

    dataset_root = sys.argv[1]
    output_csv = sys.argv[2]
    mode = sys.argv[3] if len(sys.argv) > 3 else "pairs"

    if not os.path.isdir(dataset_root):
        print(f"\n❌ Error: Dataset path does not exist: {dataset_root}")
        sys.exit(1)

    if mode == "single":
        csv_file_generator_single_images(dataset_root, output_csv)

    elif mode == "pairs":
        csv_file_generator_with_classes(
            root_dir=dataset_root,
            output_csv=output_csv,
            num_images_per_folder_for_negatives=2
        )

    else:
        print(f"\n❌ Unknown mode: {mode}")
        print("Use 'single' or 'pairs'")