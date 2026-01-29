"""Feature extraction script"""
import sys
import os
from extraction.extractor import IrisFeatureExtractor


def main():
    if len(sys.argv) < 2:
        print("=" * 60)
        print("FEATURE EXTRACTION FROM DATASET")
        print("=" * 60)
        print("\nUsage:")
        print("  python extract.py <dataset_path> [output_name]")
        print("\nExamples:")
        print("  python extract.py /path/to/CASIA_v1")
        print("  python extract.py /path/to/CASIA_v1 casia_features")
        print("  python extract.py /path/to/IITD iitd_features")
        print("\nDataset Structure:")
        print("  dataset_path/")
        print("    ├── class1/")
        print("    │   ├── image1.jpg")
        print("    │   └── image2.jpg")
        print("    ├── class2/")
        print("    │   └── image3.jpg")
        print("    └── ...")
        return

    dataset_path = sys.argv[1]
    output_name = sys.argv[2] if len(sys.argv) > 2 else os.path.basename(dataset_path.rstrip('/'))

    print("=" * 60)
    print("FEATURE EXTRACTION")
    print("=" * 60)
    print(f"\nDataset: {dataset_path}")
    print(f"Output name: {output_name}")

    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"\n❌ Error: Dataset path does not exist: {dataset_path}")
        return

    if not os.path.isdir(dataset_path):
        print(f"\n❌ Error: Path is not a directory: {dataset_path}")
        return

    # Initialize extractor
    print("\nInitializing feature extractor...")
    try:
        extractor = IrisFeatureExtractor()
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have trained the model first by running:")
        print("  python train.py")
        return

    # Extract features
    output_path = f"outputs_simple/features_{output_name}.pkl"

    print("\nExtracting features...")
    features = extractor.extract_from_dataset(dataset_path, save_output=output_path)

    # Summary
    num_classes = len(features)
    total_images = sum(len(v) for v in features.values())

    print("\n" + "=" * 60)
    print("✅ EXTRACTION COMPLETE!")
    print("=" * 60)
    print(f"Classes: {num_classes}")
    print(f"Total images: {total_images}")
    print(f"Features saved to: {output_path}")
    print("\nYou can load these features using:")
    print(f"  from extraction import IrisFeatureExtractor")
    print(f"  extractor = IrisFeatureExtractor()")
    print(f"  features = extractor.load_features('{output_path}')")


if __name__ == "__main__":
    main()