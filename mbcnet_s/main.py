# # import os
# # import torch
# # import pandas as pd
# # from torch.utils.data import DataLoader
# # from torchvision import transforms
# # from torch.optim import Adam
# # from torch.optim.lr_scheduler import MultiStepLR

# # import config
# # from preprocess import csv_file_generator_with_classes
# # from utils import (
# #     split_dataset,
# #     convert_class_labels,
# #     generate_equal_label_dataframes,
# #     save_metrics_to_csv
# # )
# # from dataset import IrisPairDataset
# # from model import MBLNet, IrisSiameseVerificationModel
# # # from loss import IrisSiameseLoss
# # from loss import IrisSiameseLoss

# # from train import train_model


# # def main():

# #     # ============================================================
# #     # STEP 1: CSV GENERATION
# #     # ============================================================

# #     if not os.path.exists(config.CSV_FILE_Train):
# #         csv_file_generator_with_classes(
# #             config.IMAGE_DIR_Train,
# #             config.CSV_FILE_Train
# #         )
# #         convert_class_labels(config.CSV_FILE_Train)
# #     else:
# #         print("✓ Training CSV already exists")

# #     if not os.path.exists(config.CSV_FILE_Recog):
# #         csv_file_generator_with_classes(
# #             config.IMAGE_DIR_Test,
# #             config.CSV_FILE_Recog,
# #             num_images_per_folder_for_negatives=2
# #         )
# #         convert_class_labels(config.CSV_FILE_Recog)
# #     else:
# #         print("✓ Testing CSV already exists")

# #     # ============================================================
# #     # STEP 2: TRANSFORMS
# #     # ============================================================
# #     print("gd")

# #     transform = transforms.Compose([
# #         transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
# #         transforms.ToTensor()
# #     ])
# #     print("hey")

# #     # ============================================================
# #     # STEP 3: VALIDATION SET (CROSS-SESSION)
# #     # ============================================================

# #     val_df = pd.read_csv(config.CSV_FILE_Recog)
# #     val_dataset = IrisPairDataset(val_df, transform)
# #     val_loader = DataLoader(
# #         val_dataset,
# #         batch_size=config.BATCH_SIZE,
# #         shuffle=False,
# #         num_workers=4
# #     )
# #     print("hey")

# #     # ============================================================
# #     # STEP 4: BALANCED TRAINING SETS
# #     # ============================================================

# #     dataframe_list, num_classes = generate_equal_label_dataframes(
# #         config.CSV_FILE_Train
# #     )

# #     print(f"✓ Balanced splits: {len(dataframe_list)}")
# #     print(f"✓ Number of identities: {num_classes}")

# #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #     print(f"✓ Using device: {device}")

# #     # ============================================================
# #     # STEP 5: MODEL, LOSS, OPTIMIZER
# #     # ============================================================

# #     print("hey")
# #     backbone = MBLNet(
# #         in_channels=1,
# #         feature_dim=config.FEATURE_DIM,
# #         attention_type=config.ATTENTION_TYPE
# #     )

# #     model = IrisSiameseVerificationModel(backbone).to(device)

# #     criterion = IrisSiameseLoss()
# #     optimizer = Adam(
# #         model.parameters(),
# #         lr=config.LEARNING_RATE,
# #         weight_decay=config.WEIGHT_DECAY
# #     )

# #     scheduler = MultiStepLR(
# #         optimizer,
# #         milestones=[30, 70, 150],
# #         gamma=0.1
# #     )

# #     best_model_path = None
# #     best_test_acc = 0.0

# #     # ============================================================
# #     # STEP 6: ITERATIVE TRAINING
# #     # ============================================================
# #     print("hey")

# #     for idx, df in enumerate(dataframe_list):
# #         print(f"\n=== Training Split {idx + 1}/{len(dataframe_list)} ===")

# #         train_df, test_df = split_dataset(df)

# #         train_dataset = IrisPairDataset(train_df, transform)
# #         test_dataset = IrisPairDataset(test_df, transform)

# #         train_loader = DataLoader(
# #             train_dataset,
# #             batch_size=config.BATCH_SIZE,
# #             shuffle=True,
# #             num_workers=4
# #         )

# #         test_loader = DataLoader(
# #             test_dataset,
# #             batch_size=config.BATCH_SIZE,
# #             shuffle=False,
# #             num_workers=4
# #         )

# #         save_path = os.path.join(config.SAVE_DIR, f"iteration_{idx + 1}")
# #         os.makedirs(save_path, exist_ok=True)

# #         if best_model_path:
# #             model.load_state_dict(torch.load(best_model_path))

# #         best_model_path, train_losses, val_losses, train_accs, val_accs, best_test_acc = train_model(
# #             model,
# #             train_loader,
# #             val_loader,
# #             test_loader,
# #             optimizer,
# #             criterion,
# #             device,
# #             scheduler,
# #             save_path,
# #             best_test_acc
# #         )

# #         save_metrics_to_csv(
# #             train_losses,
# #             val_losses,
# #             train_accs,
# #             val_accs,
# #             save_path
# #         )

# #         print(f"✓ Best model saved at: {best_model_path}")


# # if __name__ == "__main__":
# #     main()



# import os
# import torch
# import pandas as pd
# from torch.utils.data import DataLoader
# from torchvision import transforms
# from torch.optim import Adam
# from torch.optim.lr_scheduler import MultiStepLR
# from tqdm import tqdm   # ✅ tqdm integrated here

# import config
# from preprocess import csv_file_generator_with_classes
# from utils import (
#     split_dataset,
#     convert_class_labels,
#     generate_equal_label_dataframes,
#     save_metrics_to_csv
# )
# from dataset import IrisPairDataset
# from model import MBLNet, IrisSiameseVerificationModel
# from loss import IrisSiameseLoss
# from train import train_model


# def main():

#     # ============================================================
#     # STEP 1: CSV GENERATION
#     # ============================================================

#     if not os.path.exists(config.CSV_FILE_Train):
#         csv_file_generator_with_classes(
#             config.IMAGE_DIR_Train,
#             config.CSV_FILE_Train
#         )
#         convert_class_labels(config.CSV_FILE_Train)
#     else:
#         print("✓ Training CSV already exists")

#     if not os.path.exists(config.CSV_FILE_Recog):
#         csv_file_generator_with_classes(
#             config.IMAGE_DIR_Test,
#             config.CSV_FILE_Recog,
#             num_images_per_folder_for_negatives=2
#         )
#         convert_class_labels(config.CSV_FILE_Recog)
#     else:
#         print("✓ Testing CSV already exists")

#     # ============================================================
#     # STEP 2: TRANSFORMS
#     # ============================================================

#     transform = transforms.Compose([
#         transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
#         transforms.ToTensor()
#     ])

#     # ============================================================
#     # STEP 3: VALIDATION SET (RECOGNITION)
#     # ============================================================

#     val_df = pd.read_csv(config.CSV_FILE_Recog)
#     val_dataset = IrisPairDataset(val_df, transform)

#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=config.BATCH_SIZE,
#         shuffle=False,
#         num_workers=config.NUM_WORKERS
#     )

#     # ============================================================
#     # STEP 4: BALANCED TRAINING SETS
#     # ============================================================

#     dataframe_list, num_classes = generate_equal_label_dataframes(
#         config.CSV_FILE_Train
#     )

#     print(f"✓ Balanced splits       : {len(dataframe_list)}")
#     print(f"✓ Number of identities  : {num_classes}")

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"✓ Using device          : {device}")

#     # ============================================================
#     # STEP 5: MODEL, LOSS, OPTIMIZER
#     # ============================================================

#     backbone = MBLNet(
#         in_channels=1,
#         feature_dim=config.FEATURE_DIM,
#         attention_type=config.ATTENTION_TYPE
#     )

#     model = IrisSiameseVerificationModel(backbone).to(device)

#     criterion = IrisSiameseLoss()

#     optimizer = Adam(
#         model.parameters(),
#         lr=config.LEARNING_RATE,
#         weight_decay=config.WEIGHT_DECAY
#     )

#     scheduler = MultiStepLR(
#         optimizer,
#         milestones=[30, 70, 150],  # OK even if epochs < 150
#         gamma=0.1
#     )

#     best_model_path = None
#     best_test_acc = 0.0

#     # ============================================================
#     # STEP 6: ITERATIVE TRAINING (tqdm here)
#     # ============================================================

#     for idx, df in enumerate(
#         tqdm(dataframe_list, desc="Training Splits", unit="split")
#     ):
#         print(f"\n=== Training Split {idx + 1}/{len(dataframe_list)} ===")

#         train_df, test_df = split_dataset(df)

#         train_dataset = IrisPairDataset(train_df, transform)
#         test_dataset  = IrisPairDataset(test_df, transform)

#         train_loader = DataLoader(
#             train_dataset,
#             batch_size=config.BATCH_SIZE,
#             shuffle=True,
#             num_workers=config.NUM_WORKERS
#         )

#         test_loader = DataLoader(
#             test_dataset,
#             batch_size=config.BATCH_SIZE,
#             shuffle=False,
#             num_workers=config.NUM_WORKERS
#         )

#         save_path = os.path.join(config.SAVE_DIR, f"iteration_{idx + 1}")
#         os.makedirs(save_path, exist_ok=True)

#         # Load previous best model (warm start)
#         if best_model_path and os.path.exists(best_model_path):
#             model.load_state_dict(torch.load(best_model_path))

#         best_model_path, train_losses, val_losses, train_accs, val_accs, best_test_acc = train_model(
#             model,
#             train_loader,
#             val_loader,
#             test_loader,
#             optimizer,
#             criterion,
#             device,
#             scheduler,
#             save_path,
#             # best_test_acc
#         )

#         save_metrics_to_csv(
#             train_losses,
#             val_losses,
#             train_accs,
#             val_accs,
#             save_path
#         )

#         print(f"✓ Best model saved at: {best_model_path}")

#     print("\n🎉 Training completed successfully!")


# if __name__ == "__main__":
#     main()



import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

import config
from preprocess import csv_file_generator_with_classes
from utils import (
    split_dataset,
    convert_class_labels,
    generate_equal_label_dataframes,
    save_metrics_to_csv
)
from dataset import IrisPairDataset
from model import MBLNet, IrisSiameseVerificationModel
from loss import IrisSiameseLoss
from train import train_model


def main():

    # ============================================================
    # STEP 1: CSV GENERATION
    # ============================================================

    if not os.path.exists(config.CSV_FILE_Train):
        csv_file_generator_with_classes(
            config.IMAGE_DIR_Train,
            config.CSV_FILE_Train
        )
        convert_class_labels(config.CSV_FILE_Train)
    else:
        print("✓ Training CSV already exists")

    if not os.path.exists(config.CSV_FILE_Recog):
        csv_file_generator_with_classes(
            config.IMAGE_DIR_Test,
            config.CSV_FILE_Recog,
            num_images_per_folder_for_negatives=2
        )
        convert_class_labels(config.CSV_FILE_Recog)
    else:
        print("✓ Testing CSV already exists")

    # ============================================================
    # STEP 2: TRANSFORMS
    # ============================================================

    transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor()
    ])

    # ============================================================
    # STEP 3: VALIDATION SET (RECOGNITION)
    # ============================================================

    val_df = pd.read_csv(config.CSV_FILE_Recog)
    val_dataset = IrisPairDataset(val_df, transform)

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )

    # ============================================================
    # STEP 4: BALANCED TRAINING SETS
    # ============================================================

    dataframe_list, num_classes = generate_equal_label_dataframes(
        config.CSV_FILE_Train
    )

    print(f"✓ Balanced splits       : {len(dataframe_list)}")
    print(f"✓ Number of identities  : {num_classes}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✓ Using device          : {device}")

    # ============================================================
    # STEP 5: MODEL, LOSS, OPTIMIZER
    # ============================================================

    backbone = MBLNet(
        in_channels=1,
        feature_dim=config.FEATURE_DIM,
        attention_type=config.ATTENTION_TYPE
    )

    model = IrisSiameseVerificationModel(backbone).to(device)

    criterion = IrisSiameseLoss()

    optimizer = Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    scheduler = MultiStepLR(
        optimizer,
        milestones=[30, 70, 150],
        gamma=0.1
    )

    best_model_path = None
    best_test_acc = 0.0

    # ============================================================
    # STEP 6: ITERATIVE TRAINING
    # ============================================================

    for idx, df in enumerate(
        tqdm(dataframe_list, desc="Training Splits", unit="split")
    ):
        print(f"\n=== Training Split {idx + 1}/{len(dataframe_list)} ===")

        train_df, test_df = split_dataset(df)

        train_dataset = IrisPairDataset(train_df, transform)
        test_dataset = IrisPairDataset(test_df, transform)

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=config.NUM_WORKERS
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS
        )

        save_path = os.path.join(config.SAVE_DIR, f"iteration_{idx + 1}")
        os.makedirs(save_path, exist_ok=True)

        # Load previous best model (warm start)
        if best_model_path and os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path))

        best_model_path, train_losses, val_losses, train_accs, val_accs, best_test_acc = train_model(
            model,
            train_loader,
            val_loader,
            test_loader,
            optimizer,
            criterion,
            device,
            scheduler,
            save_path,
            num_epochs=config.EPOCHS
        )

        save_metrics_to_csv(
            train_losses,
            val_losses,
            train_accs,
            val_accs,
            save_path
        )

        print(f"✓ Best model saved at: {best_model_path}")

    print("\n🎉 Training completed successfully!")


if __name__ == "__main__":
    main()