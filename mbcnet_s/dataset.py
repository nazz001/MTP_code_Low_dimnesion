# import torch
# from PIL import Image
# from torch.utils.data import Dataset
# import pandas as pd
# import os


# class IrisSingleImageDataset(Dataset):
#     """
#     Dataset for SINGLE iris images using CSV input.

#     Expected CSV format:
#         image_path, class_label

#     Example:
#         /data/iris/001/img1.png, 1
#         /data/iris/001/img2.png, 1
#         /data/iris/002/img1.png, 2
#     """

#     def __init__(self, images_df, transform=None, mode="train"):
#         """
#         Args:
#             images_df (pd.DataFrame): DataFrame with image paths and labels
#             transform (callable): Image transformations
#             mode (str): train / test / eval (for future use)
#         """
#         self.images_df = images_df
#         self.transform = transform
#         self.mode = mode

#     def __len__(self):
#         """Return total number of samples"""
#         return len(self.images_df)

#     def __getitem__(self, idx):
#         """
#         Load image and label for given index
#         """

#         # ----------------------------------
#         # Load image
#         # ----------------------------------
#         img_path = self.images_df.iloc[idx, 0]
#         # img = Image.open(img_path).convert("RGB")
#         Image.open(img_path).convert("L")

#         # here read image and convert into rgp 

#         if self.transform:
#             img = self.transform(img)

#         # ----------------------------------
#         # Load class label (identity)

#         # ----------------------------------
#         class_label = int(self.images_df.iloc[idx, 1])
#         class_label = torch.tensor(class_label, dtype=torch.long)
#         # here class lavel 1 which class it belong to 

#         return img, class_label



# # import os
# # import torch
# # import pandas as pd
# # from PIL import Image
# # from torch.utils.data import Dataset


# class IrisPairDataset(Dataset):
#     """
#     Dataset for GENUINE / IMPOSTER iris image pairs using CSV input.

#     Expected CSV format:
#         image_path1, class1, image_path2, class2, label

#     where:
#         label = 1 → genuine pair (same identity)
#         label = 0 → imposter pair (different identities)
#     """

#     def __init__(self, pairs_df, transform=None):
#         """
#         Args:
#             pairs_df (pd.DataFrame): Pair CSV loaded as DataFrame
#             transform (callable): Image transformations
#         """
#         self.pairs_df = pairs_df
#         self.transform = transform

#         # Filter out invalid pairs (missing images)
#         self.valid_pairs_df = self._filter_valid_pairs()

#     def _filter_valid_pairs(self):
#         """
#         Remove pairs where one or both image paths do not exist.
#         """
#         valid_rows = []

#         for idx in range(len(self.pairs_df)):
#             img1_path = self.pairs_df.iloc[idx, 0]
#             img2_path = self.pairs_df.iloc[idx, 2]

#             if os.path.exists(img1_path) and os.path.exists(img2_path):
#                 valid_rows.append(self.pairs_df.iloc[idx])

#         return pd.DataFrame(valid_rows).reset_index(drop=True)

#     def __len__(self):
#         """Total number of valid image pairs"""
#         return len(self.valid_pairs_df)

#     def __getitem__(self, idx):
#         """
#         Return one pair:
#             (img1, class1, img2, class2, pair_label)
#         """

#         # ----------------------------------
#         # Read CSV row
#         # ----------------------------------
#         img1_path = self.valid_pairs_df.iloc[idx, 0]
#         class1 = int(self.valid_pairs_df.iloc[idx, 1])

#         img2_path = self.valid_pairs_df.iloc[idx, 2]
#         class2 = int(self.valid_pairs_df.iloc[idx, 3])

#         pair_label = int(self.valid_pairs_df.iloc[idx, 4])

#         # ----------------------------------
#         # Load images
#         # ----------------------------------
#         img1 = Image.open(img1_path).convert("RGB")
#         img2 = Image.open(img2_path).convert("RGB")

#         if self.transform:
#             img1 = self.transform(img1)
#             img2 = self.transform(img2)

#         # ----------------------------------
#         # Convert labels to tensors
#         # ----------------------------------
#         class1 = torch.tensor(class1, dtype=torch.long)
#         class2 = torch.tensor(class2, dtype=torch.long)
#         pair_label = torch.tensor(pair_label, dtype=torch.float32)

#         return img1, class1, img2, class2, pair_label



import torch
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import os


class IrisSingleImageDataset(Dataset):
    """
    Dataset for SINGLE iris images using CSV input.

    Expected CSV format:
        image_path, class_label

    Example:
        /data/iris/001/img1.png, 1
        /data/iris/001/img2.png, 1
        /data/iris/002/img1.png, 2
    """

    def __init__(self, images_df, transform=None, mode="train"):
        """
        Args:
            images_df (pd.DataFrame): DataFrame with image paths and labels
            transform (callable): Image transformations
            mode (str): train / test / eval (for future use)
        """
        self.images_df = images_df
        self.transform = transform
        self.mode = mode

    def __len__(self):
        """Return total number of samples"""
        return len(self.images_df)

    def __getitem__(self, idx):
        """
        Load image and label for given index
        """

        # ----------------------------------
        # Load image
        # ----------------------------------
        img_path = self.images_df.iloc[idx, 0]
        img = Image.open(img_path).convert("L")  # Convert to grayscale
        
        if self.transform:
            img = self.transform(img)

        # ----------------------------------
        # Load class label (identity)
        # ----------------------------------
        class_label = int(self.images_df.iloc[idx, 1])
        class_label = torch.tensor(class_label, dtype=torch.long)

        return img, class_label


class IrisPairDataset(Dataset):
    """
    Dataset for GENUINE / IMPOSTER iris image pairs using CSV input.

    Expected CSV format:
        image_path1, class1, image_path2, class2, label

    where:
        label = 1 → genuine pair (same identity)
        label = 0 → imposter pair (different identities)
    """

    def __init__(self, pairs_df, transform=None):
        """
        Args:
            pairs_df (pd.DataFrame): Pair CSV loaded as DataFrame
            transform (callable): Image transformations
        """
        self.pairs_df = pairs_df
        self.transform = transform

        # Filter out invalid pairs (missing images)
        self.valid_pairs_df = self._filter_valid_pairs()

    def _filter_valid_pairs(self):
        """
        Remove pairs where one or both image paths do not exist.
        """
        valid_rows = []

        for idx in range(len(self.pairs_df)):
            img1_path = self.pairs_df.iloc[idx, 0]
            img2_path = self.pairs_df.iloc[idx, 2]

            if os.path.exists(img1_path) and os.path.exists(img2_path):
                valid_rows.append(self.pairs_df.iloc[idx])

        if len(valid_rows) == 0:
            raise ValueError("No valid image pairs found. Check your image paths.")

        return pd.DataFrame(valid_rows).reset_index(drop=True)

    def __len__(self):
        """Total number of valid image pairs"""
        return len(self.valid_pairs_df)

    def __getitem__(self, idx):
        """
        Return one pair:
            (img1, class1, img2, class2, pair_label)
        """

        # ----------------------------------
        # Read CSV row
        # ----------------------------------
        img1_path = self.valid_pairs_df.iloc[idx, 0]
        class1 = int(self.valid_pairs_df.iloc[idx, 1])

        img2_path = self.valid_pairs_df.iloc[idx, 2]
        class2 = int(self.valid_pairs_df.iloc[idx, 3])

        pair_label = int(self.valid_pairs_df.iloc[idx, 4])

        # ----------------------------------
        # Load images - Convert to grayscale for iris
        # ----------------------------------
        img1 = Image.open(img1_path).convert("L")  # L = grayscale
        img2 = Image.open(img2_path).convert("L")  # L = grayscale

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        # ----------------------------------
        # Convert labels to tensors
        # ----------------------------------
        class1 = torch.tensor(class1, dtype=torch.long)
        class2 = torch.tensor(class2, dtype=torch.long)
        pair_label = torch.tensor(pair_label, dtype=torch.float32)

        return img1, class1, img2, class2, pair_label