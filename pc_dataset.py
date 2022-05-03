import torch
from torch.utils.data import Dataset

class PointCloudDataset(Dataset):
    def __init__(self, features_file, bbox_labels_file):
        self.features = torch.load(features_file)
        self.bbox_labels = torch.load(bbox_labels_file)

    def __len__(self):
        return self.bbox_labels.shape[0]

    def __getitem__(self, idx):
        '''
        idx: environment index.
        Returns features and labels.
            features: batch of box features returned by 3DETR on the raw point clouds
            labels: batch of labels; this contains the batch of ground truth bbox vertices
                    and also contains the predictions made by 3DETR (which will be used to compute loss).
        '''

        # Get features
        features = self.features["box_features"][idx, :, :, :]

        # Get label
        label = {
            "bbox_ground_truth": self.bbox_labels[idx, :, :]
            }

        return features, label
