import torch
import pandas as pd
import os
from torch.utils.data import Dataset, TensorDataset

class TargetsDataset(Dataset):
    def __init__(self, csv_dir, save_dir, normalize=False, regenerate=False):
        super(TargetsDataset, self).__init__()

        self.save_path = os.path.join(save_dir, 'targets.pt')
        self.normalize = normalize

        if os.path.exists(self.save_path) and not regenerate:
            self.targets = torch.load(self.save_path)
        else:
            # Exclude uncharacterized molecules because they don't have graphs in the dataset
            exclude_df = pd.read_csv(os.path.join(csv_dir, "uncharacterized.csv")) 
            exclude_indices = exclude_df['exclude_indices'].tolist()

            targets_df = pd.read_csv(os.path.join(csv_dir, "gdb9.sdf.csv"))
            numeric_df = targets_df.iloc[:, 1:].astype('float32')
            self.targets = torch.tensor(numeric_df.values)


            mask = torch.ones(len(self.targets), dtype=torch.bool)
            mask[exclude_indices] = False
            
            self.targets = self.targets[mask]

            #Might not be the best way to normalize, or even necessary tbh
            if normalize:
                column_min = self.targets.min(axis=0).values  # Shape: [19]
                column_max = self.targets.max(axis=0).values  # Shape: [19]

                # print(column_min)
                # print(column_max)

                # Normalize each column independently
                self.targets = 2 * (self.targets - column_min) / (column_max - column_min) - 1

            os.makedirs(save_dir, exist_ok=True)
            torch.save(self.targets, self.save_path)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.targets[idx]

# Example usage:
if __name__ == "__main__":
    # csv_path = "../../data/custom_qm9/raw"
    csv_path = "../../data/custom_qm9/raw"
    save_dir = "../../data/etc"
    targets_dataset = TargetsDataset(csv_path, save_dir, normalize=False, regenerate=True)

    target = targets_dataset[0]
    print(target)
    print(len(targets_dataset.targets))