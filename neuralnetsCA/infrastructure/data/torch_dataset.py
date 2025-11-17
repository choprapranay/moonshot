import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from domain.entities import DatasetSplit

class SuperDataSet(Dataset):
    def __init__(self, batter_ids, pitch_type_ids, features, targets):
        self.batter_ids = torch.from_numpy(batter_ids).long()
        self.pitch_type_ids = torch.from_numpy(pitch_type_ids).long()
        self.features = torch.from_numpy(features).float()
        self.targets = torch.from_numpy(targets).long()
    
    def __getitem__(self, index):
        batter_id = self.batter_ids[index]
        pitch_type_id = self.pitch_type_ids[index]
        features = self.features[index]
        y = self.targets[index]
        return batter_id, pitch_type_id, features, y
    
    def __len__(self):
        return self.targets.shape[0]

def build_dataloaders(split: DatasetSplit, batch_size: int):

    train_ds = SuperDataSet(
        split.batter_ids_train,
        split.pitch_type_ids_train,
        split.features_train,
        split.y_train,
    )
    val_ds = SuperDataSet(
        split.batter_ids_val,
        split.pitch_type_ids_val,
        split.features_val,
        split.y_val,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader