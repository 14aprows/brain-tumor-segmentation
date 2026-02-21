from torch.utils.data import DataLoader
from data.dataset import BrainSegmentationDataset
from data.preprocessing import get_train_transforms, get_valid_transforms

def get_dataloaders(train_img_dir, train_ann_file, valid_img_dir, valid_ann_file, batch_size=8, num_workers=4):
    train_dataset = BrainSegmentationDataset(img_dir = train_img_dir, ann_file = train_ann_file, transform = get_train_transforms())
    valid_dataset = BrainSegmentationDataset(img_dir = valid_img_dir, ann_file = valid_ann_file, transform = get_valid_transforms())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, valid_loader