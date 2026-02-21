import random
import numpy as np
import torch

from configs.config import Config
from data.dataloader import get_dataloaders
from models.unet import UNet
from losses.dice_loss import BCEDiceLoss
from trainer.trainer import train_model

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    set_seed(Config.SEED)
    DEVICE = Config.DEVICE

    train_loader, val_loader = get_dataloaders(
        Config.TRAIN_IMG_DIR,
        Config.TRAIN_ANN_FILE,
        Config.VAL_IMG_DIR,
        Config.VAL_ANN_FILE,
        Config.BATCH_SIZE,
        Config.NUM_WORKERS
    )

    model = UNet(Config.IN_CHANNELS, Config.OUT_CHANNELS).to(DEVICE)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.T_MAX)

    trained_model, best_dice = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        DEVICE,
        Config.SAVE_DIR,
        Config.EPOCHS
    )

    print(f"Best Dice Score: {best_dice}")

if __name__ == "__main__":
    main()