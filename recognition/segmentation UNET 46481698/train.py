import torch
import torch.optim as optim
import argparse
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

from modules import UNet2D, dice_loss, dice_coefficient
from dataset import get_dataloaders

def main():
    parser = argparse.ArgumentParser(description='Train U-Net for HipMRI segmentation')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to dataset')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--base_channels', type=int, default=32,
                        help='Base number of channels in U-Net')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save model checkpoints')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')

    args = parser.parse_args()

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    device = args.device if torch.cuda.is_available() else 'cpu'
    print('Using device: ', device)

    train_dataloader, val_dataloader, test_dataloader, id_channels, num_classes = get_dataloaders(args.data_path, num_workers=2, batch_size=args.batch_size)

    #Setting model

    model = UNet2D(in_channels=1, num_classes=num_classes, base_channels=args.base_channels).to(device)

    num_parameters = 0
    trainable_parameters = 0

    for param in model.parameters():
        num_parameters += param.numel()
        if param.requires_grad:
            trainable_parameters += param.numel()

    print('Total number of parameters: ', num_parameters)
    print('Trainable parameters: ', trainable_parameters)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')



if __name__ == "__main__":
    main()