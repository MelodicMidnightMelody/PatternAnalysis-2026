import time
import torch
import torch.optim as optim
import argparse
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

from modules import UNet2D, dice_loss, dice_coefficient
from dataset import get_dataloaders

def main():
    parser = argparse.ArgumentParser(description='Train U-Net for HipMRI segmentation')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to dataset')
    parser.add_argument('--epochs', type=int, default=40,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save model checkpoints')

    args = parser.parse_args()

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using device: ', device)

    train_dataloader, val_dataloader, test_dataloader, id_channels, num_classes = get_dataloaders(args.data_path, num_workers=2, batch_size=args.batch_size)

    #Setting model

    model = UNet2D(in_channels=1, num_classes=num_classes).to(device)

    num_parameters = 0
    trainable_parameters = 0

    for param in model.parameters():
        num_parameters += param.numel()
        if param.requires_grad:
            trainable_parameters += param.numel()

    print('Total number of parameters: ', num_parameters)
    print('Trainable parameters: ', trainable_parameters)

    optimiser = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, patience=2)

    training_losses = []
    val_losses = []
    best_val_loss = float('inf')
    save_path = Path(args.save_dir) / 'best_model.pth'

    start_time = time.time()

    for epoch in tqdm(range(args.epochs)):

        training_loss = run_epoch(model, train_dataloader, optimiser, device)
        training_losses.append(training_loss)

        scheduler.step(training_loss[-1])

        val_loss = epoch(model, val_dataloader, device=device)
        val_losses.append(val_loss)

        print("Epoch: ", epoch)
        print("Training Loss: ", training_loss)
        print("Validation Loss: ", val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            checkpoint = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimiser': optimiser.state_dict(),
                'val_loss': val_loss,
            }
            torch.save(checkpoint, save_path)
            print(f"Best model loss (val_loss): {val_loss:.4f}")

def run_epoch(model, dataloader, optimiser=None, device='cpu'):

    if optimiser is not None:
        model.train()
    else:
        model.eval()
    total_loss = 0.0

    for image, mask in tqdm(dataloader):
        image = image.to(device)
        mask = mask.to(device)

        logits = model(image)
        loss = dice_loss(logits, mask)

        if optimiser is not None:
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


if __name__ == "__main__":
    main()