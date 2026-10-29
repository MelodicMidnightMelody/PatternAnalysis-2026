import time
import torch
import torch.optim as optim
import argparse
import json
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from modules import UNet2D, dice_counts, combined_loss
from dataset import get_dataloaders

def plot_training_curves(training_losses, val_losses, save_path):

    epochs = range(1, len(training_losses) + 1)
    best_epoch = val_losses.index(min(val_losses)) + 1

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, training_losses, label="Training Loss", marker="o")
    ax.plot(epochs, val_losses, label="Validation Loss", marker="o")
    ax.axvline(best_epoch, color="red", linestyle="--", alpha=0.6, label=f"Best val (epoch {best_epoch})")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Dice + CE Combined Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend()

    fig.savefig(save_path)
    plt.close(fig)

@torch.no_grad()
def dice_score(model, dataloader, device, eps = 1e-6):
    model.eval()
    inter_total = None
    union_total = None
    for image, mask in tqdm(dataloader):
        image, mask = image.to(device), mask.to(device)
        inter, union = dice_counts(model(image), mask)
        inter_total = inter if inter_total is None else inter_total + inter
        union_total = union if union_total is None else union_total + union

    dice = ((2 * inter_total + eps) / (union_total + eps))
    return dice.cpu()

def run_epoch(model, dataloader, optimiser=None, device='cpu'):

    if optimiser is not None:
        model.train()
    else:
        model.eval()
    total_loss = 0.0
    with torch.set_grad_enabled(optimiser is not None):
        for image, mask in tqdm(dataloader):
            image = image.to(device)
            mask = mask.to(device)

            logits = model(image)
            loss = combined_loss(logits, mask)

            if optimiser is not None:
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss




def main():
    """

    :return:
    """

    parser = argparse.ArgumentParser(description='Train U-Net for HipMRI segmentation')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to dataset')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save model checkpoints')

    args = parser.parse_args()

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

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

    #img, m = train_dataloader.dataset[0]
    #print(img.shape, img.mean().item(), img.std().item())
    #print(m.shape, (m.sum(dim=(1, 2)) / m[0].numel()))  # fraction of pixels per class
    #import matplotlib.pyplot as plt
    #fig, ax = plt.subplots(1, 2)
    #ax[0].imshow(img[0], cmap="gray")
    #ax[1].imshow(m.argmax(0))
    #plt.show()

    for epoch in range(args.epochs):

        training_loss = run_epoch(model, train_dataloader, optimiser, device)
        training_losses.append(training_loss)

        scheduler.step(training_losses[-1])

        val_loss = run_epoch(model, val_dataloader, device=device)
        val_losses.append(val_loss)

        print("\nEpoch: ", epoch + 1)
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
            print(f"\nBest model loss (val_loss): {val_loss:.4f}")

    total_time = time.time() - start_time

    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint['state_dict'])
    plot_training_curves(training_losses, val_losses, "./plots/training_curves.png")

    test_dice = dice_score(model, test_dataloader, device=device)

    print(f"Dice per channel: ")
    for label, score in enumerate(test_dice.tolist()):
        print(f"Label: {label}, Dice score: {score:.4f}")

    print(f"Mean dice: {test_dice.mean().item():.4f}")

    prostate_channel = 5

    results = {
        'dice_score_per_channel': test_dice.tolist(),
        'mean_dice': test_dice.mean().item(),
        'prostate_dice': test_dice[prostate_channel].item(),
        'num_classes': num_classes,
        'label_to_ch': id_channels,
        'training time': total_time,
    }


    with open(Path(args.save_dir) / 'test_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.save_dir}")

    # Check if requirement is met
    if test_dice[prostate_channel].item() >= 0.75:
        print("\nSUCCESS: Prostate Dice Score >= 0.75")
    else:
        print("\nFAILURE: Prostate Dice Score< 0.75")





if __name__ == "__main__":
    main()