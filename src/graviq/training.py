"""Training loop, losses, and metrics for GraviQ."""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

from graviq.models import get_model
from graviq.data import get_dataloaders

# Optional TensorBoard support
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None


class DiceLoss(nn.Module):
    """Dice loss for segmentation"""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        predictions = torch.sigmoid(predictions)
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        intersection = (predictions * targets).sum()
        dice = (2. * intersection + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)
        return 1 - dice


def compute_pos_weight(train_loader, device='cpu'):
    """pos_weight = neg_count / pos_count for BCEWithLogitsLoss (minority=tunnel)."""
    pos_count = 0.0
    neg_count = 0.0
    for batch in train_loader:
        m = batch['mask']
        pos_count += (m > 0.5).float().sum().item()
        neg_count += (m <= 0.5).float().sum().item()
    if pos_count == 0:
        return torch.tensor([1.0], device=device)
    ratio = neg_count / (pos_count + 1e-9)
    ratio = max(1.0, min(100.0, float(ratio)))
    return torch.tensor([ratio], dtype=torch.float32, device=device)


class CombinedLoss(nn.Module):
    """Combination of BCE and Dice loss with optional label smoothing"""

    def __init__(
        self,
        pos_weight,
        bce_weight: float = 0.3,
        dice_weight: float = 0.7,
        label_smoothing: float = 0.05,
        device: str = 'cpu',
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.label_smoothing = float(label_smoothing)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice = DiceLoss()

    def forward(self, predictions, targets):
        # Binary label smoothing: move hard labels slightly toward 0.5
        if self.label_smoothing > 0.0:
            eps = self.label_smoothing
            targets = targets * (1.0 - eps) + 0.5 * eps
        bce_loss = self.bce(predictions, targets)
        dice_loss = self.dice(predictions, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def calculate_metrics(predictions, targets, threshold=0.5):
    """Calculate IoU, Dice, Precision, Recall"""
    preds = (torch.sigmoid(predictions) > threshold).float()
    preds = preds.view(-1)
    targets = targets.view(-1)
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    iou = (intersection + 1e-7) / (union + 1e-7)
    dice = (2 * intersection + 1e-7) / (preds.sum() + targets.sum() + 1e-7)
    true_positive = intersection
    predicted_positive = preds.sum()
    actual_positive = targets.sum()
    precision = (true_positive + 1e-7) / (predicted_positive + 1e-7)
    recall = (true_positive + 1e-7) / (actual_positive + 1e-7)
    return {
        'iou': iou.item(),
        'dice': dice.item(),
        'precision': precision.item(),
        'recall': recall.item()
    }


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    epoch_loss = 0
    epoch_metrics = {'iou': 0, 'dice': 0, 'precision': 0, 'recall': 0}
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')

    for batch in pbar:
        inputs = batch['input'].to(device)
        targets = batch['mask'].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        metrics = calculate_metrics(outputs, targets)
        epoch_loss += loss.item()
        for key in epoch_metrics:
            epoch_metrics[key] += metrics[key]
        pbar.set_postfix({'loss': loss.item(), 'dice': metrics['dice']})

    num_batches = len(train_loader)
    epoch_loss /= num_batches
    for key in epoch_metrics:
        epoch_metrics[key] /= num_batches
    return epoch_loss, epoch_metrics


def validate(model, val_loader, criterion, device, epoch):
    """Validate the model"""
    model.eval()
    epoch_loss = 0
    epoch_metrics = {'iou': 0, 'dice': 0, 'precision': 0, 'recall': 0}
    pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')

    with torch.no_grad():
        for batch in pbar:
            inputs = batch['input'].to(device)
            targets = batch['mask'].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            metrics = calculate_metrics(outputs, targets)
            epoch_loss += loss.item()
            for key in epoch_metrics:
                epoch_metrics[key] += metrics[key]
            pbar.set_postfix({'loss': loss.item(), 'dice': metrics['dice']})

    num_batches = len(val_loader)
    epoch_loss /= num_batches
    for key in epoch_metrics:
        epoch_metrics[key] /= num_batches
    return epoch_loss, epoch_metrics


def save_predictions(model, val_loader, device, save_dir, num_samples=5):
    """Save sample predictions for visualization"""
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    batch = next(iter(val_loader))
    inputs = batch['input'].to(device)
    targets = batch['mask'].to(device)

    with torch.no_grad():
        outputs = model(inputs)
        preds = torch.sigmoid(outputs)

    inputs = inputs.cpu().numpy()
    targets = targets.cpu().numpy()
    preds = preds.cpu().numpy()

    for i in range(min(num_samples, inputs.shape[0])):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].imshow(inputs[i, 0], cmap='inferno', origin='upper')
        axes[0].set_title('Input Density Grid')
        axes[0].axis('off')
        axes[1].imshow(targets[i, 0], cmap='binary', origin='upper', vmin=0, vmax=1)
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        axes[2].imshow(preds[i, 0], cmap='binary', origin='upper', vmin=0, vmax=1)
        axes[2].set_title('Prediction')
        axes[2].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'sample_{i}.png'), dpi=100)
        plt.close()


def train(
    data_dir='training_data',
    num_epochs=100,
    batch_size=8,
    learning_rate=1e-3,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    save_dir='checkpoints',
    log_dir='runs'
):
    """Main training function"""
    if not TENSORBOARD_AVAILABLE:
        print("WARNING: TensorBoard not available. Training will proceed without logging.")
    else:
        print(f"TensorBoard logging enabled. Run: tensorboard --logdir={log_dir}")

    print(f"Using device: {device}")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir) if TENSORBOARD_AVAILABLE else None

    print("Loading data...")
    train_loader, val_loader = get_dataloaders(
        data_dir,
        batch_size=batch_size,
        train_split=0.8,
        num_workers=0
    )

    print("Computing pos_weight from train set...")
    pos_weight = compute_pos_weight(train_loader, device)
    print(f"  pos_weight (neg/pos): {pos_weight.item():.2f}")

    print("Initializing model...")
    model = get_model(device)
    criterion = CombinedLoss(
        pos_weight=pos_weight,
        bce_weight=0.3,
        dice_weight=0.7,
        label_smoothing=0.05,
        device=device,
    )
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    best_val_dice = 0
    print(f"\nStarting training for {num_epochs} epochs...")

    for epoch in range(1, num_epochs + 1):
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_loss, val_metrics = validate(model, val_loader, criterion, device, epoch)
        scheduler.step(val_loss)

        if writer:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Dice/train', train_metrics['dice'], epoch)
            writer.add_scalar('Dice/val', val_metrics['dice'], epoch)
            writer.add_scalar('IoU/val', val_metrics['iou'], epoch)
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        print(f"\nEpoch {epoch}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Dice: {train_metrics['dice']:.4f}")
        print(f"  Val   Loss: {val_loss:.4f}, Dice: {val_metrics['dice']:.4f}, IoU: {val_metrics['iou']:.4f}")

        if val_metrics['dice'] > best_val_dice:
            best_val_dice = val_metrics['dice']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': best_val_dice,
                'val_loss': val_loss,
            }, os.path.join(save_dir, 'best_model.pth'))
            print(f"  ✓ Saved best model (Dice: {best_val_dice:.4f})")

        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth'))
            save_predictions(model, val_loader, device,
                             os.path.join(save_dir, f'predictions_epoch_{epoch}'))

    if writer:
        writer.close()
    print(f"\nTraining complete! Best validation Dice: {best_val_dice:.4f}")
