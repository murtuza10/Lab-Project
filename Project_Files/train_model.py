import torch
import torch.nn as nn
import wandb
import numpy as np
import json
import matplotlib.pyplot as plt
from tqdm import tqdm

from dust3r.model import AsymmetricCroCo3DStereo
import croco.utils.misc as misc


class Regr3D(nn.Module):
    """3D Regression Loss (Regr3D) for DUSt3R."""

    def __init__(self, norm_type='L21', norm_mode='avg_dis'):
        super(Regr3D, self).__init__()
        self.norm_type = norm_type
        self.norm_mode = norm_mode

    def forward(self, pred_pts, gt_pts, valid_mask=None):
        error = pred_pts - gt_pts  # Compute per-point error

        if valid_mask is not None:
            error = error * valid_mask.unsqueeze(-1)  # Apply mask

        if self.norm_type == 'L1':
            loss = torch.abs(error).sum(dim=-1)
        elif self.norm_type == 'L2':
            loss = torch.norm(error, p=2, dim=-1)
        elif self.norm_type == 'L21':
            loss = torch.norm(error, p=2, dim=-1).sum(dim=-1).sqrt()
        else:
            raise ValueError(f"Invalid norm type: {self.norm_type}")

        if valid_mask is not None:
            loss = loss.sum(dim=-1) / valid_mask.sum(dim=-1).clamp(min=1)
        else:
            loss = loss.mean(dim=-1)

        return loss.mean()


class ConfLoss(nn.Module):
    """Confidence-Weighted Loss for DUSt3R."""

    def __init__(self, base_loss, alpha=0.2):
        super(ConfLoss, self).__init__()
        self.base_loss = base_loss
        self.alpha = alpha

    def forward(self, gt1, gt2, pred1, pred2):
        total_loss = 0.0

        for v, (gt, pred) in enumerate([(gt1, pred1), (gt2, pred2)]):
            try:
                pred_pts = pred['pts3d']
            except KeyError:
                pred_pts = pred['pts3d_in_other_view']

            gt_pts = gt
            confidence = pred.get('conf', None)

            regr_loss = self.base_loss(pred_pts, gt_pts)

            if confidence is not None:
                weighted_loss = (confidence * regr_loss).mean()
                conf_reg = -self.alpha * torch.log(confidence + 1e-8).mean()
                view_loss = weighted_loss + conf_reg
            else:
                view_loss = regr_loss.mean()

            total_loss += view_loss

        return total_loss / 2  # Average loss


def train(NUM_EPOCHS, model, criterion, train_loader, val_loader, optimizer, scheduler, device, use_wandb=True):
    """Training loop for the model."""

    loss_dic, val_dic = {}, {}

    model.to(device)

    for epoch in range(NUM_EPOCHS):
        train_loss, val_loss = 0.0, 0.0
        num_train_batches = len(train_loader)
        num_val_batches = len(val_loader)

        model.train()
        train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} Training")

        for data in train_progress_bar:
            view1, view2, gt1, gt2 = (
                data['view1'].to(device),
                data['view2'].to(device),
                data['absolute_dept'].to(device),
                data['relative_dept'].to(device),
            )

            gt1, gt2 = gt1.view(gt1.shape[0], 224, 224, 3), gt2.view(gt2.shape[0], 224, 224, 3)
            view1, view2 = {'img': view1}, {'img': view2}

            preds = model(view1, view2)
            loss = criterion(gt1, gt2, preds[0], preds[1])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            loss_dic.setdefault(epoch, []).append(loss.item())

            if use_wandb:
                wandb.log({"training_loss": loss.item()})

            train_progress_bar.set_postfix({"loss": loss.item()})

        scheduler.step()

        # Validation loop
        model.eval()
        with torch.no_grad():
            val_progress_bar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} Validation")
            for data in val_progress_bar:
                view1, view2, gt1, gt2 = (
                    data['view1'].to(device),
                    data['view2'].to(device),
                    data['absolute_dept'].to(device),
                    data['relative_dept'].to(device),
                )

                gt1, gt2 = gt1.view(gt1.shape[0], 224, 224, 3), gt2.view(gt2.shape[0], 224, 224, 3)
                view1, view2 = {'img': view1}, {'img': view2}

                preds = model(view1, view2)
                loss_val = criterion(gt1, gt2, preds[0], preds[1])

                val_loss += loss_val.item()
                val_dic.setdefault(epoch, []).append(loss_val.item())

                if use_wandb:
                    wandb.log({"validation_loss": loss_val.item()})

    return model, loss_dic, val_dic


def train_model(model, train_dataloader, test_dataloader):
    """Initialize and train the model."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_loss = Regr3D(norm_type='L2', norm_mode='avg_dis')
    criterion = ConfLoss(base_loss, alpha=0.2).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=1 / 3)

    NUM_EPOCHS = 30

    # Initialize WandB
    wandb.login(key="ed7faaa7784428261467aee38c86ccc5c316f954")
    wandb.init(project="Dust3r_thermal", name="Dust3r_thermal", config={"learning_rate": 1e-5, "epochs": NUM_EPOCHS})

    wandb.watch(model, log_freq=5)

    trained_model, loss_train, loss_val = train(NUM_EPOCHS, model, criterion, train_dataloader, test_dataloader, optimizer, scheduler, device)

    # Save Model
    model_path = "../checkpoints/dust3r_thermal_epoch16_lr1e-5_224x224.pth"
    torch.save({'model': trained_model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'epoch': NUM_EPOCHS}, model_path)
    print(f"Model saved at {model_path}")


def main():
    """Main function to execute training."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model_path = "../checkpoints/DUSt3R_ViTLarge_BaseDecoder_224_linear.pth"
    model = AsymmetricCroCo3DStereo.from_pretrained(model_path).to(device)

    if train_dataloader and test_dataloader:
        train_model(model, train_dataloader, test_dataloader)
    else:
        print("Error: Dataset not loaded!")


if __name__ == "__main__":
    main()
