import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
from utils import CropRowDataset, CropRowLabel, CropRowDataset_image_mask, EarlyStopping, DiceLoss, clDiceLoss, DiceLoss1, myIoU, SoftIoU, SoftIoULoss
from uNet import my_UNet, UNet
from torchmetrics.classification import BinaryJaccardIndex, Dice
from sklearn.model_selection import train_test_split
from cldice import soft_dice_cldice
import albumentations as A
import cv2

def get_train_augmentation():
    transform = [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        #A.RandomBrightnessContrast(p=0.5),
        #A.GaussianBlur(blur_limit=(3, 3), sigma_limit=(1, 1), p=1.0),
        #A.Normalize(mean=(0.5), std=(0.2)),
    ]
    return A.Compose(transform, additional_targets={'mask': 'image'})

def get_val_transform():
    transform = [
        A.Normalize(mean=(0.5), std=(0.2)),
    ]
    return A.Compose(transform, additional_targets={'mask': 'image'})

'''
def get_image_augmentation():
    image_transform = [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.GaussianBlur(blur_limit=(3, 3), sigma_limit=(1, 1), p=1.0),
        A.Normalize(mean=[0.5], std=[0.5]),
    ]
    return A.Compose(image_transform)

def get_mask_augmentation():
    mask_transform = [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
    ]
    return A.Compose(mask_transform, additional_targets={'mask': 'image'})
'''


class ToInt64Tensor:
    def __call__(self, data):
        return data.to(torch.int64)


if __name__ == "__main__":
    # Load the data
    ids = pd.read_csv("train and test ids.csv")
    train_ids = ids["train_ids"].tolist()

    train_ids, val_ids = train_test_split(train_ids, test_size=0.15, random_state=42)

    # Create data loader
    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    label_transform = transforms.Compose([
        transforms.ToTensor(),
        #ToInt64Tensor(),
    ])

    img_folder = "Images"
    label_folder = "train_labels"
    train_dataset = CropRowDataset_image_mask(img_folder, label_folder, train_ids, transform=get_train_augmentation(), trainOrNot=True)
    val_dataset = CropRowDataset_image_mask(img_folder, label_folder, val_ids, transform=val_transform, trainOrNot=False)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate the model
    seed = 1234
    torch.manual_seed(seed)
    model = my_UNet(n_channels=1, n_classes=1).to(device)

    # Set up the loss function and optimizer
    criterion = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0025)
    IoU_Jaccard = BinaryJaccardIndex().to(device)

    # Train the model
    early_stopping = EarlyStopping(patience=10, delta=1e-5)
    num_epochs = 150
    train_loss_list, train_IoU_list, val_loss_list, val_IoU_list = [], [], [], []
    Early_stop_epoch = 0
    for epoch in tqdm(range(num_epochs)):
        print(f"Epoch {epoch+1}/{num_epochs}")
        running_loss = 0.0
        running_IoU = 0.0

        for (train_inputs, train_targets) in train_loader:
            train_inputs, train_targets = train_inputs.float().to(device), train_targets.float().to(device)
        
            optimizer.zero_grad()

            train_outputs = model(train_inputs)
            train_loss = criterion(train_outputs, train_targets)
            train_outputs = torch.sigmoid(train_outputs)
            train_predict = torch.where(train_outputs < 0.5, 0, 1)
            train_IoU = IoU_Jaccard(train_predict, train_targets)

            train_loss.backward()
            optimizer.step()

            running_loss += train_loss.item()
            running_IoU += train_IoU.item()

        epoch_loss = running_loss / len(train_loader)
        train_loss_list.append(epoch_loss)
        print(f"Train Dice Loss: {epoch_loss:.4f}")

        epoch_IoU = running_IoU / len(train_loader)
        train_IoU_list.append(epoch_IoU)
        print(f"Train IoU: {epoch_IoU:.4f}")

        # Validate the model
        val_loss = 0.0
        val_IoU = 0.0
        for (val_inputs, val_targets) in val_loader:
            val_inputs, val_targets = val_inputs.float().to(device), val_targets.float().to(device)

            with torch.no_grad():
                val_outputs = model(val_inputs)
                loss = criterion(val_outputs, val_targets)
                val_outputs = torch.sigmoid(val_outputs)
                val_predict = torch.where(val_outputs < 0.5, 0, 1)
                IoU = IoU_Jaccard(val_outputs, val_targets)
                val_loss += loss.item()
                val_IoU += IoU.item()

        val_loss /= len(val_loader)
        val_IoU /= len(val_loader)
        print(f"Validation Dice Loss: {val_loss:.4f}")
        print(f"Validation IoU: {val_IoU:.4f}")
        val_loss_list.append(val_loss)
        val_IoU_list.append(val_IoU)

        # Save the model every 25 epochs
        if epoch > 0 and epoch % 5 == 0:
            model_save_path = "model/model_epoch_" + str(epoch) + ".pth"
            torch.save(model.state_dict(), model_save_path)

        # Check early stopping
        early_stopping(val_loss)
        if early_stopping.stop:
            Early_stop_epoch = epoch + 1
            print("Early stopping")
            # break

        torch.cuda.empty_cache()

    # visualize loss value as a graph
    fig1 = plt.figure()
    ax1 = plt.axes()
    plt.xlabel('Epoch')
    plt.ylabel('Dice loss')
    plt.title('Dice loss vs. Epochs')
    ax1.plot(range(Early_stop_epoch), train_loss_list, label="Training")
    ax1.plot(range(Early_stop_epoch), val_loss_list, label="Validation")
    plt.legend()
    plt.savefig("loss_curve_epoch_" + str(Early_stop_epoch) + ".png", dpi=300)
    plt.show()

    # visualize loss value as a graph
    fig2 = plt.figure()
    ax2 = plt.axes()
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.title('IoU vs. Epochs')
    ax2.plot(range(Early_stop_epoch), train_IoU_list, label="Training")
    ax2.plot(range(Early_stop_epoch), val_IoU_list, label="Validation")
    plt.legend()
    plt.savefig("IoU_curve_epoch_" + str(Early_stop_epoch) + ".png", dpi=300)
    plt.show()
