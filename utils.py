import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image, ImageOps
from tqdm import tqdm
import cv2

# Define Dice loss
class DiceLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        dice_loss = 0
        
        logits = torch.sigmoid(logits)
        logits = logits.view(-1, num_classes, logits.shape[2] * logits.shape[3])
        targets = targets.view(-1, num_classes, targets.shape[2] * targets.shape[3])

        for cls in range(num_classes):
            intersection = torch.sum(logits[:, cls, :] * targets[:, cls, :])
            cardinality = torch.sum(logits[:, cls, :] + targets[:, cls, :])

            # Calculate Dice Loss for the current class and add it to the total loss
            dice_loss += (1 - ((2. * intersection + self.eps) / (cardinality + self.eps)))
        
        # Average the Dice Loss across all classes
        dice_loss /= num_classes
        
        return dice_loss

class DiceLoss1(nn.Module):
    def __init__(self, eps=1e-7):
        super(DiceLoss1, self).__init__()
        self.eps = eps

    def forward(self, logits, targets):
        logits = torch.sigmoid(logits)
        batch_size = targets.size(0)

        logits = logits.view(batch_size, -1)
        #print(logits)
        #print(logits.shape)
        targets = targets.view(batch_size, -1)
        #print(targets)
        #print(targets.shape)
        intersection = (logits * targets).sum(dim=-1)
        #print(intersection)
        union = (logits*logits).sum(dim=-1) + (targets*targets).sum(dim=-1)
        #print(union)

        dice_score = 2 * intersection / (union + self.eps)
        #print(dice_score)
        dice_loss = 1 - dice_score.mean()
        #print(dice_loss)
        #sys.exit()

        return dice_loss

# Define the clDice loss

class clDiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(clDiceLoss, self).__init__()
        self.eps = eps

    def forward(self, logits, targets):
        batch_size = targets.size(0)

        logits = logits.view(batch_size, -1)
        targets = targets.view(batch_size, -1)

        tp = torch.sum(logits * targets, dim=1)
        fp = torch.sum(logits * (1 - targets), dim=1)
        fn = torch.sum((1 - logits) * targets, dim=1)

        cl_dice = 2 * tp / (2 * tp + fp + fn + self.eps)
        cl_dice_loss = torch.mean(1 - cl_dice)

        return cl_dice_loss

# Import image data
def load_image(folder_path, image_ids):
    images = []

    for image_id in image_ids:
        # Convert the id to a three-digit string
        image_id_str = f"{int(image_id):03d}"
        image_name = f"crop_row_{image_id_str}.jpg"
        image_path = os.path.join(folder_path, image_name)

        # Load the image
        image = Image.open(image_path)
        # Select only one channel (e.g., red channel)
        channel_1, _, _ = image.split()

        # Convert the selected channel to a NumPy array and normalize it
        channel_1 = (np.array(channel_1) / 255.0).astype(np.float32)

        images.append(channel_1)

    return images
    

class CropRowDataset(Dataset):
    def __init__(self, image_folder, image_ids, transform=None):
        self.image_folder = image_folder
        self.image_ids = image_ids
        self.transform = transform
        self.images = load_image(image_folder, image_ids)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img = self.images[idx]

        if self.transform:
            img = self.transform(img)

        return img

# Impot image labels
def load_label(folder_path, label_ids):
    labels= []

    for id in tqdm(label_ids):
        label_id_str = f"{int(id):03d}"
        label_name = f"crop_row_{label_id_str}.npy"
        label_path = os.path.join(folder_path, label_name)
        label = np.load(label_path)
        label_1_channel = label[:, :, 0]
        label_1_channel = (label_1_channel / 255.0).astype(np.float32) # convert pixel=255 to 1
        labels.append(label_1_channel)
    
    return labels

class CropRowLabel(Dataset):
    def __init__(self, label_folder, label_ids, transform=None):
        self.label_folder = label_folder
        self.label_ids = label_ids
        self.transform = transform
        self.labels = load_label(label_folder, label_ids)

    def __len__(self):
        return len(self.label_ids)

    def __getitem__(self, idx):
        label = self.labels[idx]

        if self.transform:
            label = self.transform(label)

        return label

def encode_mask_to_rle(mask):
    '''
    mask: numpy array binary mask 
    255 - mask 
    0 - background
    Returns encoded run length 
    '''
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    
    return ' '.join(str(x) for x in runs)


def decode_rle_to_mask(rle, height = 240, width = 320):
    '''
    rle : run-length as string formated (start value, count)
    height : height of the mask 
    width : width of the mask
    returns binary mask
    '''
    rle = np.array(rle.split(' ')).reshape(-1, 2)
    mask = np.zeros((height*width))
    color = 255
    for i in rle:
        mask[int(i[0]):int(i[0])+int(i[1])] = color

    return mask.reshape(height, width)


def removeEmptyLabel(data):
    drop_index = []
    for i in tqdm(range(0, data.shape[0])):
        if (data["labels"].isna()[i] and i!=120):
            data["labels"][i-1] = data["labels"][i-1] + data.index[i]
            drop_index.append(i)
        elif (data["labels"].isna()[i] and i==120):
            data["labels"][i-1] = data["labels"][i-1] + " " + data.index[i]
            drop_index.append(i)
    return data.drop(data.index[index] for index in drop_index)


class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.delta:
            self.counter = 0
            self.best_loss = val_loss
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True


# Define image and label loader together
def load_image_and_mask(folder_path, image_ids, mask_folder):
    images = []
    masks = []

    for image_id in image_ids:
        # Convert the id to a three-digit string
        image_id_str = f"{int(image_id):03d}"
        image_name = f"crop_row_{image_id_str}.jpg"
        mask_name = f"crop_row_{image_id_str}.npy"
        image_path = os.path.join(folder_path, image_name)
        mask_path = os.path.join(mask_folder, mask_name)

        # Load the image using OpenCV
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        # Convert the image from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = np.load(mask_path)

        # Select only one channel (e.g., red channel)
        channel_1, _, _ = cv2.split(image)

        # Convert the selected channel and mask to a NumPy array and normalize them
        channel_1 = (channel_1 / 255.0).astype(np.float32)
    
        label_1_channel = mask[:, :, 0]
        label_1_channel = (label_1_channel / 255.0).astype(np.float32) # convert pixel=255 to 1
        
        '''
        # Extract the ROI
        h, w = channel_1.shape
        roi_h, roi_w = 128, 256
        start_h = (h - roi_h) // 2
        start_w = (w - roi_w) // 2
        channel_1_roi = channel_1[start_h:start_h + roi_h, start_w:start_w + roi_w]
        label_1_channel_roi = label_1_channel[start_h:start_h + roi_h, start_w:start_w + roi_w]

        images.append(channel_1_roi)
        masks.append(label_1_channel_roi)
        ''' 

        images.append(channel_1)
        masks.append(label_1_channel)
        
    return images, masks


class CropRowDataset_image_mask(Dataset):
    def __init__(self, image_folder, mask_folder, image_ids, transform=None, trainOrNot=True):
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.image_ids = image_ids
        self.transform = transform
        self.trainOrNot = trainOrNot
        self.images, self.masks = load_image_and_mask(image_folder, image_ids, mask_folder)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img = self.images[idx]
        mask = self.masks[idx]

        if self.trainOrNot:
            if self.transform:
                augmented = self.transform(image=img, mask=mask)
                img = augmented['image']
                mask = augmented['mask']
                #mask = (mask > 0.5).astype(np.float32)

            img = np.expand_dims(img, axis=0)
            mask = np.expand_dims(mask, axis=0)  
            return torch.from_numpy(img).float(), torch.from_numpy(mask).float()
              
        else:
            img = self.transform(img)
            mask = self.transform(mask)
            return img, mask

class myIoU(nn.Module):
    def __init__(self, threshold=0.5, eps=1e-6):
        super(myIoU, self).__init__()
        self.threshold = threshold
        self.eps = eps

    def forward(self, prediction, target):
        assert prediction.size() == target.size(), "Input and target sizes must match"

        prediction = prediction.squeeze(1).double()  # BATCH x 1 x H x W => BATCH x H x W
        target = target.squeeze(1).double()  # BATCH x 1 x H x W => BATCH x H x W
    
        intersection = (prediction & target).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
        union = (prediction | target).float().sum((1, 2))         # Will be zero if both are 0
        
        iou = (intersection + self.eps) / (union + self.eps)  # We smooth our devision to avoid 0/0
        
        thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
        
        return thresholded.mean() # average across the batch

class SoftIoU(nn.Module):
    def __init__(self, eps=1e-6):
        super(SoftIoU, self).__init__()
        self.eps = eps

    def forward(self, logits, targets):
        logits = torch.sigmoid(logits)
        batch_size = targets.size(0)

        logits = logits.view(batch_size, -1)
        targets = targets.view(batch_size, -1)

        intersection = (logits * targets).sum(dim=-1)
        union = logits.sum(dim=-1) + targets.sum(dim=-1) - intersection

        iou_score = intersection / (union + self.eps)
        return iou_score.mean()


class SoftIoULoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(SoftIoULoss, self).__init__()
        self.eps = eps

    def forward(self, prediction, target):
        assert prediction.size() == target.size(), "Input and target sizes must match"

        # Calculate the intersection and union
        intersection = (prediction * target).sum(dim=(2, 3))
        union = prediction.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection

        # Calculate the soft IoU
        soft_iou = (intersection + self.eps) / (union + self.eps)

        # Return the soft IoU loss (1 - soft IoU)
        return 1 - soft_iou.mean()    