import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import SimpleITK as sitk
from torch.cuda.amp import autocast, GradScaler
import time
import shutil
from pathlib import Path

# Import the model architecture
from BuildingBlocks import Encoder, Decoder, FinalConv, DoubleConv, ExtResNetBlock, SingleConv
from sca_3d import SCA3D
from 3d_attention_unet import UNet3D

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define the paths
base_path = './data'
images_path = os.path.join(base_path, 'imagesTr')
labels_path = os.path.join(base_path, 'labelsTr')
test_path = os.path.join(base_path, 'imagesTs')
predictions_path = os.path.join(base_path, 'predictions')

# Create predictions directory if it doesn't exist
os.makedirs(predictions_path, exist_ok=True)

# Dataset class for NII files
class CTDataset(Dataset):
    def __init__(self, image_files, label_files=None, transform=None, patch_size=(128, 128, 128), is_test=False):
        self.image_files = image_files
        self.label_files = label_files
        self.transform = transform
        self.patch_size = patch_size
        self.is_test = is_test
        
    def __len__(self):
        return len(self.image_files)
    
    def normalize(self, image):
        # Apply window level normalization for CT scans
        min_bound = -1000.0
        max_bound = 400.0
        image = np.clip(image, min_bound, max_bound)
        image = (image - min_bound) / (max_bound - min_bound)
        return image
    
    def random_crop_3d(self, image, label=None):
        # Extract random patch
        d, h, w = image.shape
        
        # Ensure patch size doesn't exceed image dimensions
        pd, ph, pw = min(self.patch_size[0], d), min(self.patch_size[1], h), min(self.patch_size[2], w)
        
        # Random crop positions
        d_idx = np.random.randint(0, d - pd + 1) if d > pd else 0
        h_idx = np.random.randint(0, h - ph + 1) if h > ph else 0
        w_idx = np.random.randint(0, w - pw + 1) if w > pw else 0
        
        image_patch = image[d_idx:d_idx+pd, h_idx:h_idx+ph, w_idx:w_idx+pw]
        
        if label is not None:
            label_patch = label[d_idx:d_idx+pd, h_idx:h_idx+ph, w_idx:w_idx+pw]
            return image_patch, label_patch
        
        return image_patch
    
    def pad_if_needed(self, image, target_shape):
        # Pad image to target shape if smaller
        current_shape = image.shape
        padded = np.zeros(target_shape, dtype=image.dtype)
        
        d = min(current_shape[0], target_shape[0])
        h = min(current_shape[1], target_shape[1])
        w = min(current_shape[2], target_shape[2])
        
        padded[:d, :h, :w] = image[:d, :h, :w]
        return padded
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_files[idx]
        img_nii = nib.load(img_path)
        image = img_nii.get_fdata().astype(np.float32)
        
        # Normalize image
        image = self.normalize(image)
        
        if not self.is_test:
            # Load label for training
            label_path = self.label_files[idx]
            label_nii = nib.load(label_path)
            label = label_nii.get_fdata().astype(np.int64)
            
            # Random crop during training
            image, label = self.random_crop_3d(image, label)
            
            # Convert to torch tensor with channel dimension
            image = torch.from_numpy(image).unsqueeze(0)
            label = torch.from_numpy(label).unsqueeze(0)
            
            return {'image': image, 'label': label, 'filename': os.path.basename(img_path)}
        else:
            # For test data, preserve original size but pad if needed for network
            original_shape = image.shape
            
            # Pad if smaller than patch size
            if any(o < p for o, p in zip(original_shape, self.patch_size)):
                image = self.pad_if_needed(image, self.patch_size)
            
            # Convert to torch tensor with channel dimension
            image = torch.from_numpy(image).unsqueeze(0)
            
            return {'image': image, 'filename': os.path.basename(img_path), 'original_shape': original_shape}

# Dice loss function
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        # Flatten prediction and target
        pred_flat = pred.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        
        # Calculate Dice coefficient
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        # Calculate Dice loss
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice

# Dice score calculation
def dice_score(pred, target):
    pred = pred.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    
    # Binarize predictions
    pred = (pred > 0.5).astype(np.int32)
    target = target.astype(np.int32)
    
    # Calculate Dice score
    intersection = np.sum(pred * target)
    union = np.sum(pred) + np.sum(target)
    
    if union == 0:
        return 1.0  # Both pred and target are empty
    
    return 2.0 * intersection / union

# Get all image and label files
def get_files():
    image_files = sorted([os.path.join(images_path, f) for f in os.listdir(images_path) if f.endswith('.nii.gz')])
    label_files = sorted([os.path.join(labels_path, f) for f in os.listdir(labels_path) if f.endswith('.nii.gz')])
    test_files = sorted([os.path.join(test_path, f) for f in os.listdir(test_path) if f.endswith('.nii.gz')])
    
    return image_files, label_files, test_files

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=30, device='cuda'):
    best_val_dice = 0.0
    train_losses = []
    val_dice_scores = []
    scaler = GradScaler()  # For mixed precision training
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        
        # Training phase
        model.train()
        epoch_loss = 0.0
        
        for batch in tqdm(train_loader, desc='Training'):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            # Use mixed precision for faster training
            with autocast():
                outputs, _ = model(images)
                loss = criterion(outputs, labels)
            
            # Scale gradients and optimize
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_dice = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                
                outputs, _ = model(images)
                
                # Apply sigmoid to get probability map
                outputs = torch.sigmoid(outputs)
                
                # Calculate Dice score
                batch_dice = dice_score(outputs, labels)
                val_dice += batch_dice
        
        avg_val_dice = val_dice / len(val_loader)
        val_dice_scores.append(avg_val_dice)
        
        print(f'Training Loss: {avg_train_loss:.4f}, Validation Dice: {avg_val_dice:.4f}')
        
        # Save best model
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'Saved new best model with Dice score: {best_val_dice:.4f}')
        
        # Step the learning rate scheduler
        scheduler.step(avg_val_dice)
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(val_dice_scores)
    plt.title('Validation Dice Score')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()
    
    return model, train_losses, val_dice_scores

# Test function with sliding window
def test_model(model, test_files, device='cuda', overlap=0.5, patch_size=(128, 128, 128)):
    model.eval()
    
    for file_path in tqdm(test_files, desc='Testing'):
        # Load and preprocess the image
        img_nii = nib.load(file_path)
        image = img_nii.get_fdata().astype(np.float32)
        affine = img_nii.affine
        
        # Normalize image
        min_bound = -1000.0
        max_bound = 400.0
        image = np.clip(image, min_bound, max_bound)
        image = (image - min_bound) / (max_bound - min_bound)
        
        # Get image dimensions
        d, h, w = image.shape
        
        # Calculate stride for overlapping patches
        stride_d = int(patch_size[0] * (1 - overlap))
        stride_h = int(patch_size[1] * (1 - overlap))
        stride_w = int(patch_size[2] * (1 - overlap))
        
        # Initialize prediction volume and weight map
        prediction = np.zeros((d, h, w), dtype=np.float32)
        weight_map = np.zeros((d, h, w), dtype=np.float32)
        
        # Sliding window inference
        for z in range(0, d, stride_d):
            z_end = min(z + patch_size[0], d)
            z_start = max(0, z_end - patch_size[0])
            
            for y in range(0, h, stride_h):
                y_end = min(y + patch_size[1], h)
                y_start = max(0, y_end - patch_size[1])
                
                for x in range(0, w, stride_w):
                    x_end = min(x + patch_size[2], w)
                    x_start = max(0, x_end - patch_size[2])
                    
                    # Extract patch
                    patch = image[z_start:z_end, y_start:y_end, x_start:x_end]
                    
                    # Pad if needed
                    if patch.shape != patch_size:
                        temp_patch = np.zeros(patch_size, dtype=np.float32)
                        temp_patch[:patch.shape[0], :patch.shape[1], :patch.shape[2]] = patch
                        patch = temp_patch
                    
                    # Convert to tensor
                    patch_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(device)
                    
                    # Predict
                    with torch.no_grad():
                        with autocast():
                            output, _ = model(patch_tensor)
                            output = torch.sigmoid(output)
                    
                    # Get prediction
                    pred_patch = output.squeeze().cpu().numpy()
                    
                    # Crop if padding was added
                    pred_patch = pred_patch[:z_end-z_start, :y_end-y_start, :x_end-x_start]
                    
                    # Apply linear weights for blending (higher weights in the center)
                    weight_patch = np.ones_like(pred_patch)
                    
                    # Accumulate prediction and weights
                    prediction[z_start:z_end, y_start:y_end, x_start:x_end] += pred_patch
                    weight_map[z_start:z_end, y_start:y_end, x_start:x_end] += weight_patch
        
        # Average predictions based on weights
        weight_map = np.maximum(weight_map, 1e-5)  # Avoid division by zero
        prediction = prediction / weight_map
        
        # Binarize prediction
        binary_prediction = (prediction > 0.5).astype(np.uint8)
        
        # Save prediction as NII
        filename = os.path.basename(file_path)
        output_path = os.path.join(predictions_path, filename)
        
        # Create NIfTI image and save
        pred_nii = nib.Nifti1Image(binary_prediction, affine)
        nib.save(pred_nii, output_path)
        
        print(f'Saved prediction to {output_path}')

# Main execution


# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Get all files
image_files, label_files, test_files = get_files()

if not image_files:
    print("No training images found!")
else:
    # Split training and validation
    train_img, val_img, train_lbl, val_lbl = train_test_split(
        image_files, label_files, test_size=0.2, random_state=42
    )
    
    # Create datasets
    train_dataset = CTDataset(train_img, train_lbl, patch_size=(128, 128, 128))
    val_dataset = CTDataset(val_img, val_lbl, patch_size=(128, 128, 128))
    test_dataset = CTDataset(test_files, is_test=True, patch_size=(128, 128, 128))
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)
    
    # Initialize model
    model = UNet3D(in_channels=1, out_channels=1, final_sigmoid=False, f_maps=16, layer_order='crg')
    model = model.to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')
    
    # Define loss, optimizer and scheduler
    criterion = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    
    # Train the model
    print("Starting training...")
    model, train_losses, val_dice_scores = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, 
        num_epochs=50, device=device
    )
    
    # Load best model for testing
    model.load_state_dict(torch.load('best_model.pth'))
    
    # Test the model
    print("Starting testing...")
    test_model(model, test_files, device=device)
    
    print("Done!")