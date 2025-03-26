import argparse
import os
import numpy as np
import math
import itertools
import sys
import time

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torch.autograd import Variable

from models import *
from datasets import ParticleDataset, create_dataloaders

import torch.nn as nn
import torch.nn.functional as F
import torch

# Create directories for saving outputs
os.makedirs("images/training", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving image samples")
parser.add_argument("--checkpoint_interval", type=int, default=5000, help="batch interval between model checkpoints")
parser.add_argument("--residual_blocks", type=int, default=23, help="number of residual blocks in the generator")
parser.add_argument("--warmup_batches", type=int, default=200, help="number of batches with pixel-wise loss only")
parser.add_argument("--lambda_adv", type=float, default=5e-3, help="adversarial loss weight")
parser.add_argument("--lambda_pixel", type=float, default=1e-2, help="pixel-wise loss weight")
parser.add_argument("--train_split", type=float, default=0.8, help="proportion of data to use for training")
opt = parser.parse_args()
print(opt)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

hr_height, hr_width = 125, 125  # High-resolution image dimensions
hr_shape = (hr_height, hr_width)

# We need a custom upsampling factor
# Factor needed: 125/64 ≈ 1.953125, which is close to 2
# We'll use the standard x2 upsampling and then resize to exact dimensions

generator = GeneratorRRDB(
    channels=3,
    filters=64, 
    num_res_blocks=opt.residual_blocks,
).to(device)

discriminator = Discriminator(input_shape=(3, hr_height, hr_width)).to(device)
feature_extractor = FeatureExtractor().to(device)

# Set feature extractor to inference mode
feature_extractor.eval()

# Losses
criterion_GAN = torch.nn.BCEWithLogitsLoss().to(device)
criterion_content = torch.nn.L1Loss().to(device)
criterion_pixel = torch.nn.L1Loss().to(device)

if opt.epoch != 0:
    # Load pretrained models
    generator.load_state_dict(torch.load(f"saved_models/generator_{opt.epoch}.pth"))
    discriminator.load_state_dict(torch.load(f"saved_models/discriminator_{opt.epoch}.pth"))

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

# Define file paths for your dataset
file_paths = [
    "data/QCDToGGQQ_IMGjet_RH1all_jet0_run0_n36272_LR.parquet", 
    "data/QCDToGGQQ_IMGjet_RH1all_jet0_run1_n47540_LR.parquet", 
    "data/QCDToGGQQ_IMGjet_RH1all_jet0_run2_n55494_LR.parquet"
]

# Create train and validation dataloaders
train_dataloader, val_dataloader = create_dataloaders(
    file_paths=file_paths,
    batch_size=opt.batch_size,
    train_split=opt.train_split,
    num_workers=opt.n_cpu
)

print(f"Training on {len(train_dataloader) * opt.batch_size} samples")
print(f"Validating on {len(val_dataloader) * opt.batch_size} samples")

# Function to handle exact resizing to 125x125
def exact_resize(x):
    return F.interpolate(x, size=(hr_height, hr_width), mode='bicubic', align_corners=True)

# Custom normalization for visualization
def normalize_for_display(tensor):
    """Normalize tensor for visualization (scale to 0-1 range)"""
    min_val = tensor.min()
    max_val = tensor.max()
    if max_val > min_val:  # Avoid division by zero
        return (tensor - min_val) / (max_val - min_val)
    return tensor

# ----------
#  Training
# ----------
best_val_loss = float('inf')
start_time = time.time()

for epoch in range(opt.epoch, opt.n_epochs):
    epoch_start_time = time.time()
    
    # Training loop
    generator.train()
    discriminator.train()
    
    for i, imgs in enumerate(train_dataloader):
        batches_done = epoch * len(train_dataloader) + i

        # Configure model input
        imgs_lr = Variable(imgs["lr"].type(Tensor))
        imgs_hr = Variable(imgs["hr"].type(Tensor))

        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()

        # Generate a high resolution image from low resolution input
        gen_hr = generator(imgs_lr)
        
        # Resize to exact dimensions (125x125)
        gen_hr = exact_resize(gen_hr)

        # Measure pixel-wise loss against ground truth
        loss_pixel = criterion_pixel(gen_hr, imgs_hr)

        if batches_done < opt.warmup_batches:
            # Warm-up (pixel-wise loss only)
            loss_pixel.backward()
            optimizer_G.step()
            print(
                "[Epoch %d/%d] [Batch %d/%d] [G pixel: %f]"
                % (epoch, opt.n_epochs, i, len(train_dataloader), loss_pixel.item())
            )
            continue

        # --------------------
        # FIX: Get discriminator output shapes first
        # --------------------
        pred_real = discriminator(imgs_hr)
        pred_fake = discriminator(gen_hr)
        
        # Create adversarial ground truths with correct shape
        valid = Variable(torch.ones_like(pred_real), requires_grad=False)
        fake = Variable(torch.zeros_like(pred_fake), requires_grad=False)

        # Adversarial loss (relativistic average GAN)
        loss_GAN = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), valid)

        # Content loss
        gen_features = feature_extractor(gen_hr)
        real_features = feature_extractor(imgs_hr).detach()
        loss_content = criterion_content(gen_features, real_features)

        # Total generator loss
        loss_G = loss_content + opt.lambda_adv * loss_GAN + opt.lambda_pixel * loss_pixel

        loss_G.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        pred_real = discriminator(imgs_hr)
        pred_fake = discriminator(gen_hr.detach())

        # Adversarial loss for real and fake images (relativistic average GAN)
        loss_real = criterion_GAN(pred_real - pred_fake.mean(0, keepdim=True), valid)
        loss_fake = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), fake)

        # Total loss
        loss_D = (loss_real + loss_fake) / 2

        loss_D.backward()
        optimizer_D.step()

        # --------------
        #  Log Progress
        # --------------

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, content: %f, adv: %f, pixel: %f]"
            % (
                epoch,
                opt.n_epochs,
                i,
                len(train_dataloader),
                loss_D.item(),
                loss_G.item(),
                loss_content.item(),
                loss_GAN.item(),
                loss_pixel.item(),
            )
        )

        if batches_done % opt.sample_interval == 0:
            # First normalize for better visualization
            imgs_lr_norm = normalize_for_display(imgs_lr)
            gen_hr_norm = normalize_for_display(gen_hr)
            imgs_hr_norm = normalize_for_display(imgs_hr)
            
            # Resize LR to HR size for better comparison
            imgs_lr_up = exact_resize(imgs_lr_norm)
            
            # Create grid: LR (upscaled) | Generated HR | Real HR
            img_grid = torch.cat((imgs_lr_up, gen_hr_norm, imgs_hr_norm), -1)
            save_image(img_grid, f"images/training/{batches_done}.png", nrow=1, normalize=False)

        if batches_done % opt.checkpoint_interval == 0:
            # Save model checkpoints
            torch.save(generator.state_dict(), f"saved_models/generator_{epoch}.pth")
            torch.save(discriminator.state_dict(), f"saved_models/discriminator_{epoch}.pth")
    
    # ----------
    # Validation
    # ----------
    generator.eval()
    val_loss_G = 0.0
    val_loss_pixel = 0.0
    val_loss_content = 0.0
    val_loss_adv = 0.0
    
    with torch.no_grad():
        for i, imgs in enumerate(val_dataloader):
            imgs_lr = imgs["lr"].type(Tensor)
            imgs_hr = imgs["hr"].type(Tensor)
            
            # Generate HR images
            gen_hr = generator(imgs_lr)
            gen_hr = exact_resize(gen_hr)
            
            # Calculate losses
            loss_pixel = criterion_pixel(gen_hr, imgs_hr)
            
            # Only compute full losses if past warmup
            if epoch * len(train_dataloader) >= opt.warmup_batches:
                pred_real = discriminator(imgs_hr)
                pred_fake = discriminator(gen_hr)
                
                # Use correct shape for validation ground truths
                valid = torch.ones_like(pred_real)
                
                loss_GAN = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), valid)
                
                gen_features = feature_extractor(gen_hr)
                real_features = feature_extractor(imgs_hr)
                loss_content = criterion_content(gen_features, real_features)
                
                loss_G = loss_content + opt.lambda_adv * loss_GAN + opt.lambda_pixel * loss_pixel
                
                val_loss_G += loss_G.item()
                val_loss_content += loss_content.item()
                val_loss_adv += loss_GAN.item()
            else:
                val_loss_G += loss_pixel.item()
            
            val_loss_pixel += loss_pixel.item()
    
    # Average validation losses
    val_loss_G /= len(val_dataloader)
    val_loss_pixel /= len(val_dataloader)
    val_loss_content /= len(val_dataloader) if epoch * len(train_dataloader) >= opt.warmup_batches else 1
    val_loss_adv /= len(val_dataloader) if epoch * len(train_dataloader) >= opt.warmup_batches else 1
    
    # Save best model
    if val_loss_G < best_val_loss:
        best_val_loss = val_loss_G
        torch.save(generator.state_dict(), "saved_models/generator_best.pth")
        torch.save(discriminator.state_dict(), "saved_models/discriminator_best.pth")
        print(f"Saved best model with validation loss: {best_val_loss:.6f}")
    
    # Print validation results
    print(
        "\n[Validation] [Epoch %d/%d] [G loss: %f, content: %f, adv: %f, pixel: %f]"
        % (
            epoch,
            opt.n_epochs,
            val_loss_G,
            val_loss_content,
            val_loss_adv,
            val_loss_pixel,
        )
    )
    
    # Calculate epoch time
    epoch_time = time.time() - epoch_start_time
    print(f"Epoch {epoch} completed in {epoch_time:.2f}s")
    
    # Save a validation sample at the end of each epoch
    if len(val_dataloader) > 0:
        imgs = next(iter(val_dataloader))
        imgs_lr = imgs["lr"].type(Tensor)
        imgs_hr = imgs["hr"].type(Tensor)
        
        with torch.no_grad():
            generator.eval()
            gen_hr = generator(imgs_lr)
            gen_hr = exact_resize(gen_hr)
            
            # Normalize for visualization
            imgs_lr_norm = normalize_for_display(imgs_lr)
            gen_hr_norm = normalize_for_display(gen_hr)
            imgs_hr_norm = normalize_for_display(imgs_hr)
            
            # Resize LR to HR size for comparison
            imgs_lr_up = exact_resize(imgs_lr_norm)
            
            # Create comparison grid
            img_grid = torch.cat((imgs_lr_up, gen_hr_norm, imgs_hr_norm), -1)
            save_image(img_grid, f"images/training/epoch_{epoch}_val.png", nrow=1, normalize=False)

# Calculate total training time
total_time = time.time() - start_time
print(f"Training completed in {total_time/3600:.2f} hours")