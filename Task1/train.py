import torch
import torch.nn as nn
import torch.optim as optim
import h5py
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
import argparse
from pathlib import Path
from tqdm import tqdm
from models import resnet15v2

optimizers_map = {
    'adamw': optim.AdamW,   
    'adam': optim.Adam,
    'sgd': optim.SGD
}

class H5Dataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.file_path = file_path
        self.transform = transform
        with h5py.File(file_path, 'r') as f:
            self.data = f['X'][...] # (N, H, W, C)
            self.labels = f['y'][...] # (N,)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        x = torch.from_numpy(x).permute(2, 0, 1).float()

        # One-hot encode the labels
        y = torch.tensor(self.labels[idx]).long()
        y = torch.nn.functional.one_hot(y, num_classes=2).float()

        if self.transform:
            x = self.transform(x)
        return x, y

class NormalizeTransform:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return (x - self.mean) / self.std

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runname', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--device', type=str, default='cuda')
    return parser.parse_args()

def get_dataloaders(batch_size):
    mean = torch.tensor([0.00114013, -0.00022465]).view(-1, 1, 1)
    std = torch.tensor([0.02360118, 0.06654396]).view(-1, 1, 1)
    
    transform = NormalizeTransform(mean, std)

    # create dataset
    dataset_e = H5Dataset('data/SingleElectronPt50_IMGCROPS_n249k_RHv1.hdf5', transform=transform)
    dataset_p = H5Dataset('data/SinglePhotonPt50_IMGCROPS_n249k_RHv1.hdf5', transform=transform)
    
    # concatenate the datasets
    full_dataset = ConcatDataset([dataset_e, dataset_p])

    # Split dataset into training and testing (80/20 split)
    total_size = len(full_dataset)
    test_size = int(0.2 * total_size)
    train_size = total_size - test_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    # DataLoader with multiple workers and pinned memory for faster data transfer
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, test_loader

# Training loop for one epoch
def train_one_epoch(model, optimizer, criterion, dataloader, device):
    model.train()
    total_loss = 0.0
    total_samples = 0
    
    for x, y in dataloader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        total_samples += x.size(0)
    return total_loss / total_samples

# Evaluation loop
def evaluate(model, criterion, dataloader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            outputs = model(x)
            loss = criterion(outputs, y)
            total_loss += loss.item() * x.size(0)
            predicted = outputs.argmax(dim=1)
            labels = y.argmax(dim=1)
            total += x.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total, total_loss / total


def main():
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    train_loader, test_loader = get_dataloaders(args.batch_size)
    
    model = resnet15v2().to(device)
    optimizer = optimizers_map[args.optim.lower()](model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    best_acc = 0.0
    output_dir = Path(args.runname)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in tqdm(range(args.epochs), desc="Training"):
        train_loss = train_one_epoch(model, optimizer, criterion, train_loader, device)
        test_acc, test_loss = evaluate(model, criterion, test_loader, device)
        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f} - Test Loss: {test_loss:.4f} - Test Acc: {test_acc:.4f}")
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), output_dir / "best.pth")
    
    torch.save(model.state_dict(), output_dir / "last.pth")

if __name__ == '__main__':
    main()
