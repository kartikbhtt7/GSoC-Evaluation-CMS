import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import os

class ParticleDataset(Dataset):
    def __init__(self, file_paths, transform=None, test=False, batch_size=10000):
        """
        Custom dataset for particle physics data from parquet files
        """
        # Load and concatenate data from all parquet files
        self.data = self._load_data(file_paths, batch_size)
        self.transform = transform
        self.test = test
        
    def _load_data(self, file_paths, batch_size=10000):
        """Load and concatenate data from multiple parquet files using iter_batches"""
        df_list = []
        
        for fp in file_paths:
            if os.path.exists(fp):
                try:
                    pf = pq.ParquetFile(fp)
                    
                    # Read the file in batches
                    result_tables = []
                    total_rows = 0
                    
                    for batch in pf.iter_batches(batch_size=batch_size):
                        # Convert batch to table and then to pandas dataframe
                        table = pa.Table.from_batches([batch])
                        batch_df = table.to_pandas()
                        
                        # Process the batch data
                        batch_df['X_jets'] = batch_df['X_jets'].apply(
                            lambda x: np.stack([np.stack(ch).astype(np.float32) for ch in x])
                        )
                        batch_df['X_jets_LR'] = batch_df['X_jets_LR'].apply(
                            lambda x: np.stack([np.stack(ch).astype(np.float32) for ch in x])
                        )
                        
                        df_list.append(batch_df)
                        total_rows += batch.num_rows
                        
                        print(f"Processed {total_rows} rows from {fp}")
                        
                except Exception as e:
                    print(f"Error processing file {fp}: {e}")
            else:
                print(f"Warning: File {fp} does not exist")
        
        if not df_list:
            raise FileNotFoundError("No valid parquet files found")
        
        # Concatenate all dataframes
        df = pd.concat(df_list, ignore_index=True)
        return df
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        lr_img = self.data['X_jets_LR'].iloc[idx].astype(np.float32)  # Shape: (3, 64, 64)
        hr_img = self.data['X_jets'].iloc[idx].astype(np.float32)     # Shape: (3, 125, 125)
        
        y = self.data['y'].iloc[idx]
        
        lr_img = torch.from_numpy(lr_img)
        hr_img = torch.from_numpy(hr_img)
        y = torch.tensor(y, dtype=torch.float32)
        
        if self.transform:
            lr_img = self.transform(lr_img)
            hr_img = self.transform(hr_img)
        
        lr_img = (lr_img - lr_img.min()) / (lr_img.max() - lr_img.min()) * 2 - 1
        hr_img = (hr_img - hr_img.min()) / (hr_img.max() - hr_img.min()) * 2 - 1
        
        return {"lr": lr_img, "hr": hr_img, "y": y}

def create_dataloaders(file_paths, batch_size=4, train_split=0.8, num_workers=4, parquet_batch_size=10000):
    """
    Create train and validation dataloaders
    """
    dataset = ParticleDataset(file_paths, batch_size=parquet_batch_size)
    
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader