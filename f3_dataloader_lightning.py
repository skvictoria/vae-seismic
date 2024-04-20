import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule
from torchvision import transforms
from typing import List, Optional, Sequence, Union, Any, Callable

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class SeismicDataset(Dataset):
    def __init__(self, data, labels, view='inline', subimage_width=64, subimage_height=64, normalize=True):
        """
        Args:
            data (np.array): 3D array of seismic data.
            labels (np.array): 3D array of labels.
            view (str): 'inline' or 'crossline' view for slicing the 3D array.
            subimage_width (int): Width of the sub-images.
            subimage_height (int): Height of the sub-images.
        """
        self.data = data
        self.labels = labels
        self.view = view
        self.subimage_width = subimage_width
        self.subimage_height = subimage_height
        self.normalize = normalize
        self.subimages = self._create_subimages()

    def _create_subimages(self):
        subimages = []
        for i in range(self.data.shape[0 if self.view == 'inline' else 1]):
            if self.view == 'inline':
                view_slice = self.data[i, :, :].T
                label_slice = self.labels[i, :, :].T
            else:  # crossline
                view_slice = self.data[:, i, :].T
                label_slice = self.labels[:, i, :].T

            if self.normalize:
                view_slice = (view_slice - np.min(view_slice)) / (np.max(view_slice) - np.min(view_slice))
            
            for y in range(0, view_slice.shape[0] - self.subimage_height + 1, self.subimage_height):
                for x in range(0, view_slice.shape[1] - self.subimage_width + 1, self.subimage_width):
                    subimage_data = view_slice[y:y+self.subimage_height, x:x+self.subimage_width]
                    subimage_label = label_slice[y:y+self.subimage_height, x:x+self.subimage_width]
                    subimages.append((subimage_data, subimage_label))
        return subimages

    def __len__(self):
        return len(self.subimages)

    def __getitem__(self, idx):
        data, label = self.subimages[idx]
        return torch.tensor(data, dtype=torch.float), torch.tensor(label, dtype=torch.long)

class VAEDataset(LightningDataModule):
    """
    PyTorch Lightning data module adapted for seismic data segmentation including test dataset.
    """
    def __init__(
        self,
        data,
        labels,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        test_batch_size: int = 8,
        subimage_width: int = 64,
        subimage_height: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        normalize: bool = True,
    ):
        super().__init__()
        self.data = data
        self.labels = labels
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.subimage_width = subimage_width
        self.subimage_height = subimage_height
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.normalize = normalize

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == 'fit' or stage is None:
            self.train_dataset = SeismicDataset(
                self.data['train'], self.labels['train'], 
                subimage_width=self.subimage_width, subimage_height=self.subimage_height, normalize = self.normalize
            )
            self.val_dataset = SeismicDataset(
                self.data['test1'], self.labels['test1'], 
                subimage_width=self.subimage_width, subimage_height=self.subimage_height, normalize = self.normalize
            )
        if stage == 'test' or stage is None:
            self.test_dataset = SeismicDataset(
                self.data['test2'], self.labels['test2'], 
                subimage_width=self.subimage_width, subimage_height=self.subimage_height, normalize = self.normalize
            )
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

# Example of setting up and using the VAEDataset
'''test1 = np.load('D:\jupyter_notebooks\ECE6254\Project\\test_once\\test1_seismic.npy')
test2 = np.load('D:\jupyter_notebooks\ECE6254\Project\\test_once\\test2_seismic.npy')
train = np.load('D:\jupyter_notebooks\ECE6254\Project\\train\\train_seismic.npy')

test1_labels = np.load('D:\jupyter_notebooks\ECE6254\Project\\test_once\\test1_labels.npy')
test2_labels = np.load('D:\jupyter_notebooks\ECE6254\Project\\test_once\\test2_labels.npy')
train_labels = np.load('D:\jupyter_notebooks\ECE6254\Project\\train\\train_labels.npy')

# Example of setting up and using the VAEDataset
vae_data_module = VAEDataset(
    data={'train': train, 'test1': test1, 'test2': test2},
    labels={'train': train_labels, 'test1': test1_labels, 'test2': test2_labels},
    train_batch_size=16,
    val_batch_size=16,
    test_batch_size=16,
    subimage_width=64,
    subimage_height=64,
    normalize = True
)

trainer = pl.Trainer()
trainer.fit(model, datamodule=vae_data_module)

trainer.validate(datamodule=vae_data_module)
trainer.test(datamodule=vae_data_module)'''
