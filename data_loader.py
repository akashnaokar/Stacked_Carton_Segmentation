import os
import sys
import json
import numpy as np
import cv2
import random
import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class CartonSegmentationDataset(Dataset):
    """
    Dataset class for Carton dataset using LabelMe polygon annotations
    Specifically designed for Mask R-CNN
    """
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (string): Root directory of the dataset
            transform (callable, optional): Optional transform to be applied on images
        """
        self.data_dir = data_dir
        self.transform = transform
        
        # Define paths - now using a single labelme directory that contains both images and annotations
        self.labelme_dir = os.path.join(data_dir, 'labelme')
        
        # Get image files from the labelme directory
        self.image_files = [f for f in os.listdir(self.labelme_dir) if f.endswith('.jpg')]
        
        # Map class names to IDs (in this case, just 'Carton')
        self.class_map = {'Carton': 1}  # Background is 0 by default
        
    def __len__(self):
        return len(self.image_files)
    
    def _get_json_file(self, img_file):
        """Find the corresponding LabelMe JSON file for an image"""
        # Extract the base name and create json filename
        base_name = os.path.splitext(img_file)[0]
        json_file = os.path.join(self.labelme_dir, f"{base_name}.json")
        
        if os.path.exists(json_file):
            return json_file
        
        return None  # No annotation found
    
    def polygons_to_mask(self, polygons, height, width):
        """Convert polygons to binary mask"""
        masks = []
        for polygon in polygons:
            # Create a binary mask from polygon
            mask = np.zeros((height, width), dtype=np.uint8)
            
            # Convert polygon to numpy array of integer pairs
            polygon_np = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
            
            # Fill the polygon
            cv2.fillPoly(mask, [polygon_np], 1)
            masks.append(mask)
        
        if masks:
            # Stack all masks into a single array (H, W, N)
            return np.stack(masks, axis=2).astype(np.uint8)
        else:
            # Return empty mask with correct shape
            return np.zeros((height, width, 0), dtype=np.uint8)
    
    def polygons_to_bboxes(self, polygons):
        """Convert polygons to bounding boxes [x1, y1, x2, y2]"""
        boxes = []
        for polygon in polygons:
            points = np.array(polygon)
            x1, y1 = points.min(axis=0)
            x2, y2 = points.max(axis=0)
            boxes.append([x1, y1, x2, y2])
        return boxes
    
    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        img_path = os.path.join(self.labelme_dir, img_file)
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        width, height = img.size
        
        # Find and load annotation file
        json_file = self._get_json_file(img_file)
        
        boxes = []
        labels = []
        masks = []
        
        if json_file and os.path.exists(json_file):
            with open(json_file, 'r') as f:
                annotation = json.load(f)
            
            polygons = []
            for shape in annotation.get('shapes', []):
                if shape['label'] in self.class_map:
                    label = self.class_map[shape['label']]
                    polygon = shape['points']
                    polygons.append(polygon)
                    labels.append(label)
            
            # Convert polygons to bboxes and masks
            if polygons:
                boxes = self.polygons_to_bboxes(polygons)
                mask_array = self.polygons_to_mask(polygons, height, width)
                
                # Convert mask array to tensor format expected by Mask R-CNN
                for i in range(mask_array.shape[2]):
                    masks.append(mask_array[:, :, i])
        
        # Create empty arrays if no annotations found
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)
            masks = torch.zeros((0, height, width), dtype=torch.uint8)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            masks = torch.as_tensor(np.array(masks), dtype=torch.uint8).bool()
        
        # Create target dictionary in the format expected by Mask R-CNN
        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'image_id': torch.tensor([idx]),
            'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64).bool()
        }
        
        # Apply transformations if any
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        
        return img, target
    
    def visualize_item(self, idx):
        """Visualize an image with its segmentation masks"""
        img, target = self[idx]
        
        # Convert tensor to numpy for visualization
        img_np = img.permute(1, 2, 0).numpy()
        
        # Plot the image
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Original image with bounding boxes
        ax1.imshow(img_np)
        ax1.set_title('Bounding Boxes')
        
        boxes = target['boxes'].numpy()
        masks = target['masks'].numpy()
        
        # Add boxes
        for box in boxes:
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='r', facecolor='none')
            ax1.add_patch(rect)
        
        # Show masks on the second subplot
        ax2.imshow(img_np)
        ax2.set_title('Segmentation Masks')
        
        # Apply a different color for each mask
        colors = plt.cm.tab10(np.linspace(0, 1, len(masks)))
        
        for i, mask in enumerate(masks):
            masked = np.ma.masked_where(mask == 0, mask)
            ax2.imshow(masked, alpha=0.5, cmap=plt.get_cmap('jet'), interpolation='none')
        
        plt.tight_layout()
        plt.show()

# Custom collate function to handle variable sized masks and boxes
def collate_fn(batch):
    return tuple(zip(*batch))

def get_maskrcnn_dataloader(data_dir='carton_data', max_samples=None, train_ratio=0.8, batch_size=2, num_workers=4, seed=42, return_dataset=False):
    """
    Create DataLoaders for the carton dataset suitable for Mask R-CNN with custom train-val split
    
    Args:
        data_dir (string): Root directory of the dataset
        max_samples (int, optional): Maximum number of samples to use (for memory constraints)
        train_ratio (float): Ratio of data to use for training (between 0 and 1)
        batch_size (int): Batch size
        num_workers (int): Number of workers for data loading
        seed (int): Random seed for reproducible train/val splits
        return_dataset: If True, return datasets instead of dataloaders
    
    Returns:
        If return_dataset is False:
            train_dataloader, val_dataloader
        If return_dataset is True:
            train_dataset, val_dataset

        Format Tuple: (train_dataloader, val_dataloader) PyTorch DataLoaders for training and validation
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create the full dataset
    full_dataset = CartonSegmentationDataset(data_dir=data_dir, transform=transform)
    
    # Limit the number of samples if specified
    dataset_size = len(full_dataset)
    if max_samples and max_samples < dataset_size:
        # Create a subset with randomly selected indices
        torch.manual_seed(seed)
        indices = torch.randperm(dataset_size).tolist()[:max_samples]
        limited_dataset = Subset(full_dataset, indices)
        dataset_size = max_samples
    else:
        limited_dataset = full_dataset
    
    # Calculate number of samples for train and validation
    train_size = int(train_ratio * dataset_size)
    val_size = dataset_size - train_size
    
    # Split the dataset
    train_dataset, val_dataset = random_split(
        limited_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )
    

    
    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    if return_dataset:
        return train_dataset, val_dataset
    else:
        return train_dataloader, val_dataloader

# Example of how to use this code with Mask R-CNN:
if __name__ == "__main__":
    # Create train and validation dataloaders with a maximum of 100 samples
    train_dataloader, val_dataloader = get_maskrcnn_dataloader(
        max_samples=100,  # Limit to 100 samples to avoid GPU memory issues
        train_ratio=0.8   # 80% training, 20% validation
    )
    print(f"Training samples: {len(train_dataloader.dataset)}")
    print(f"Validation samples: {len(val_dataloader.dataset)}")
    
    # # Example of iterating through the dataloader
    # for images, targets in train_dataloader:
    #     print(f"Batch size: {len(images)}")
    #     print(f"Image shape: {images[0].shape}")
    #     print(f"Number of boxes: {targets[0]['boxes'].shape[0]}")
    #     print(f"Number of masks: {targets[0]['masks'].shape[0]}")
    #     break

    # Visualize a sample
    original_dataset = train_dataloader.dataset.dataset
    if isinstance(original_dataset, Subset):
        original_dataset = original_dataset.dataset
    
    # Get a list of indices from the training subset
    train_indices = train_dataloader.dataset.indices

    # Choose a random index
    train_idx = random.choice(train_indices)

    if isinstance(original_dataset, Subset):
        # If we're using a subset of a subset
        train_idx = original_dataset.indices[train_idx]

    image, target = original_dataset[train_idx]

    print(f"Visualized Image shape: {image.shape}")
    print(f"Visualized Number of boxes: {target['boxes'].shape[0]}")
    print(f"Visualized Number of masks: {target['masks'].shape[0]}")
    
    original_dataset.visualize_item(train_idx)