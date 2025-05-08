import os
import torch
import numpy as np
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import matplotlib.pyplot as plt
from tqdm import tqdm

def get_model_instance(num_classes: int):
    """
    Get Mask R-CNN model with ResNet-50 backbone
    
    Args:
        num_classes (int): Number of classes (including background)
    
    Returns:
        model: Mask R-CNN model pre-trained on COCO and modified for carton segmentation
    """
    # Load pre-trained model
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn(weights=weights)
    
    # Get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the box predictor with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Get number of input features for the mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # Replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    
    return model



def train_one_epoch(model, optimizer, data_loader, device, epoch, writer=None, print_freq=10):
    """Train the model for one epoch"""
    model.train()
    
    metrics = {'loss': 0, 'loss_classifier': 0, 'loss_box_reg': 0, 'loss_mask': 0, 'loss_objectness': 0, 'loss_rpn_box_reg': 0}
    total_batches = 0
    
    pbar = tqdm(data_loader, desc=f"Epoch {epoch}")
    for images, targets in pbar:
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        batch_loss = 0
        
        # Handle the case where loss_dict is a list
        if isinstance(loss_dict, list):
            for d in loss_dict:
                if isinstance(d, dict):
                    batch_total_loss = sum(v for v in d.values() if torch.is_tensor(v) and v.numel() == 1)
                    batch_loss += batch_total_loss
                    # Update individual metrics
                    for k, v in d.items():
                        if k in metrics and torch.is_tensor(v) and v.numel() == 1:
                            metrics[k] += v.item()
        else:
            # Calculate total loss for the batch
            batch_total_loss = sum(v for v in loss_dict.values() if torch.is_tensor(v) and v.numel() == 1)
            batch_loss += batch_total_loss
            
            # Update individual metrics
            for k, v in loss_dict.items():
                if k in metrics and torch.is_tensor(v) and v.numel() == 1:
                    metrics[k] += v.item()
        
        # Update the total loss metric
        metrics['loss'] += batch_loss.item()
        
        # Backward pass
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()
        
        # Update progress bar
        pbar.set_postfix({'loss': batch_loss.item()})
        total_batches += 1
    
    # Log metrics to TensorBoard
    if writer:
        for k, v in metrics.items():
            writer.add_scalar(f'train/{k}', v / total_batches, epoch)
    
    return {k: v / total_batches for k, v in metrics.items()}

def evaluate(model, data_loader, device):
    """Evaluate the model on the validation set"""
    model.eval()
    
    metrics = {'loss': 0, 'loss_classifier': 0, 'loss_box_reg': 0, 'loss_mask': 0, 'loss_objectness': 0, 'loss_rpn_box_reg': 0}
    total_batches = 0
    
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Validation"):
            try:
                images = [image.to(device) for image in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                # Force training mode temporarily to compute losses
                model.train()
                loss_dict = model(images, targets)
                model.eval()
                
                batch_loss = 0
                
                # Handle the case where loss_dict is a list
                if isinstance(loss_dict, list):
                    for d in loss_dict:
                        if isinstance(d, dict):
                            batch_total_loss = sum(v.item() for v in d.values() if torch.is_tensor(v) and v.numel() == 1)
                            batch_loss += batch_total_loss
                            # Update individual metrics
                            for k, v in d.items():
                                if k in metrics and torch.is_tensor(v) and v.numel() == 1:
                                    metrics[k] += v.item()
                else:
                    # Calculate total loss for the batch
                    batch_total_loss = sum(v.item() for v in loss_dict.values() if torch.is_tensor(v) and v.numel() == 1)
                    batch_loss += batch_total_loss
                    
                    # Update individual metrics
                    for k, v in loss_dict.items():
                        if k in metrics and torch.is_tensor(v) and v.numel() == 1:
                            metrics[k] += v.item()
                
                # Update the total loss metric
                metrics['loss'] += batch_loss
                
                # Increment batch counter
                total_batches += 1
                # torch.cuda.empty_cache()
                
            except RuntimeError as e:
                print(f"Warning: Error during validation batch: {e}")
                continue
    
    # Avoid division by zero
    if total_batches == 0:
        print("Warning: No validation batches were processed successfully")
        return metrics
    
    avg_metrics = {k: v / total_batches for k, v in metrics.items()}
    
    return avg_metrics


def visualize_predictions(model, data_loader, device, num_samples=3):
    """Visualize model predictions on a few samples"""
    model.eval()
    
    # Get dataset from dataloader
    dataset = data_loader.dataset
    # Handle if dataset is a Subset
    while hasattr(dataset, 'dataset'):
        dataset = dataset.dataset
    
    fig, axs = plt.subplots(num_samples, 2, figsize=(12, 6 * num_samples))
    
    # Ensure axs is always a 2D array even with one sample
    if num_samples == 1:
        axs = axs.reshape(1, -1)
    
    for i in range(num_samples):
        # Get a random sample from the validation set
        sample_idx = torch.randint(0, len(dataset), (1,)).item()
        img, target = dataset[sample_idx]
        
        # Original image with ground truth
        img_np = img.permute(1, 2, 0).numpy()
        # Denormalize for visualization
        img_np = np.clip((img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])), 0, 1)
        
        axs[i, 0].imshow(img_np)
        axs[i, 0].set_title('Ground Truth')
        
        boxes = target['boxes'].numpy()
        masks = target['masks'].numpy()
        
        # Draw ground truth boxes and masks
        for box in boxes:
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            rect = plt.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='g', facecolor='none')
            axs[i, 0].add_patch(rect)
        
        for mask in masks:
            masked = np.ma.masked_where(mask == 0, mask)
            axs[i, 0].imshow(masked, alpha=0.5, cmap='jet', interpolation='none')
        
        # Get predictions
        with torch.no_grad():
            model_input = img.unsqueeze(0).to(device)
            prediction = model(model_input)[0]
        
        # Draw predictions on second column
        axs[i, 1].imshow(img_np)
        axs[i, 1].set_title('Predictions')
        
        pred_boxes = prediction['boxes'].cpu().numpy()
        pred_masks = prediction['masks'].cpu().numpy()
        pred_scores = prediction['scores'].cpu().numpy()
        
        # Only show predictions with score > 0.5
        for j, (box, mask, score) in enumerate(zip(pred_boxes, pred_masks, pred_scores)):
            if score > 0.8:
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1
                rect = plt.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='r', facecolor='none')
                axs[i, 1].add_patch(rect)
                axs[i, 1].text(x1, y1, f"{score:.2f}", bbox=dict(facecolor='red', alpha=0.5))
                
                # Handle case for multiple masks
                mask_to_show = mask[0] if mask.shape[0] == 1 else mask
                masked = np.ma.masked_where(mask_to_show < 0.5, mask_to_show)
                axs[i, 1].imshow(masked, alpha=0.5, cmap='jet', interpolation='none')
    
    plt.tight_layout()
    plt.savefig('carton_predictions.png')
    plt.close()
    print(f"Visualization saved to carton_predictions.png")
