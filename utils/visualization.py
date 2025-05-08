import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def visualize_predictions(model, data_loader, config, num_samples=None):
    """Visualize model predictions on validation samples"""
    if num_samples is None:
        num_samples = config.visualization_samples
    
    model.eval()
    
    # Get dataset from dataloader
    dataset = data_loader.dataset
    # Handle if dataset is a Subset
    while hasattr(dataset, 'dataset'):
        dataset = dataset.dataset
    
    # Create a directory for visualizations
    vis_dir = os.path.join(config.results_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    print(f"\nGenerating {num_samples} visualization samples...")
    
    for i in range(min(num_samples, len(dataset))):
        try:
            # Get a random sample from the validation set
            sample_idx = torch.randint(0, len(dataset), (1,)).item()
            img, target = dataset[sample_idx]
            
            fig, axs = plt.subplots(1, 2, figsize=(16, 8))
            
            # Original image with ground truth
            img_np = img.permute(1, 2, 0).numpy()
            
            axs[0].imshow(img_np)
            axs[0].set_title('Ground Truth', fontsize=14)
            
            boxes = target['boxes'].numpy()
            masks = target['masks'].numpy()
            
            # Draw ground truth boxes and masks
            for box in boxes:
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1
                rect = plt.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='g', facecolor='none')
                axs[0].add_patch(rect)
            
            for mask in masks:
                masked = np.ma.masked_where(mask == 0, mask)
                axs[0].imshow(masked, alpha=0.5, cmap='jet', interpolation='none')
            
            # Get predictions
            with torch.no_grad():
                model_input = img.unsqueeze(0).to(config.device)
                prediction = model(model_input)[0]
            
            # Draw predictions on second column
            axs[1].imshow(img_np)
            axs[1].set_title('Model Predictions', fontsize=14)
            
            pred_boxes = prediction['boxes'].cpu().numpy()
            pred_masks = prediction['masks'].cpu().numpy()
            pred_scores = prediction['scores'].cpu().numpy()
            
            # Only show predictions with score > threshold
            threshold = 0.7
            for j, (box, mask, score) in enumerate(zip(pred_boxes, pred_masks, pred_scores)):
                if score > threshold:
                    x1, y1, x2, y2 = box
                    width = x2 - x1
                    height = y2 - y1
                    rect = plt.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='r', facecolor='none')
                    axs[1].add_patch(rect)
                    axs[1].text(x1, y1, f"{score:.2f}", bbox=dict(facecolor='red', alpha=0.5))
                    
                    # Handle case for multiple masks
                    mask_to_show = mask[0] if mask.shape[0] == 1 else mask
                    masked = np.ma.masked_where(mask_to_show < 0.5, mask_to_show)
                    axs[1].imshow(masked, alpha=0.5, cmap='jet', interpolation='none')
            
            # Add IoU and prediction count information
            num_gt = len(boxes)
            num_pred = len([s for s in pred_scores if s > threshold])
            axs[0].set_xlabel(f"Number of ground truth objects: {num_gt}", fontsize=12)
            axs[1].set_xlabel(f"Number of predictions (score > {threshold}): {num_pred}", fontsize=12)
            
            # Remove axis ticks for cleaner visualization
            for ax in axs:
                ax.set_xticks([])
                ax.set_yticks([])
            
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, f'prediction_sample_{i+1}.png'), dpi=150)
            plt.close()
        except Exception as e:
            print(f"Error generating visualization for sample {i+1}: {e}")
            plt.close('all')  # Close any open figures to prevent memory leak
            continue
        
    print(f"Visualizations saved to {vis_dir}")