import os
import torch
import json
import numpy as np
from tqdm import tqdm
from datetime import datetime
from torchmetrics.detection import MeanAveragePrecision

# Import project modules
from model import get_model_instance
from data_loader import get_maskrcnn_dataloader
from utils.visualization import visualize_predictions
from utils.evaluation_analysis import generate_confusion_matrix, analyze_prediction_errors
from utils.sequential_evaluation import run_sequential_evaluation

class EvaluationConfig:
    # Paths
    run_no = 1 # change for selecting run
    data_dir = 'carton_data'
    model_path = os.path.join('checkpoints',f'run_{run_no}', 'best_model.pth')
    results_dir = os.path.join('evaluation_results', f'run_{run_no}')
    
    # Evaluation parameters
    batch_size = 2  # Reduced batch size to avoid memory issues
    num_workers = 2  # Disable multiprocessing to avoid IPC issues
    max_samples = 800  # Use all available validation samples
    visualization_samples = 10  # Number of samples to visualize
    
    # Device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Dataset
    train_ratio = 0.8  # Same as training to ensure consistent validation set
    seed = 42  # Use the same seed as training

def evaluate_model():
    config = EvaluationConfig()
    
    # Create results directory
    os.makedirs(config.results_dir, exist_ok=True)
    
    print(f"Using device: {config.device}")
    
    # Initialize multiprocessing method to avoid issues
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    # Load data
    print("Loading data...")
    _, val_dataloader = get_maskrcnn_dataloader(
        data_dir=config.data_dir,
        max_samples=config.max_samples,
        train_ratio=config.train_ratio,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        seed=config.seed
    )
    print(f"Validation samples: {len(val_dataloader.dataset)}")
    
    # Initialize model
    num_classes = 2  # Background + Carton
    model = get_model_instance(num_classes)
    
    # Load trained weights
    print(f"Loading model from {config.model_path}")
    checkpoint = torch.load(config.model_path, map_location=config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(config.device)
    model.eval()
    
    print(f"Model trained for {checkpoint['epoch'] + 1} epochs")
    print(f"Best validation loss: {checkpoint['val_loss']:.6f}")
    
    # Initialize metrics
    metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox')
    mask_metric = MeanAveragePrecision(box_format='xyxy', iou_type='segm')
    
    # Evaluation loop
    all_predictions = []
    all_targets = []
    
    print("Running evaluation...")
    try:
        with torch.no_grad():
            for images, targets in tqdm(val_dataloader):
                images = [image.to(config.device) for image in images]
                
                # Get predictions
                predictions = model(images)
                
                # Convert predictions and targets to the format expected by MeanAveragePrecision
                for i, (pred, target) in enumerate(zip(predictions, targets)):
                    pred_boxes = pred['boxes'].cpu()
                    pred_scores = pred['scores'].cpu()
                    pred_labels = pred['labels'].cpu()
                    pred_masks = (pred['masks'].cpu() > 0.5).squeeze(1).to(torch.uint8)
                    
                    # Format predictions
                    prediction = {
                        'boxes': pred_boxes,
                        'scores': pred_scores,
                        'labels': pred_labels,
                        'masks': pred_masks
                    }
                    
                    # Format target
                    target_dict = {
                        'boxes': target['boxes'],
                        'labels': target['labels'],
                        'masks': (target['masks'] > 0.5).to(torch.uint8)
                    }
                    
                    all_predictions.append(prediction)
                    all_targets.append(target_dict)
    except Exception as e:
        print(f"Error during evaluation: {e}")
        # If error occurs, try to continue with collected data
        print(f"Continuing with {len(all_predictions)} successful evaluations...")
    
    # Calculate metrics if we have collected predictions
    if len(all_predictions) > 0:
        metric.update(all_predictions, all_targets)
        mask_metric.update(all_predictions, all_targets)
        
        # Get results
        box_results = metric.compute()
        mask_results = mask_metric.compute()
        
        # Print and save results
        print("\nBounding Box Detection Results:")
        for k, v in box_results.items():
            if isinstance(v, torch.Tensor) and v.numel() == 1:
                print(f"{k}: {v.item():.4f}")
        
        print("\nMask Segmentation Results:")
        for k, v in mask_results.items():
            if isinstance(v, torch.Tensor) and v.numel() == 1:
                print(f"{k}: {v.item():.4f}")
        
        # Save metrics to file
        results = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_path': config.model_path,
            'epochs_trained': checkpoint['epoch'] + 1,
            'val_loss': checkpoint['val_loss'],
            'box_metrics': {k: v.item() if isinstance(v, torch.Tensor) and v.numel() == 1 else v.tolist() 
                            for k, v in box_results.items()},
            'mask_metrics': {k: v.item() if isinstance(v, torch.Tensor) and v.numel() == 1 else v.tolist() 
                             for k, v in mask_results.items()}
        }
        
        with open(os.path.join(config.results_dir, 'evaluation_metrics.json'), 'w') as f:
            json.dump(results, f, indent=4)
        
        # Visualize predictions
        visualize_predictions(model, val_dataloader, config)
        
        return results
    else:
        print("No predictions collected. Could not calculate metrics.")
        return None

if __name__ == "__main__":
    # Set multiprocessing method
    torch.multiprocessing.set_start_method('spawn', force=True)
    
    # Create config
    config = EvaluationConfig()
    
    # Try regular evaluation first
    try:
        print("Attempting standard evaluation...")
        results = evaluate_model()
    except Exception as e:
        print(f"Standard evaluation failed with error: {e}")
        print("Falling back to sequential evaluation...")
        # Modify the data_loader.py to add return_dataset=True parameter
        # and then use sequential evaluation
        results = run_sequential_evaluation(config, get_model_instance, get_maskrcnn_dataloader)
    
    if results:
        # Load the model again (just to be safe)
        num_classes = 2
        model = get_model_instance(num_classes)
        checkpoint = torch.load(config.model_path, map_location=config.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(config.device)
        model.eval()
        
        # Load validation dataloader with safer settings
        _, val_dataloader = get_maskrcnn_dataloader(
            data_dir=config.data_dir,
            max_samples=config.max_samples,
            train_ratio=config.train_ratio,
            batch_size=1,  # Single batch size for safety
            num_workers=0,  # No multiprocessing
            seed=config.seed
        )
        
        # Run additional analyses
        try:
            generate_confusion_matrix(model, val_dataloader, config)
            analyze_prediction_errors(model, val_dataloader, config)
        except Exception as e:
            print(f"Error during additional analyses: {e}")
        
        print(f"\nAll evaluation results saved to {config.results_dir}")
    else:
        print("Evaluation failed to produce results.")