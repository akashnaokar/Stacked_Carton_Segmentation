import os
import torch
import json
from tqdm import tqdm
from datetime import datetime
from torchmetrics.detection import MeanAveragePrecision

def run_sequential_evaluation(config, get_model_instance, get_maskrcnn_dataloader):
    """Run evaluation in sequential mode to avoid multiprocessing issues"""
    print("Running sequential evaluation...")
    
    # Load model
    num_classes = 2
    model = get_model_instance(num_classes)
    checkpoint = torch.load(config.model_path, map_location=config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(config.device)
    model.eval()
    
    # Get dataset directly
    _, val_dataset = get_maskrcnn_dataloader(
        data_dir=config.data_dir,
        max_samples=config.max_samples,
        train_ratio=config.train_ratio,
        batch_size=1,  # Not used in sequential mode
        num_workers=0,
        seed=config.seed,
        return_dataset=True  # Add this parameter to your dataloader function
    )
    
    # Initialize metrics
    metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox')
    mask_metric = MeanAveragePrecision(box_format='xyxy', iou_type='segm')
    
    # Evaluation loop
    all_predictions = []
    all_targets = []
    
    print(f"Evaluating {len(val_dataset)} samples...")
    with torch.no_grad():
        for idx in tqdm(range(len(val_dataset))):
            try:
                # Get sample
                img, target = val_dataset[idx]
                img = img.unsqueeze(0).to(config.device)  # Add batch dimension
                
                # Get prediction
                prediction = model(img)[0]
                
                # Format prediction and target
                pred_dict = {
                    'boxes': prediction['boxes'].cpu(),
                    'scores': prediction['scores'].cpu(),
                    'labels': prediction['labels'].cpu(),
                    'masks': prediction['masks'].cpu()
                }
                
                target_dict = {
                    'boxes': target['boxes'],
                    'labels': target['labels'],
                    'masks': target['masks']
                }
                
                all_predictions.append(pred_dict)
                all_targets.append(target_dict)
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                continue
    
    # Calculate metrics
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
        
        return results
    else:
        print("No predictions collected. Could not calculate metrics.")
        return None