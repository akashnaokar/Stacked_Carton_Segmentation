import os
import torch
import numpy as np
import json
from tqdm import tqdm
from utils.iou_utils import calculate_iou

def generate_confusion_matrix(model, data_loader, config):
    """Generate confusion matrix for the model predictions"""
    # This is a simplified version - for a binary case like carton detection
    # We'll calculate TP, FP, FN based on IoU threshold
    
    model.eval()
    
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    iou_threshold = 0.5
    score_threshold = 0.7
    
    print("Generating confusion matrix...")
    try:
        with torch.no_grad():
            for images, targets in tqdm(data_loader):
                images = [image.to(config.device) for image in images]
                
                # Get predictions
                predictions = model(images)
                
                for i, (pred, target) in enumerate(zip(predictions, targets)):
                    # Filter predictions by score
                    high_scores = pred['scores'] > score_threshold
                    pred_boxes = pred['boxes'][high_scores].cpu().numpy()
                    
                    target_boxes = target['boxes'].cpu().numpy()
                    
                    # Calculate IoU for each prediction and ground truth pair
                    if len(pred_boxes) > 0 and len(target_boxes) > 0:
                        for p_box in pred_boxes:
                            # Check if this prediction matches any ground truth
                            matched = False
                            for t_box in target_boxes:
                                iou = calculate_iou(p_box, t_box)
                                if iou > iou_threshold:
                                    true_positives += 1
                                    matched = True
                                    break
                            if not matched:
                                false_positives += 1
                                
                        # Count false negatives - ground truths that weren't matched
                        for t_box in target_boxes:
                            matched = False
                            for p_box in pred_boxes:
                                iou = calculate_iou(p_box, t_box)
                                if iou > iou_threshold:
                                    matched = True
                                    break
                            if not matched:
                                false_negatives += 1
                    else:
                        # If no predictions but there are targets
                        if len(target_boxes) > 0:
                            false_negatives += len(target_boxes)
                        # If predictions but no targets
                        if len(pred_boxes) > 0:
                            false_positives += len(pred_boxes)
    except Exception as e:
        print(f"Error generating confusion matrix: {e}")
    
    if (true_positives + false_positives) > 0:
        precision = true_positives / (true_positives + false_positives)
    else:
        precision = 0
        
    if (true_positives + false_negatives) > 0:
        recall = true_positives / (true_positives + false_negatives)
    else:
        recall = 0
        
    if (precision + recall) > 0:
        f1_score = 2 * precision * recall / (precision + recall)
    else:
        f1_score = 0
    
    print(f"\nConfusion Matrix Results (IoU threshold: {iou_threshold}, Score threshold: {score_threshold}):")
    print(f"True Positives: {true_positives}")
    print(f"False Positives: {false_positives}")
    print(f"False Negatives: {false_negatives}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1_score:.4f}")
    
    # Save results
    confusion_results = {
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'iou_threshold': iou_threshold,
        'score_threshold': score_threshold
    }
    
    with open(os.path.join(config.results_dir, 'confusion_matrix.json'), 'w') as f:
        json.dump(confusion_results, f, indent=4)
    
    return confusion_results

def analyze_prediction_errors(model, data_loader, config):
    """Analyze common prediction errors"""
    model.eval()
    
    # Statistics to track
    total_images = 0
    images_with_fp = 0
    images_with_fn = 0
    small_object_fn = 0  # False negatives for small objects
    large_object_fn = 0  # False negatives for large objects
    edge_object_fn = 0   # False negatives for objects near edges
    
    iou_threshold = 0.5
    score_threshold = 0.7
    
    # Size thresholds (in pixels)
    small_threshold = 1000  # Area in pixels
    
    # Edge threshold (in pixels from image border)
    edge_threshold = 20
    
    print("Analyzing prediction errors...")
    try:
        with torch.no_grad():
            for images, targets in tqdm(data_loader):
                total_images += len(images)
                
                images = [image.to(config.device) for image in images]
                predictions = model(images)
                
                for i, (image, pred, target) in enumerate(zip(images, predictions, targets)):
                    image_height, image_width = image.shape[1:3]
                    
                    # Filter predictions by score
                    high_scores = pred['scores'] > score_threshold
                    pred_boxes = pred['boxes'][high_scores].cpu().numpy()
                    
                    target_boxes = target['boxes'].cpu().numpy()
                    
                    # Track if this image has false positives or negatives
                    has_fp = False
                    has_fn = False
                    
                    # Check for false positives
                    if len(pred_boxes) > 0:
                        for p_box in pred_boxes:
                            matched = False
                            for t_box in target_boxes:
                                iou = calculate_iou(p_box, t_box)
                                if iou > iou_threshold:
                                    matched = True
                                    break
                            if not matched:
                                has_fp = True
                    
                    # Check for false negatives and analyze them
                    if len(target_boxes) > 0:
                        for t_idx, t_box in enumerate(target_boxes):
                            matched = False
                            for p_box in pred_boxes:
                                iou = calculate_iou(p_box, t_box)
                                if iou > iou_threshold:
                                    matched = True
                                    break
                            
                            if not matched:
                                has_fn = True
                                
                                # Analyze the false negative
                                x1, y1, x2, y2 = t_box
                                area = (x2 - x1) * (y2 - y1)
                                
                                # Check if it's a small object
                                if area < small_threshold:
                                    small_object_fn += 1
                                else:
                                    large_object_fn += 1
                                
                                # Check if it's near the edge
                                if (x1 < edge_threshold or y1 < edge_threshold or 
                                    x2 > image_width - edge_threshold or y2 > image_height - edge_threshold):
                                    edge_object_fn += 1
                    
                    if has_fp:
                        images_with_fp += 1
                    if has_fn:
                        images_with_fn += 1
    except Exception as e:
        print(f"Error analyzing prediction errors: {e}")
    
    # Calculate percentages
    if total_images > 0:
        fp_percent = (images_with_fp / total_images) * 100
        fn_percent = (images_with_fn / total_images) * 100
    else:
        fp_percent = fn_percent = 0
    
    total_fn = small_object_fn + large_object_fn
    if total_fn > 0:
        small_fn_percent = (small_object_fn / total_fn) * 100
        large_fn_percent = (large_object_fn / total_fn) * 100
        edge_fn_percent = (edge_object_fn / total_fn) * 100
    else:
        small_fn_percent = large_fn_percent = edge_fn_percent = 0
    
    print("\nError Analysis:")
    print(f"Total images analyzed: {total_images}")
    print(f"Images with false positives: {images_with_fp} ({fp_percent:.1f}%)")
    print(f"Images with false negatives: {images_with_fn} ({fn_percent:.1f}%)")
    print("\nFalse Negative Analysis:")
    print(f"Small object misses: {small_object_fn} ({small_fn_percent:.1f}% of FNs)")
    print(f"Large object misses: {large_object_fn} ({large_fn_percent:.1f}% of FNs)")
    print(f"Edge object misses: {edge_object_fn} ({edge_fn_percent:.1f}% of FNs)")
    
    # Save results
    error_analysis = {
        'total_images': total_images,
        'images_with_false_positives': images_with_fp,
        'images_with_false_negatives': images_with_fn,
        'small_object_false_negatives': small_object_fn,
        'large_object_false_negatives': large_object_fn,
        'edge_object_false_negatives': edge_object_fn,
        'fp_percentage': fp_percent,
        'fn_percentage': fn_percent,
        'small_fn_percentage': small_fn_percent,
        'large_fn_percentage': large_fn_percent,
        'edge_fn_percentage': edge_fn_percent
    }
    
    with open(os.path.join(config.results_dir, 'error_analysis.json'), 'w') as f:
        json.dump(error_analysis, f, indent=4)
    
    return error_analysis