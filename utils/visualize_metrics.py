import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

run = 1
metrics_path = f'evaluation_results/run_{run}'
evaluation_metrics_path = os.path.join(metrics_path, 'evaluation_metrics.json')
error_analysis_path = os.path.join(metrics_path, 'error_analysis.json')
confusion_matrix_path = os.path.join(metrics_path, 'confusion_matrix.json')
output_dir = os.path.join(metrics_path, 'plots')
os.makedirs(output_dir, exist_ok=True)

# Load JSON files
with open(evaluation_metrics_path) as f:
    eval_metrics = json.load(f)
with open(error_analysis_path) as f:
    error_analysis = json.load(f)
with open(confusion_matrix_path) as f:
    confusion_matrix = json.load(f)

# 1. Box and Mask mAP/mAR
def plot_map_mar(metrics, title_prefix, filename):
    labels = ['mAP', 'mAP@50', 'mAP@75', 'mAR@1', 'mAR@10', 'mAR@100']
    values = [
        metrics['map'], metrics['map_50'], metrics['map_75'],
        metrics['mar_1'], metrics['mar_10'], metrics['mar_100']
    ]
    plt.figure()
    plt.bar(labels, values, color='skyblue')
    plt.ylim(0, 1)
    plt.title(f"{title_prefix} Detection Metrics")
    plt.ylabel("Score")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

plot_map_mar(eval_metrics['box_metrics'], "Box", "box_metrics.png")
plot_map_mar(eval_metrics['mask_metrics'], "Mask", "mask_metrics.png")

# 2. Error analysis pie chart
def plot_error_analysis(data, filename):
    labels = ['False Positives', 'False Negatives', 'Correct']
    total = data['total_images']
    correct = total - data['images_with_false_positives'] - data['images_with_false_negatives']
    sizes = [data['images_with_false_positives'], data['images_with_false_negatives'], correct]
    plt.figure()
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=['red', 'orange', 'green'])
    plt.title("Image-Level Error Distribution")
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

plot_error_analysis(error_analysis, "error_analysis_pie.png")

# 3. Confusion matrix heatmap
def plot_confusion_heatmap(data, filename):
    tp = data['true_positives']
    fp = data['false_positives']
    fn = data['false_negatives']
    
    matrix = np.array([[tp, fn], [fp, 0]])  # TN is unknown
    labels = ['Actual Positive', 'Actual Negative']
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='YlGnBu',
                xticklabels=['Predicted Positive', 'Predicted Negative'],
                yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

plot_confusion_heatmap(confusion_matrix, "confusion_matrix_heatmap.png")
