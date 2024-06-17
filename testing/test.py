import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc, f1_score, precision_score, recall_score
import torch
from ultralytics import YOLO
from PIL import Image


test_images_path = 'images/' 
output_images_path = 'results/images/'

print(test_images_path)

os.makedirs(output_images_path, exist_ok=True)

model = YOLO('../models/yolo_player_model.pt')

def load_images(test_images_path):
    images = []
    for filename in os.listdir(test_images_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(test_images_path, filename)
            img = Image.open(img_path)
            images.append(img)
    return images

def predict(model, images):
    predictions = []
    for img in images:
        result = model.predict(img)
        predictions.append(result)
    return predictions

def calculate_metrics(true_labels, predictions):
    y_true = np.concatenate(true_labels)
    y_pred = np.concatenate(predictions)
    
    # Confusion Matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    # Precision, Recall, F1 Score
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    # Precision-Recall Curve
    precision_values, recall_values, _ = precision_recall_curve(y_true, y_pred)
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    return conf_matrix, precision, recall, f1, precision_values, recall_values, fpr, tpr, roc_auc

def save_curves(precision_values, recall_values, fpr, tpr, roc_auc, output_images_path):
    # Precision-Recall Curve
    plt.figure()
    plt.plot(recall_values, precision_values, marker='.')
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig(os.path.join(output_images_path, 'precision_recall_curve.png'))
    plt.close()
    
    # ROC Curve
    plt.figure()
    plt.plot(fpr, tpr, marker='.')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.savefig(os.path.join(output_images_path, 'roc_curve.png'))
    plt.close()

images, true_labels = load_images_and_annotations(test_images_path)
predictions = predict(model, images)
conf_matrix, precision, recall, f1, precision_values, recall_values, fpr, tpr, roc_auc = calculate_metrics(true_labels, predictions)

print('Confusion Matrix:')
print(conf_matrix)
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'ROC AUC: {roc_auc:.4f}')

save_curves(precision_values, recall_values, fpr, tpr, roc_auc, output_images_path)
