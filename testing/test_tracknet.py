import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = 'test_model/model_last.pt'
model = torch.load(model_path, map_location=device)
model.eval()

# Caricare i dati di valutazione (presumo immagini)
data_dir = 'images'
dataset = ImageFolder(data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Liste per le previsioni e i veri target
all_predictions = []
all_targets = []

with torch.no_grad():
    for images, labels in dataloader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_predictions.append(predicted.item())
        all_targets.append(labels.item())

accuracy = accuracy_score(all_targets, all_predictions)
classification_rep = classification_report(all_targets, all_predictions)

print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(classification_rep)

for i in range(len(all_predictions)):
    print(f'Predicted: {all_predictions[i]}, True: {all_targets[i]}')
