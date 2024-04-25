import os
from tqdm import tqdm
import torch
import numpy as np

# Function to extract features using the pre-trained model
def extract_features(data_loader, model, save_dir='features'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    features_file = os.path.join(save_dir, 'features.npy')
    labels_file = os.path.join(save_dir, 'labels.npy')

    if os.path.exists(features_file) and os.path.exists(labels_file):
        print("Loading features and labels...")
        features = np.load(features_file)
        labels = np.load(labels_file)
    else:
        print("Extracting features...")
        features = []
        labels = []
        with torch.no_grad():
            for images, targets in tqdm(data_loader):
                features.append(model(images).squeeze().numpy())
                labels.append(targets.numpy())
        features = np.concatenate(features)
        labels = np.concatenate(labels)
        np.save(features_file, features)
        np.save(labels_file, labels)
    return features, labels