import numpy as np
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt

# Load the embeddings
embeddings = np.load('features/features.npy')

# Load the DataFrame with labels
df = pd.read_csv('results_labels.csv')

# Extract labels and convert them to arrays
labels = df.values[:, 1:]  # Assuming the first column is index
categories = df.columns[1:]

# Apply t-SNE to reduce dimensionality to 2D
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

# Plot the embeddings in 2D with colored labels for each label column
plt.figure(figsize=(24, 16),dpi=600)
for i, category in enumerate(categories):
    plt.subplot(2, 4, i+1)
    for label_value in np.unique(labels[:, i]):
        plt.scatter(embeddings_2d[labels[:, i] == label_value, 0], 
                    embeddings_2d[labels[:, i] == label_value, 1], 
                    s=10, label=f'{category}={label_value}')
    plt.title(f't-SNE Visualization with {category}')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()
plt.tight_layout()
plt.savefig('embeddings_with_clusters.png')
