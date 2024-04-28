# Project README

## Introduction
This project is designed for training and evaluating various machine learning models on the CIFAR-10 dataset. It includes functionalities for loading the dataset, loading the pre-trained EfficientNetV2-S model, extracting features, training and evaluating multiple models such as K-Nearest Neighbors (KNN), Logistic Regression, Decision Tree, Random Forest, Support Vector Machine (SVM), Gaussian Naive Bayes Classifier (GNBC), and Bernoulli Naive Bayes Classifier (BNBC). Additionally, it provides a script to visualize the embeddings in 2D using t-distributed Stochastic Neighbor Embedding (t-SNE).

## Files
- **main.py**: This script serves as the entry point of the project. It orchestrates the training and evaluation of machine learning models and generates a CSV file containing the results.
- **utils/load_dataset.py**: Contains functions for loading the CIFAR-10 dataset.
- **utils/load_model.py**: Contains a function for loading the pre-trained EfficientNetV2-S model.
- **utils/extract_features.py**: Includes functions for extracting features from the dataset.
- **utils/print_results.py**: Provides a function for printing the evaluation results.
- **models/**:
  - **knn.py**: Implementation of K-Nearest Neighbors model along with functions for finding the best K and training/evaluating the model.
  - **logistic_regression.py**: Implementation of Logistic Regression model and functions for training/evaluating the model.
  - **decision_tree.py**: Implementation of Decision Tree model and functions for training/evaluating the model.
  - **random_forest.py**: Implementation of Random Forest model and functions for training/evaluating the model.
  - **svm.py**: Implementation of Support Vector Machine model and functions for training/evaluating the model.
  - **naive_bayes.py**: Implementation of Gaussian Naive Bayes Classifier and Bernoulli Naive Bayes Classifier along with functions for training/evaluating these models.
- **embeddings_tsne.py**: Script to perform dimensionality reduction using t-SNE and visualize embeddings in 2D with colored clusters.

## Usage
1. **Environment Setup**: Ensure Python 3.x is installed along with required dependencies. You can install dependencies via `pip install -r requirements.txt`.
   
2. **Data Preparation**: CIFAR-10 dataset will be automatically downloaded and prepared by the `load_dataset()` function in `utils/load_dataset.py`.
   
3. **Model Loading**: The pre-trained EfficientNetV2-S model will be loaded by the `load_model()` function in `utils/load_model.py`.
   
4. **Training and Evaluation**: Execute `main.py` to train and evaluate the machine learning models. This will generate a CSV file named `results_labels.csv` containing the evaluation results, including accuracy_score, f1_score, and confusion_matrix.

5. **Embedding Visualization**: Run `embeddings_tsne.py` to visualize the embeddings in 2D with colored clusters. The visualization will be saved as `embedding_with_clusters.png`.

## Evaluation Metrics
- **Accuracy Score**: The proportion of correctly classified samples.
- **F1 Score**: The harmonic mean of precision and recall, providing a balance between them.
- **Confusion Matrix**: A table showing the counts of true positive, true negative, false positive, and false negative predictions.

## Dependencies
- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- torch
- torchvision
- tqdm
