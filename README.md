# Image Classification Project

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

## Classification Methods Used:

1. **K-Nearest Neighbors (KNN)**
   - **Methodology**: KNN is a non-parametric, instance-based learning algorithm. It predicts the class of a sample by finding the majority class among its K nearest neighbors in the feature space. The distance metric, often Euclidean distance, is used to measure the similarity between instances.
   - **Parameters Tuned**: The main parameter to tune is the number of neighbors, K.
   - **Implementation**: `models/knn.py`

2. **Logistic Regression**
   - **Methodology**: Logistic Regression is a linear classification algorithm that models the probability of the default class via a logistic function. It estimates the parameters of the logistic function using maximum likelihood estimation. Despite its name, it is used for binary classification and can be extended to multiclass classification using techniques like one-vs-rest or softmax.
   - **Parameters Tuned**: Regularization strength (C) can be tuned to control overfitting.
   - **Implementation**: `models/logistic_regression.py`

3. **Decision Tree**
   - **Methodology**: Decision Trees recursively split the data into subsets based on features, aiming to maximize information gain or minimize impurity at each split. Each internal node represents a feature, each branch represents a decision based on that feature, and each leaf node represents a class label. Decision trees are prone to overfitting but can be regularized by limiting the tree's depth or the minimum samples required to split.
   - **Parameters Tuned**: Maximum depth of the tree and minimum samples required to split or create a leaf node.
   - **Implementation**: `models/decision_tree.py`

4. **Random Forest**
   - **Methodology**: Random Forest is an ensemble learning method that constructs multiple decision trees during training. It aims to reduce overfitting by averaging the predictions of individual trees. Each tree is trained on a bootstrapped subset of the data, and at each split, a random subset of features is considered. Random Forests are more robust than individual decision trees and less prone to overfitting.
   - **Parameters Tuned**: Number of trees in the forest, maximum depth of the trees, and minimum samples required to split a node.
   - **Implementation**: `models/random_forest.py`

5. **Support Vector Machine (SVM)**
   - **Methodology**: SVM constructs a hyperplane or set of hyperplanes in a high-dimensional space, which can be used for classification or regression. The objective is to maximize the margin between classes, with support vectors being the data points closest to the hyperplane(s). SVM can handle linear and non-linear classification through the use of different kernel functions.
   - **Parameters Tuned**: Regularization parameter (C) and choice of kernel (e.g., linear, polynomial, radial basis function).
   - **Implementation**: `models/svm.py`

6. **Gaussian Naive Bayes Classifier (GNBC)**
   - **Methodology**: Naive Bayes classifiers are probabilistic classifiers based on Bayes' theorem with the "naive" assumption of independence between features. GNBC assumes that features are continuous and normally distributed. It calculates the likelihood of observing each feature given each class and combines these probabilities with prior probabilities to make predictions.
   - **Implementation**: `models/naive_bayes.py`

7. **Bernoulli Naive Bayes Classifier (BNBC)**
   - **Methodology**: Bernoulli Naive Bayes Classifier is similar to GNBC but specifically designed for binary/Boolean features. It assumes that features are binary-valued (e.g., presence or absence of a feature). It calculates the likelihood of observing each feature given each class and combines these probabilities with prior probabilities to make predictions.
   - **Implementation**: `models/naive_bayes.py`

These methods provide a diverse set of approaches for classification tasks, each with its own strengths and weaknesses. Evaluation includes metrics such as accuracy, F1 score, and confusion matrix.

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

Finally there is a medium article you can read for deeper insight. This is a [link](https://medium.com/@a.r.amouzad.m/classic-machine-learning-part-3-4-classification-on-image-dataset-944ed3353d05) to story.