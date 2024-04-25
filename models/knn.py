import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Function to find the best K for KNN classifier
def find_best_k(train_features, train_labels):
    print("Finding best K value for KNN classifier...")
    k_values = list(range(1, 11))
    accuracies = []
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, train_features, train_labels, cv=5)
        accuracies.append(scores.mean())
    best_k = k_values[np.argmax(accuracies)]
    return best_k

# Function to train and evaluate the best KNN classifier
def train_and_evaluate_knn(train_features, train_labels, test_features, test_labels, best_k):
    print("Training and evaluating KNN classifier...")
    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(train_features, train_labels)
    predictions = knn.predict(test_features)
    accuracy = accuracy_score(test_labels, predictions)
    f1 = f1_score(test_labels, predictions, average='weighted')
    cm = confusion_matrix(test_labels, predictions)
    return (accuracy, f1, cm)