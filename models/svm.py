from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Function to train and evaluate SVM classifier
def train_and_evaluate_svm(train_features, train_labels, test_features, test_labels):
    print("Training and evaluating SVM classifier...")
    svm = SVC(kernel='rbf')
    svm.fit(train_features, train_labels)
    predictions = svm.predict(test_features)
    accuracy = accuracy_score(test_labels, predictions)
    f1 = f1_score(test_labels, predictions, average='weighted')
    cm = confusion_matrix(test_labels, predictions)
    return predictions, (accuracy, f1, cm)