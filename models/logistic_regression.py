from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Function to train and evaluate Logistic Regression classifier
def train_and_evaluate_logistic_regression(train_features, train_labels, test_features, test_labels):
    print("Training and evaluating Logistic Regression classifier...")
    logistic_regression = LogisticRegression(max_iter=1000)
    logistic_regression.fit(train_features, train_labels)
    predictions = logistic_regression.predict(test_features)
    accuracy = accuracy_score(test_labels, predictions)
    f1 = f1_score(test_labels, predictions, average='weighted')
    cm = confusion_matrix(test_labels, predictions)
    return (accuracy, f1, cm)