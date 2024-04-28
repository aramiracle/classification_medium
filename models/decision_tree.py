from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Function to train and evaluate Decision Tree classifier
def train_and_evaluate_decision_tree(train_features, train_labels, test_features, test_labels):
    print("Training and evaluating Decision Tree classifier...")
    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(train_features, train_labels)
    predictions = decision_tree.predict(test_features)
    accuracy = accuracy_score(test_labels, predictions)
    f1 = f1_score(test_labels, predictions, average='weighted')
    cm = confusion_matrix(test_labels, predictions)
    return predictions, (accuracy, f1, cm)