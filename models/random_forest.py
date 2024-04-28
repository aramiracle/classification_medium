from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Function to train and evaluate Random Forest classifier
def train_and_evaluate_random_forest(train_features, train_labels, test_features, test_labels):
    print("Training and evaluating Random Forest classifier...")
    random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
    random_forest.fit(train_features, train_labels)
    predictions = random_forest.predict(test_features)
    accuracy = accuracy_score(test_labels, predictions)
    f1 = f1_score(test_labels, predictions, average='weighted')
    cm = confusion_matrix(test_labels, predictions)
    return predictions, (accuracy, f1, cm)