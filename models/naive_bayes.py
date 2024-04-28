from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Function to train and evaluate Naive Bayes classifier
def train_and_evaluate_gnbc(train_features, train_labels, test_features, test_labels):
    print("Training and evaluating Gussian Naive Bayes classifier...")
    nbc = GaussianNB()
    nbc.fit(train_features, train_labels)
    predictions = nbc.predict(test_features)
    accuracy = accuracy_score(test_labels, predictions)
    f1 = f1_score(test_labels, predictions, average='weighted')
    cm = confusion_matrix(test_labels, predictions)
    return predictions, (accuracy, f1, cm)

# Function to train and evaluate Naive Bayes classifier
def train_and_evaluate_bnbc(train_features, train_labels, test_features, test_labels):
    print("Training and evaluating Bernoulli Naive Bayes classifier...")
    nbc = BernoulliNB()
    nbc.fit(train_features, train_labels)
    predictions = nbc.predict(test_features)
    accuracy = accuracy_score(test_labels, predictions)
    f1 = f1_score(test_labels, predictions, average='weighted')
    cm = confusion_matrix(test_labels, predictions)
    return predictions, (accuracy, f1, cm)