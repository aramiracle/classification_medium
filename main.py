from utils.load_dataset import load_dataset
from utils.load_model import load_model
from utils.extract_features import extract_features
from utils.print_results import print_results

from models.knn import find_best_k, train_and_evaluate_knn
from models.logistic_regression import train_and_evaluate_logistic_regression
from models.decision_tree import train_and_evaluate_decision_tree
from models.random_forest import train_and_evaluate_random_forest
from models.svm import train_and_evaluate_svm
from models.naive_bayes import train_and_evaluate_nbc


# Main function
def main():
    trainloader, testloader = load_dataset()
    model = load_model()
    train_features, train_labels = extract_features(trainloader, model)
    test_features, test_labels = extract_features(testloader, model)
    best_k = find_best_k(train_features, train_labels)
    knn_metrics = train_and_evaluate_knn(train_features, train_labels, test_features, test_labels, best_k)
    logistic_regression_metrics = train_and_evaluate_logistic_regression(train_features, train_labels, test_features, test_labels)
    decision_tree_metrics = train_and_evaluate_decision_tree(train_features, train_labels, test_features, test_labels)
    random_forest_metrics = train_and_evaluate_random_forest(train_features, train_labels, test_features, test_labels)
    svm_metrics = train_and_evaluate_svm(train_features, train_labels, test_features, test_labels)
    nbc_metrics = train_and_evaluate_nbc(train_features, train_labels, test_features, test_labels)
    
    print_results("KNN", knn_metrics)
    print_results("Logistic Regression", logistic_regression_metrics)
    print_results("Decision Tree", decision_tree_metrics)
    print_results("Random Forest", random_forest_metrics)
    print_results("SVM", svm_metrics)
    print_results("NBC", nbc_metrics)

if __name__ == "__main__":
    main()
