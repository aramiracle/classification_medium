import pandas as pd

from utils.load_dataset import load_dataset
from utils.load_model import load_model
from utils.extract_features import extract_features
from utils.print_results import print_results

from models.knn import find_best_k, train_and_evaluate_knn
from models.logistic_regression import train_and_evaluate_logistic_regression
from models.decision_tree import train_and_evaluate_decision_tree
from models.random_forest import train_and_evaluate_random_forest
from models.svm import train_and_evaluate_svm
from models.naive_bayes import train_and_evaluate_gnbc, train_and_evaluate_bnbc


# Main function
def main():
    trainloader, testloader = load_dataset()
    model = load_model()
    train_features, train_labels = extract_features(trainloader, model)
    test_features, test_labels = extract_features(testloader, model)
    best_k = find_best_k(train_features, train_labels)
    knn_results = train_and_evaluate_knn(train_features, train_labels, test_features, test_labels, best_k)
    logistic_regression_results = train_and_evaluate_logistic_regression(train_features, train_labels, test_features, test_labels)
    decision_tree_results = train_and_evaluate_decision_tree(train_features, train_labels, test_features, test_labels)
    random_forest_results = train_and_evaluate_random_forest(train_features, train_labels, test_features, test_labels)
    svm_results = train_and_evaluate_svm(train_features, train_labels, test_features, test_labels)
    gnbc_results = train_and_evaluate_gnbc(train_features, train_labels, test_features, test_labels)
    bnbc_results = train_and_evaluate_bnbc(train_features, train_labels, test_features, test_labels)
    
    print_results("KNN", knn_results[1])
    print_results("Logistic Regression", logistic_regression_results[1])
    print_results("Decision Tree", decision_tree_results[1])
    print_results("Random Forest", random_forest_results[1])
    print_results("SVM", svm_results[1])
    print_results("GNBC", gnbc_results[1])
    print_results("BNBC", bnbc_results[1])

    df_results = pd.DataFrame({
        "Real classes" : test_labels,
        "KNN" : knn_results[0],
        "Logistic Regression" : logistic_regression_results[0],
        "Decision Tree" : decision_tree_results[0],
        "Random Forest" : random_forest_results[0],
        "SVM" : svm_results[0],
        "GNBC" : gnbc_results[0],
        "BNBC" : bnbc_results[0]
    })

    df_results.to_csv('results_labels.csv')

if __name__ == "__main__":
    main()
