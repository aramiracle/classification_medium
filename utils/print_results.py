def print_results(classifier_name, metrics):
    print(f"{classifier_name} Results:")
    print("Accuracy:", metrics[0])
    print("F1 Score:", metrics[1])
    print("Confusion Matrix:")
    print(metrics[2])
    print()