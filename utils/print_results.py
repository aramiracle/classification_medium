def print_results(classifier_name, metrics):
    print(f"{classifier_name} Results:")
    print("Accuracy:", round(metrics[0], 4))
    print("F1 Score:", round(metrics[1], 4))
    print("Confusion Matrix:")
    print(metrics[2])
    print()