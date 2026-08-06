import numpy as np

def majority_classifier(y_train, X_test):
    """
    Predict the most frequent label in training data for all test samples.
    """
    # Find unique labels and their frequency counts
    unique_classes, counts = np.unique(y_train, return_counts=True)
    
    # Identify the class with the maximum count
    # np.argmax returns the first index of the maximum value if there are ties
    majority_class = unique_classes[np.argmax(counts)]
    
    # Create an array of the same length as X_test filled with the majority class
    predictions = np.full(len(X_test), fill_value=majority_class, dtype=int)
    
    return predictions