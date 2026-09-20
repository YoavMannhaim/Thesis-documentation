# --- Updated Simple_Threshold_Test.py ---
import numpy as np
from sklearn.metrics import accuracy_score


def simple_threshold(X_test, y_test, grip_categories):
    """
    Find the best threshold for a simple threshold-based grip classification.
    Returns: accuracy, predictions, thresholds_list, accuracies_list
    """
    sensor_values = X_test[:, 0]  # assuming 1st sensor used for thresholding
    thresholds = np.linspace(sensor_values.min(), sensor_values.max(), 100)
    best_accuracy = 0
    best_threshold = None
    best_predictions = None

    accuracies_list = []  # Track all accuracies for plotting

    for threshold in thresholds:
        # Simple thresholding logic
        predictions = np.where(sensor_values > threshold, grip_categories[-1], grip_categories[0])
        pred_labels = np.array([grip_categories.index(p) for p in predictions])

        accuracy = accuracy_score(y_test, pred_labels)
        accuracies_list.append(accuracy)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
            best_predictions = pred_labels

    print(f"Optimal threshold found: {best_threshold:.3f} with accuracy: {best_accuracy:.4f}")
    # Return thresholds and accuracies lists for plotting
    return best_accuracy, best_predictions, thresholds, accuracies_list