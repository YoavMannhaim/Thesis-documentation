from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import learning_curve
from Model_Save import save_model_with_timestamp
import numpy as np

def decision_tree(X_train, X_test, y_grip_train, y_grip_test, grip_categories):
    # Grip classification
    grip_model = DecisionTreeClassifier(max_depth=10, min_samples_split=20)
    grip_model.fit(X_train, y_grip_train)
    y_grip_pred = grip_model.predict(X_test)
    grip_accuracy = accuracy_score(y_grip_test, y_grip_pred)

    # Print results
    print("Grip Classification Accuracy:", grip_accuracy)
    print("Grip Feature Importance:", grip_model.feature_importances_)
    print("Grip Classification Report:")
    print(classification_report(y_grip_test, y_grip_pred, target_names=grip_categories))

    # Save model
    save_model_with_timestamp(grip_model, 'grip_decision_tree_model')

    # Compute learning curve
    train_sizes, train_scores, val_scores = learning_curve(
        grip_model,
        X_train, y_grip_train,
        cv=5,
        scoring='accuracy',
        train_sizes=np.linspace(0.1, 1.0, 10),
        n_jobs=-1,
        shuffle=True,
        random_state=42
    )

    # Take mean scores across folds
    train_scores_mean = np.mean(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)

    return grip_model, grip_accuracy, y_grip_pred, train_sizes, train_scores_mean, val_scores_mean
