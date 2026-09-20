import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from Model_Save import save_model_with_timestamp
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def one_phase_nn(X_train_scaled, X_test_scaled, y_grip_train, y_grip_test, grip_categories):
    """
    Train a One-Phase Neural Network and plot its confusion matrix.
    """
    # Creation of multi-output network
    input_layer = Input(shape=(4,))

    # Shared hidden layers
    hidden1 = Dense(64, activation='relu')(input_layer)
    hidden1 = Dropout(0.3)(hidden1)
    hidden2 = Dense(32, activation='relu')(hidden1)
    hidden2 = Dropout(0.3)(hidden2)

    # Single output head for grip classification only
    grip_output = Dense(4, activation='softmax', name='grip_output')(hidden2)

    # Build and train NN
    model = Model(inputs=input_layer, outputs=grip_output)

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print("Model Architecture:")
    model.summary()

    print("Training One-Phase Neural Network (Grip Classification Only)...")
    history = model.fit(
        X_train_scaled,
        y_grip_train,
        validation_data=(X_test_scaled, y_grip_test),
        epochs=100,
        batch_size=32,
        verbose=0
    )

    # Make predictions
    predictions = model.predict(X_test_scaled)
    y_grip_pred = np.argmax(predictions, axis=1)

    grip_accuracy = accuracy_score(y_grip_test, y_grip_pred)

    # Print results
    print(f"Grip Classification Accuracy: {grip_accuracy:.4f}")
    print("\nGrip Classification Report:")
    print(classification_report(y_grip_test, y_grip_pred, target_names=grip_categories))

    # Save model
    save_model_with_timestamp(model, 'one_phase_nn_grip_model')

    # --- ADDED: Confusion Matrix Plotting ---
    cm = confusion_matrix(y_grip_test, y_grip_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=grip_categories, yticklabels=grip_categories)
    plt.title('One-Phase NN Grip Classification - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

    return model, grip_accuracy, y_grip_pred, history