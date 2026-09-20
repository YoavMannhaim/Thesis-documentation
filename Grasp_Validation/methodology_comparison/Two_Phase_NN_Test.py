import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score ,classification_report
import keras_tuner as kt


class GripValidationSystem:
    def __init__(self):
        self.orientation_model = None
        self.grip_model = None
        self.scaler = StandardScaler()
        self.orientation_encoder = LabelEncoder()
        self.grip_encoder = LabelEncoder()

    def build_orientation_network(self, hp):
        model = keras.Sequential()
        model.add(layers.Input(shape=(4,)))
        # Tune number of neurons and activation
        model.add(layers.Dense(
            units=hp.Int('units_o1', 20, 128, step=1),
            activation=hp.Choice('activation_o1', ['relu', 'tanh'])
        ))
        model.add(layers.Dropout(hp.Float('dropout_o1', 0.1, 0.5, step=0.02)))
        model.add(layers.Dense(
            units=hp.Int('units_o2', 8, 128, step=1),
            activation=hp.Choice('activation_o2', ['relu', 'tanh'])
        ))
        model.add(layers.Dropout(hp.Float('dropout_o2', 0.0, 0.8, step=0.02)))
        model.add(layers.Dense(len(np.unique(self.orientation_encoder.transform(self.orientation_encoder.classes_))),
                               activation='softmax'))

        model.compile(
            optimizer=keras.optimizers.Adam(hp.Float('learning_rate_o', 1e-4, 1e-2, sampling='log')),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def build_grip_network(self, hp, input_dim):
        model = keras.Sequential()
        model.add(layers.Input(shape=(input_dim,)))
        model.add(layers.Dense(
            units=hp.Int('units_g1', 20, 128, step=1),
            activation=hp.Choice('activation_g1', ['relu', 'tanh'])
        ))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(hp.Float('dropout_g1', 0.1, 0.5, step=0.02)))
        model.add(layers.Dense(
            units=hp.Int('units_g2', 12, 64, step=1),
            activation=hp.Choice('activation_g2', ['relu', 'tanh'])
        ))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(hp.Float('dropout_g2', 0.0, 0.8, step=0.02)))
        model.add(layers.Dense(len(np.unique(self.grip_encoder.transform(self.grip_encoder.classes_))),
                               activation='softmax'))

        model.compile(
            optimizer=keras.optimizers.Adam(hp.Float('learning_rate_g', 1e-4, 1e-2, sampling='log')),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def prepare_data(self, data):
        X_sensors = data[['sensor_1', 'sensor_2', 'sensor_3', 'sensor_4']].values
        X_sensors_scaled = self.scaler.fit_transform(X_sensors)
        y_orientation = self.orientation_encoder.fit_transform(data['orientation'])
        y_grip = self.grip_encoder.fit_transform(data['grip_quality'])
        return X_sensors_scaled, y_orientation, y_grip

    def tune_orientation_model(self, X_train, y_orient_train, validation_data, max_epochs=50):
        tuner = kt.RandomSearch(
            self.build_orientation_network,
            objective='val_accuracy',
            max_trials=6,
            executions_per_trial=1,
            directory='orientation_tuning',
            project_name='orientation_model'
        )
        tuner.search(
            X_train, y_orient_train,
            epochs=max_epochs,
            validation_data=validation_data,
            callbacks=[keras.callbacks.EarlyStopping(patience=10)]
        )
        best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
        print("Best hyperparameters for orientation model:")
        print(best_hp.values)
        best_model = tuner.hypermodel.build(best_hp)
        return best_model

    def tune_grip_model(self, X_train_combined, y_grip_train, validation_data, input_dim, max_epochs=50):
        def build_grip(hp):
            return self.build_grip_network(hp, input_dim)

        tuner = kt.RandomSearch(
            build_grip,
            objective='val_accuracy',
            max_trials=6,
            executions_per_trial=1,
            directory='grip_tuning',
            project_name='grip_model'
        )
        tuner.search(
            X_train_combined, y_grip_train,
            epochs=max_epochs,
            validation_data=validation_data,
            callbacks=[keras.callbacks.EarlyStopping(patience=10)]
        )
        best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
        print("Best hyperparameters for grip model:")
        print(best_hp.values)
        best_model = tuner.hypermodel.build(best_hp)
        return best_model

    def train_system_with_tuning(self, test_size=0.2, validation_split=0.2, epochs=100):
        print("Preparing data...")
        data = generate_sample_data(2000)
        X_sensors, y_orientation, y_grip = self.prepare_data(data)

        # Split data
        X_train, X_test, y_orient_train, y_orient_test, y_grip_train, y_grip_test = train_test_split(
            X_sensors, y_orientation, y_grip, test_size=test_size, random_state=42, stratify=y_grip)

        # Further split training for validation
        val_split = int(len(X_train) * (1 - validation_split))
        X_train_sub, X_val = X_train[:val_split], X_train[val_split:]
        y_orient_train_sub, y_orient_val = y_orient_train[:val_split], y_orient_train[val_split:]
        y_grip_train_sub, y_grip_val = y_grip_train[:val_split], y_grip_train[val_split:]

        # Tune orientation model
        print("\nTuning orientation detection model...")
        self.orientation_model = self.tune_orientation_model(
            X_train_sub, y_orient_train_sub, validation_data=(X_val, y_orient_val), max_epochs=epochs)

        # Train orientation model with the best HP on full train for stability
        # Correction: Capture the history object
        orient_history = self.orientation_model.fit(
            X_train, y_orient_train,
            epochs=epochs,
            validation_split=validation_split,
            callbacks=[keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)],
            verbose=1)

        # Predict orientation probabilities for grip model input
        orient_pred_train = self.orientation_model.predict(X_train)
        orient_pred_test = self.orientation_model.predict(X_test)

        X_train_combined = np.concatenate([X_train, orient_pred_train], axis=1)
        X_test_combined = np.concatenate([X_test, orient_pred_test], axis=1)

        # Tune grip model
        print("\nTuning grip quality model...")
        self.grip_model = self.tune_grip_model(
            X_train_combined[:val_split], y_grip_train[:val_split],
            validation_data=(X_train_combined[val_split:], y_grip_train[val_split:]),
            input_dim=X_train_combined.shape[1],
            max_epochs=epochs)

        # Train grip model with the best HP on full train set
        # Correction: Capture the history object
        grip_history = self.grip_model.fit(
            X_train_combined, y_grip_train,
            epochs=epochs,
            validation_split=validation_split,
            callbacks=[keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)],
            verbose=1)

        # Evaluation
        print("\nEvaluating orientation model...")
        orient_test_pred = self.orientation_model.predict(X_test)
        orient_test_pred_classes = np.argmax(orient_test_pred, axis=1)

        print(classification_report(y_orient_test, orient_test_pred_classes,
                                    target_names=self.orientation_encoder.classes_))

        print("\nEvaluating grip quality model...")
        grip_test_pred = self.grip_model.predict(X_test_combined)
        grip_test_pred_classes = np.argmax(grip_test_pred, axis=1)

        print(classification_report(y_grip_test, grip_test_pred_classes,
                                    target_names=self.grip_encoder.classes_))

        # Correction: Return history and test data for plotting
        return {
            'best_orientation_model': self.orientation_model,
            'best_grip_model': self.grip_model,
            'test_grip': y_grip_test,
            'prediction_grip': grip_test_pred_classes,
            'test_orientation': y_orient_test,
            'prediction_orientation': orient_test_pred_classes,
            'orientation_history': orient_history,
            'grip_history': grip_history,
            'test_data': (X_test, X_test_combined, y_orient_test, y_grip_test),
            'predictions': (orient_test_pred_classes, grip_test_pred_classes)
        }

    def predict_grip_quality(self, sensor_readings):
        """
        Predict grip quality for new sensor readings.
        sensor_readings: array-like with 4 values from photo-resistors
        """
        if self.orientation_model is None or self.grip_model is None:
            raise ValueError("Models must be trained before prediction")

        # Preprocess sensor readings
        sensor_readings = np.array(sensor_readings).reshape(1, -1)
        sensor_readings_scaled = self.scaler.transform(sensor_readings)

        # Step 1: Predict orientation probabilities
        orientation_probs = self.orientation_model.predict(sensor_readings_scaled, verbose=0)
        predicted_orientation = np.argmax(orientation_probs, axis=1)[0]

        # Step 2: Predict grip quality using sensor readings + orientation probabilities
        combined_features = np.concatenate([sensor_readings_scaled, orientation_probs], axis=1)
        assert combined_features.shape[1] == self.grip_model.input_shape[1], \
            f"Input feature mismatch: got {combined_features.shape[1]}, expected {self.grip_model.input_shape[1]}"

        grip_probs = self.grip_model.predict(combined_features, verbose=0)
        predicted_grip = np.argmax(grip_probs, axis=1)

        return {
            'orientation': self.orientation_encoder.inverse_transform([predicted_orientation]),
            'orientation_confidence': float(np.max(orientation_probs)),
            'grip_quality': self.grip_encoder.inverse_transform([predicted_grip]),
            'grip_confidence': float(np.max(grip_probs)),
            'grip_probabilities': dict(zip(self.grip_encoder.classes_, grip_probs))
        }

    def plot_training_history(self, training_results):
        """
        Plot training history for both models
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Orientation model plots
        orient_history = training_results['orientation_history']

        axes[0, 0].plot(orient_history.history['loss'], label='Training Loss')
        axes[0, 0].plot(orient_history.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Orientation Model - Loss')
        axes[0, 0].legend()

        axes[0, 1].plot(orient_history.history['accuracy'], label='Training Accuracy')
        axes[0, 1].plot(orient_history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 1].set_title('Orientation Model - Accuracy')
        axes[0, 1].legend()

        # Grip model plots
        grip_history = training_results['grip_history']

        axes[1, 0].plot(grip_history.history['loss'], label='Training Loss')
        axes[1, 0].plot(grip_history.history['val_loss'], label='Validation Loss')
        axes[1, 0].set_title('Grip Quality Model - Loss')
        axes[1, 0].legend()

        axes[1, 1].plot(grip_history.history['accuracy'], label='Training Accuracy')
        axes[1, 1].plot(grip_history.history['val_accuracy'], label='Validation Accuracy')
        axes[1, 1].set_title('Grip Quality Model - Accuracy')
        axes[1, 1].legend()

        plt.tight_layout()
        plt.savefig("results_figures/two_phase_training_history.png", dpi=150)
        plt.close()

    def plot_confusion_matrices(self, training_results):
        """
        Plot confusion matrices for both models
        """
        X_test, X_test_combined, y_orient_test, y_grip_test = training_results['test_data']
        orient_pred, grip_pred = training_results['predictions']

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Orientation confusion matrix
        cm_orient = confusion_matrix(y_orient_test, orient_pred)
        sns.heatmap(cm_orient, annot=True, fmt='d', ax=axes[0],
                    xticklabels=self.orientation_encoder.classes_,
                    yticklabels=self.orientation_encoder.classes_)
        axes[0].set_title('Orientation Detection - Confusion Matrix')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')

        # Grip quality confusion matrix
        cm_grip = confusion_matrix(y_grip_test, grip_pred)
        sns.heatmap(cm_grip, annot=True, fmt='d', ax=axes[1],
                    xticklabels=self.grip_encoder.classes_,
                    yticklabels=self.grip_encoder.classes_)
        axes[1].set_title('Grip Quality Classification - Confusion Matrix')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('Actual')

        plt.tight_layout()
        plt.savefig('results_figures/two_phase_confusion_matrices.png', dpi=150)
        plt.close()


# Generating usage and data generation function
def generate_sample_data(n_samples=1000):
    """
    Generate sample data for demonstration purposes
    In real application, this would be replaced with actual sensor data
    """
    # List of data files
    # #files = ['No_Grip.xlsx', 'Spong_Bottom_Bad', 'Spong_Bottom_Semi', 'Spong_Bottom_Good', 'Spong_Top_Bad', 'Spong_Top_Semi', 'Spong_Top_Good', 'Spong_Side_Bad', 'Spong_Side_Semi', 'Spong_Side_Good']
    # Create data file

    # Create variables for df creation:
    df_dict = {}
    pointer = 1

    # Debugging
    #print("Check_1")

    # Create df list:

    # Bottom grip
    [df_dict, pointer] = create_df("data/No_Grip.xlsx", "data/Spong_Bottom_Bad.xlsx", "data/Spong_Bottom_Semi.xlsx", "data/Spong_Bottom_Good.xlsx", 'face_down',df_dict, pointer)

    # Debugging
    #print("Check_2")

    # Top grip
    [df_dict, pointer] = create_df("data/No_Grip.xlsx", "data/Spong_Top_Bad.xlsx", "data/Spong_Top_Semi.xlsx", "data/Spong_Top_Good.xlsx",'face_up', df_dict, pointer)

    # Debugging
    #print("Check_3")

    # Side grip
    [df_dict, pointer] = create_df("data/No_Grip.xlsx", "data/Spong_Side_Bad.xlsx", "data/Spong_Side_Semi.xlsx", "data/Spong_Side_Good.xlsx",'left_side', df_dict, pointer)

    # Debugging
    #print("Check_4")

    #print(df_dict)
    #print("Pointer")
    #print(pointer)

    # Connect all DataFrames
    all_data = pd.concat(df_dict.values(), ignore_index=True)

    # Save combine data as CSV and DAtaFrame for training
    all_data.to_csv('all_sponge_data.csv', index=False)
    result_df = all_data

    return result_df

def create_df(file_path_1, file_path_2, file_path_3, file_path_4, orientation,df_dict, pointer):


    # Creating and saving first df
    df = pd.read_excel(file_path_1, header=None, names=['sensor_1', 'sensor_2', 'sensor_3', 'sensor_4'])
    df['orientation'] = orientation
    df['grip_quality'] = 'no_grip'  # or pass as argument if variable
    df_dict[pointer] = df

    # Moving pointer forward
    pointer = pointer +1

    # Creating and saving seconde df
    df = pd.read_excel(file_path_2, header=None, names=['sensor_1', 'sensor_2', 'sensor_3', 'sensor_4'])
    df['orientation'] = orientation
    df['grip_quality'] = 'bad_grip'  # or pass as argument if variable
    df_dict[pointer] = df

    # Moving pointer forward
    pointer = pointer + 1

    # Creating and saving third df
    df = pd.read_excel(file_path_3, header=None, names=['sensor_1', 'sensor_2', 'sensor_3', 'sensor_4'])
    df['orientation'] = orientation
    df['grip_quality'] = 'semi_good_grip'  # or pass as argument if variable
    df_dict[pointer] = df

    # Moving pointer forward
    pointer = pointer + 1

    # Creating and saving fourth df
    df = pd.read_excel(file_path_4, header=None, names=['sensor_1', 'sensor_2', 'sensor_3', 'sensor_4'])
    df['orientation'] = orientation
    df['grip_quality'] = 'good_grip'  # or pass as argument if variable
    df_dict[pointer] = df

    # Moving pointer forward
    pointer = pointer + 1

    # Debugging
    # print("done")

    return df_dict, pointer