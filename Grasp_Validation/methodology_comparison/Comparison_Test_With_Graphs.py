# Libraries
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import dump
import tensorflow as tf
import warnings
import pandas as pd
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# Files
from Simple_Threshold_Test import simple_threshold
from Logistic_Regression_Test import logistic_regression
from Single_Layer_Perceptron_Test import single_layer_perceptron
from Decision_Tree_Test import decision_tree
from One_Phase_NN_Test import one_phase_nn
import Two_Phase_NN_Test as tp_nn



class GraspingValidationSystem:
    """
    Complete grasping validation system for adhesive strip gripper
    Compares multiple approaches for grip classification and orientation detection
    """

    def __init__(self, random_state=42):
        self.X_test_scaled = None
        self.X_train_scaled = None
        self.scaler = None
        self.y_orient_test = None
        self.y_orient_train = None
        self.y_grip_test = None
        self.y_grip_train = None
        self.X_test = None
        self.X_train = None
        self.y_orientation = None
        self.y_grip = None
        self.X = None
        self.random_state = random_state
        np.random.seed(random_state)
        tf.random.set_seed(random_state)

        # Initialize storage for models and results
        self.models = {}
        self.scalers = {}
        self.results = {}

        # Define categories
        self.grip_categories = ['no_grip', 'bad_grip', 'semi_good_grip', 'good_grip']
        self.orientation_categories = ['face_A', 'face_B', 'face_C', 'face_D']

        print("Grasping Validation System initialized")
        print(f"Grip categories: {self.grip_categories}")
        print(f"Orientation categories: {self.orientation_categories}")

    def generate_synthetic_data(self, n_samples=5000):
        """
        Generate realistic synthetic data for 4 photo-resistors
        Each sensor value represents light intensity (0-1023 for 10-bit ADC)
        """
        print("\nGenerating synthetic sensor data...")
        # Create variables for df creation:
        df_dict = {}
        pointer = 1

        # Debugging
        # print("Check_1")

        # Create df list:

        # Bottom grip
        [df_dict, pointer] = tp_nn.create_df(
            "data/No_Grip.xlsx",
            "data/Spong_Bottom_Bad.xlsx",
            "data/Spong_Bottom_Semi.xlsx",
            "data/Spong_Bottom_Good.xlsx",
            'face_down', df_dict, pointer)

        # Debugging
        # print("Check_2")

        # Top grip
        [df_dict, pointer] = tp_nn.create_df(
            "data/No_Grip.xlsx",
            "data/Spong_Top_Bad.xlsx",
            "data/Spong_Top_Semi.xlsx",
            "data/Spong_Top_Good.xlsx",
            'face_up', df_dict, pointer)

        # Debugging
        # print("Check_3")

        # Side grip
        [df_dict, pointer] = tp_nn.create_df(
            "data/No_Grip.xlsx",
            "data/Spong_Side_Bad.xlsx",
            "data/Spong_Side_Semi.xlsx",
            "data/Spong_Side_Good.xlsx",
            'left_side', df_dict, pointer)

        # Debugging
        # print("Check_4")

        # print(df_dict)
        # print("Pointer")
        # print(pointer)

        # Connect all DataFrames
        all_data = pd.concat(df_dict.values(), ignore_index=True)

        # Save combine data as CSV and DAtaFrame for training
        all_data.to_csv('all_sponge_data.csv', index=False)
        result_df = all_data

        # Map labels to numeric indices consistent with synthetic data expectations
        label_map_orientation = {'face_down': 0, 'face_up': 1, 'left_side': 2, 'right_side': 3}
        label_map_grip = {'no_grip': 0, 'bad_grip': 1, 'semi_good_grip': 2, 'good_grip': 3}

        all_data['orientation'] = all_data['orientation'].map(label_map_orientation)
        all_data['grip_quality'] = all_data['grip_quality'].map(label_map_grip)

        self.X = all_data[['sensor_1', 'sensor_2', 'sensor_3', 'sensor_4']].values
        self.y_orientation = all_data['orientation'].values
        self.y_grip = all_data['grip_quality'].values

        print(f"Generated {n_samples} samples with 4 sensor readings each")
        print(f"Sensor data shape: {self.X.shape}")
        print(f"Grip distribution: {np.bincount(self.y_grip)}")
        print(f"Orientation distribution: {np.bincount(self.y_orientation)}")
        print(f"y_orientation: {self.y_orientation}")
        print(f"y_grip: {self.y_grip}")

        return self.X, self.y_grip, self.y_orientation

    def prepare_data(self):
        """Split and scale the data for training"""
        print("\nPreparing data for training...")

        # Split data
        (self.X_train, self.X_test,
         self.y_grip_train, self.y_grip_test,
         self.y_orient_train, self.y_orient_test) = train_test_split(
            self.X, self.y_grip, self.y_orientation,
            test_size=0.2, random_state=self.random_state, stratify=self.y_grip
        )

        # Scale features
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        # Save the fitted scaler object for real time use
        dump(self.scaler, 'fitted_scaler.joblib')

        print("Fitted scaler saved to 'fitted_scaler.joblib'")
        print(f"Training set: {self.X_train.shape[0]} samples")
        print(f"Test set: {self.X_test.shape[0]} samples")

    def compare_all_methods(self):
        """
        Compare all methods and display comprehensive results
        """
        print("\n" + "=" * 70)
        print("COMPREHENSIVE COMPARISON OF ALL METHODS")
        print("=" * 70)

        # Create comparison table
        comparison_data = []
        for method_name, results in self.results.items():
            comparison_data.append({
                'Method': method_name.replace('_', ' ').title(),
                'Grip Accuracy': f"{results['grip_accuracy']:.4f}",
            })

        # Sort by average accuracy
        comparison_data.sort(key=lambda x: float(x['Grip Accuracy']), reverse=True)

        # Print results
        print('results')
        print(comparison_data)

        return comparison_data

    def plot_learning_curves_sklearn(self, method_name, train_sizes, train_scores, val_scores):
        plt.figure(figsize=(8, 6))
        plt.plot(train_sizes, train_scores, 'o-', label='Training score', linewidth=2)
        plt.plot(train_sizes, val_scores, 'o-', label='Cross-validation score', linewidth=2)
        plt.xlabel('Training set size')
        plt.ylabel('Accuracy')
        plt.title(f'Learning Curves - {method_name}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'results_figures/learning_curve_{method_name.replace(" ", "_")}.png', dpi=150)
        plt.close()

    def plot_history_one_phase(self, history):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Accuracy
        ax1.plot(history.history['accuracy'], label='Train acc', linewidth=2)
        if 'val_accuracy' in history.history:
            ax1.plot(history.history['val_accuracy'], label='Val acc', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('One-phase NN Accuracy')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Loss
        ax2.plot(history.history['loss'], label='Train loss', linewidth=2)
        if 'val_loss' in history.history:
            ax2.plot(history.history['val_loss'], label='Val loss', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('One-phase NN Loss')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        plt.savefig('results_figures/training_curves_OnePhaseNN.png', dpi=150)
        plt.close()

    def plot_simple_threshold_curve(self, thresholds, accuracies, best_threshold):
        """Plots the accuracy vs threshold curve for the simple threshold method."""
        plt.figure(figsize=(8, 6))
        plt.plot(thresholds, accuracies, color='teal', linewidth=2, label='Accuracy Trace')
        plt.axvline(x=best_threshold, color='red', linestyle='--',
                    label=f'Optimal Threshold: {best_threshold:.2f}')
        plt.xlabel('Sensor Threshold Value')
        plt.ylabel('Classification Accuracy')
        plt.title('Simple Threshold Performance Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('results_figures/simple_threshold_curve.png', dpi=150)
        plt.close()

    def plot_confusion_matrix(self, y_true, y_pred, method_name):
        """Generic method to plot a confusion matrix for any classifier"""
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='viridis',
                    xticklabels=self.grip_categories,
                    yticklabels=self.grip_categories)
        plt.title(f'{method_name} - Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(f'results_figures/confusion_matrix_{method_name.replace(" ", "_")}.png', dpi=150)
        plt.close()

    def run_complete_analysis(self):
        """
        Run the complete analysis pipeline with learning curve plotting for all methods
        """
        print("Starting Complete Grasping Validation System Analysis")
        print("=" * 60)

        # Generate and prepare data
        self.generate_synthetic_data(n_samples=5000)
        self.prepare_data()

        # === 1. SIMPLE THRESHOLD ===
        print("\n" + "=" * 50)
        print("RUNNING SIMPLE THRESHOLD")
        print("=" * 50)
        # Updated to receive 4 values: acc, pred, thresholds, and accuracy_list
        grip_acc_1, y_pred_1, thresholds_1, accs_1 = simple_threshold(
            self.X_test, self.y_grip_test, self.grip_categories
        )
        # Identify the best threshold value from the list for the plot label
        best_thresh_val = thresholds_1[np.argmax(accs_1)]
        self.plot_simple_threshold_curve(thresholds_1, accs_1, best_thresh_val)

        # === 2. LOGISTIC REGRESSION ===
        print("\n" + "=" * 50)
        print("RUNNING LOGISTIC REGRESSION")
        print("=" * 50)
        model_2, grip_acc_2, y_pred_2, sizes_2, train_2, val_2 = logistic_regression(
            self.X_train_scaled, self.X_test_scaled,
            self.y_grip_train, self.y_grip_test, self.grip_categories
        )
        self.plot_learning_curves_sklearn('Logistic Regression', sizes_2, train_2, val_2)

        # === 3. SINGLE LAYER PERCEPTRON ===
        print("\n" + "=" * 50)
        print("RUNNING SINGLE LAYER PERCEPTRON")
        print("=" * 50)
        model_3, grip_acc_3, y_pred_3, sizes_3, train_3, val_3 = single_layer_perceptron(
            self.X_train_scaled, self.X_test_scaled,
            self.y_grip_train, self.y_grip_test, self.grip_categories
        )
        self.plot_learning_curves_sklearn('Single Layer Perceptron', sizes_3, train_3, val_3)

        # === 4. DECISION TREE ===
        print("\n" + "=" * 50)
        print("RUNNING DECISION TREE")
        print("=" * 50)
        model_4, grip_acc_4, y_pred_4, sizes_4, train_4, val_4 = decision_tree(
            self.X_train_scaled, self.X_test_scaled,
            self.y_grip_train, self.y_grip_test, self.grip_categories
        )
        self.plot_learning_curves_sklearn('Decision Tree', sizes_4, train_4, val_4)

        # === 5. ONE PHASE NN ===
        print("\n" + "=" * 50)
        print("RUNNING ONE PHASE NEURAL NETWORK")
        print("=" * 50)
        model_5, grip_acc_5, y_pred_5, history = one_phase_nn(
            self.X_train_scaled, self.X_test_scaled,
            self.y_grip_train, self.y_grip_test, self.grip_categories
        )
        self.plot_history_one_phase(history)

        # Save results for first 5 methods
        self.results['simple_threshold'] = {'grip_accuracy': grip_acc_1, 'grip_prediction': y_pred_1}
        self.results['logistic_regression'] = {'grip_accuracy': grip_acc_2, 'grip_prediction': y_pred_2}
        self.results['single_layer_perceptron'] = {'grip_accuracy': grip_acc_3, 'grip_prediction': y_pred_3}
        self.results['decision_tree'] = {'grip_accuracy': grip_acc_4, 'grip_prediction': y_pred_4}
        self.results['one_phase_nn'] = {'grip_accuracy': grip_acc_5, 'grip_prediction': y_pred_5}

        # === 6. TWO PHASE NN ===
        print("\n" + "=" * 50)
        print("RUNNING TWO PHASE NEURAL NETWORK")
        print("=" * 50)
        # Using the grip_system initialized in the main block
        result_2_nn = grip_system.train_system_with_tuning(epochs=100)

        # Trigger the built-in plots for Two-Phase NN
        grip_system.plot_training_history(result_2_nn)
        grip_system.plot_confusion_matrices(result_2_nn)

        grip_accuracy_6 = accuracy_score(result_2_nn['test_grip'], result_2_nn['prediction_grip'])
        self.results['two_phase_nn'] = {
            'grip_accuracy': grip_accuracy_6,
            'grip_prediction': result_2_nn['prediction_grip']
        }

        # === FINAL COMPARISON ===
        print("\n" + "=" * 70)
        print("FINAL COMPARISON")
        print("=" * 70)
        comparison_results = self.compare_all_methods()

        return comparison_results


# Example usage and main execution
if __name__ == "__main__":
    # Initialize the system
    system = GraspingValidationSystem(random_state=42)
    grip_system = tp_nn.GripValidationSystem()

    # Run complete analysis
    results = system.run_complete_analysis()

    # Additional analysis can be performed here
    print("\nSystem ready for additional testing and validation...")

