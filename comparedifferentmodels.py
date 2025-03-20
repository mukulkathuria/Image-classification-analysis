import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2B3, ResNet50V2, MobileNetV3Large # type: ignore
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
import time
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configuration
BATCH_SIZE = 32
IMG_SIZE = 96  # Reduced image size for memory efficiency
EPOCHS = 10
NUM_CLASSES = 10
DATASET_DIR = 'dataset'

# Prepare dataset more memory-efficiently
def prepare_dataset():
    print("Preparing dataset...")
    # Use Fashion MNIST (smaller dataset)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    
    # Take a subset of the data for faster processing
    x_train = x_train[:10000]
    y_train = y_train[:10000]
    x_test = x_test[:2000]
    y_test = y_test[:2000]
    
    # Reshape and add channel dimension (grayscale to RGB)
    x_train = np.stack([x_train, x_train, x_train], axis=-1)
    x_test = np.stack([x_test, x_test, x_test], axis=-1)
    
    # Resize images in batches to save memory
    x_train_resized = np.zeros((len(x_train), IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    x_test_resized = np.zeros((len(x_test), IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    
    batch_size = 500
    for i in range(0, len(x_train), batch_size):
        end = min(i + batch_size, len(x_train))
        batch = tf.convert_to_tensor(x_train[i:end])
        resized_batch = tf.image.resize(batch, (IMG_SIZE, IMG_SIZE))
        x_train_resized[i:end] = resized_batch.numpy()
    
    for i in range(0, len(x_test), batch_size):
        end = min(i + batch_size, len(x_test))
        batch = tf.convert_to_tensor(x_test[i:end])
        resized_batch = tf.image.resize(batch, (IMG_SIZE, IMG_SIZE))
        x_test_resized[i:end] = resized_batch.numpy()
    
    # Normalize pixel values
    x_train_normalized = x_train_resized / 255.0
    x_test_normalized = x_test_resized / 255.0
    
    # One-hot encode labels
    y_train_encoded = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test_encoded = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)
    
    print(f"Dataset prepared. Training samples: {len(x_train_normalized)}, Test samples: {len(x_test_normalized)}")
    return x_train_normalized, y_train_encoded, x_test_normalized, y_test_encoded

# Data augmentation
def create_data_generators(x_train, y_train, x_test, y_test):
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    train_generator = train_datagen.flow(
        x_train, y_train,
        batch_size=BATCH_SIZE
    )
    
    test_datagen = ImageDataGenerator()
    test_generator = test_datagen.flow(
        x_test, y_test,
        batch_size=BATCH_SIZE
    )
    
    return train_generator, test_generator

# Model creation functions
def create_efficientnet_model():
    base_model = EfficientNetV2B3(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)  # Reduced size
    x = Dropout(0.5)(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Freeze base layers
    for layer in base_model.layers:
        layer.trainable = False
        
    return model, "EfficientNetV2B3"

def create_resnet50v2_model():
    base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)  # Reduced size
    x = Dropout(0.5)(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Freeze base layers
    for layer in base_model.layers:
        layer.trainable = False
        
    return model, "ResNet50V2"

def create_mobilenetv3_model():
    base_model = MobileNetV3Large(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)  # Reduced size
    x = Dropout(0.5)(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Freeze base layers
    for layer in base_model.layers:
        layer.trainable = False
        
    return model, "MobileNetV3Large"

# Simple CNN model (replacing YOLO)
def create_yolo_model():
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = tf.keras.layers.Conv2D(32, 3, activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(128, 3, activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)  # Reduced size
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model, "YOLO-like"

# Simplified hyperparameter tuning function
def hyperparameter_tuning(model_func, x_train, y_train):
    print(f"Performing hyperparameter tuning for {model_func.__name__}...")
    
    # Define a smaller hyperparameter grid
    learning_rates = [0.001, 0.0001]
    dropout_rates = [0.3, 0.5]
    
    best_val_accuracy = 0
    best_params = {}
    
    # Use a smaller subset for faster tuning
    train_subset_size = min(2000, len(x_train))
    x_train_subset = x_train[:train_subset_size]
    y_train_subset = y_train[:train_subset_size]
    
    # Split into train and validation
    x_train_sub, x_val, y_train_sub, y_val = train_test_split(
        x_train_subset, y_train_subset, test_size=0.2, random_state=42
    )
    
    for lr in learning_rates:
        for dr in dropout_rates:
            print(f"  Testing: LR={lr}, Dropout={dr}")
            
            # Create model
            model, model_name = model_func()
            
            # Update dropout rate
            for layer in model.layers:
                if isinstance(layer, Dropout):
                    layer.rate = dr
            
            # Compile model
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Train model with early stopping
            early_stopping = EarlyStopping(monitor='val_accuracy', patience=2, restore_best_weights=True)
            
            # Train for very few epochs during tuning
            history = model.fit(
                x_train_sub, y_train_sub,
                batch_size=BATCH_SIZE,
                epochs=3,
                validation_data=(x_val, y_val),
                callbacks=[early_stopping],
                verbose=0
            )
            
            # Get validation accuracy
            val_accuracy = max(history.history['val_accuracy'])
            
            # Update best parameters if needed
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_params = {
                    'learning_rate': lr,
                    'batch_size': BATCH_SIZE,
                    'dropout_rate': dr
                }
            
            # Clean up
            tf.keras.backend.clear_session()
    
    print(f"Best hyperparameters for {model_func.__name__}: {best_params}")
    return best_params

# Train model with best hyperparameters
def train_model(model_func, best_params, x_train, y_train, x_test, y_test):
    print(f"Training {model_func.__name__} with best hyperparameters...")
    
    # Create model
    model, model_name = model_func()
    
    # Update dropout rate
    for layer in model.layers:
        if isinstance(layer, Dropout):
            layer.rate = best_params['dropout_rate']
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=best_params['learning_rate']),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6)
    
    # Start timer
    start_time = time.time()
    
    # Train model
    history = model.fit(
        x_train, y_train,
        batch_size=best_params['batch_size'],
        epochs=EPOCHS,
        validation_split=0.1,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # End timer
    training_time = time.time() - start_time
    
    # Evaluate model
    start_time = time.time()
    y_pred_proba = model.predict(x_test)
    inference_time = time.time() - start_time
    
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    # Print results
    print(f"\nModel: {model_name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Training Time: {training_time:.2f} seconds")
    print(f"Inference Time: {inference_time:.4f} seconds")
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Return results
    results = {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'training_time': training_time,
        'inference_time': inference_time,
        'history': history.history,
        'confusion_matrix': cm
    }
    
    return results

# Function to plot learning curves
def plot_learning_curves(histories_dict):
    plt.figure(figsize=(12, 5))
    
    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)
    for model_name, history in histories_dict.items():
        plt.plot(history['history']['accuracy'], label=f'{model_name} Train')
        plt.plot(history['history']['val_accuracy'], label=f'{model_name} Val')
    
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    
    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    for model_name, history in histories_dict.items():
        plt.plot(history['history']['loss'], label=f'{model_name} Train')
        plt.plot(history['history']['val_loss'], label=f'{model_name} Val')
    
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('learning_curves.png')
    plt.close()

# Function to plot confusion matrices
def plot_confusion_matrices(results_dict):
    models = list(results_dict.keys())
    n_models = len(models)
    
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 5))
    
    for i, model_name in enumerate(models):
        cm = results_dict[model_name]['confusion_matrix']
        if n_models > 1:
            ax = axes[i]
        else:
            ax = axes
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f'{model_name} Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
    
    plt.tight_layout()
    plt.savefig('confusion_matrices.png')
    plt.close()

# Function to compare model performances
def plot_model_comparison(results_dict):
    models = list(results_dict.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    # Create a DataFrame for easy plotting
    data = []
    for model in models:
        for metric in metrics:
            data.append({
                'Model': model,
                'Metric': metric.capitalize(),
                'Value': results_dict[model][metric]
            })
    
    df = pd.DataFrame(data)
    
    # Plot metrics comparison
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Model', y='Value', hue='Metric', data=df)
    plt.title('Model Performance Comparison')
    plt.ylim(0, 1)
    plt.legend(loc='lower right')
    plt.savefig('model_metrics_comparison.png')
    plt.close()
    
    # Plot training and inference time
    time_data = []
    for model in models:
        time_data.append({
            'Model': model,
            'Metric': 'Training Time (s)',
            'Value': results_dict[model]['training_time']
        })
        time_data.append({
            'Model': model,
            'Metric': 'Inference Time (s)',
            'Value': results_dict[model]['inference_time']
        })
    
    time_df = pd.DataFrame(time_data)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Model', y='Value', hue='Metric', data=time_df)
    plt.title('Model Time Comparison')
    plt.yscale('log')
    plt.ylabel('Time (seconds)')
    plt.savefig('model_time_comparison.png')
    plt.close()

# Create a results table
def create_results_table(results_dict):
    # Create a DataFrame for the results
    data = []
    for model_name, results in results_dict.items():
        data.append({
            'Model': model_name,
            'Accuracy': f"{results['accuracy']:.4f}",
            'Precision': f"{results['precision']:.4f}",
            'Recall': f"{results['recall']:.4f}",
            'F1 Score': f"{results['f1_score']:.4f}",
            'Training Time (s)': f"{results['training_time']:.2f}",
            'Inference Time (s)': f"{results['inference_time']:.4f}"
        })
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv('model_comparison_results.csv', index=False)
    
    # Display the table
    print("\nModel Comparison Results:")
    print(df.to_string(index=False))
    
    return df

# Main function
def main():
    print("Starting Image Classification Model Comparison")
    
    # Prepare dataset
    x_train, y_train, x_test, y_test = prepare_dataset()
    
    # Model creation functions
    model_functions = [
        create_efficientnet_model,
        create_resnet50v2_model,
        create_mobilenetv3_model,
        create_yolo_model
    ]
    
    # Dictionary to store results
    results_dict = {}
    
    # For each model
    for model_func in model_functions:
        # Hyperparameter tuning
        best_params = hyperparameter_tuning(model_func, x_train, y_train)
        
        # Train model with best hyperparameters
        results = train_model(model_func, best_params, x_train, y_train, x_test, y_test)
        
        # Store results
        model_name = results['model_name']
        results_dict[model_name] = results
    
    print("Training YOLO v11")

    # Plot learning curves
    plot_learning_curves(results_dict)
    
    # Plot confusion matrices
    plot_confusion_matrices(results_dict)
    
    # Plot model comparison
    plot_model_comparison(results_dict)
    
    # Create results table
    results_table = create_results_table(results_dict)
    
    print("\nComparison completed. Results saved to CSV and visualizations saved as PNG files.")

if __name__ == "__main__":
    main()