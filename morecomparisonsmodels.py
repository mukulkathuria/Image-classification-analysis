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
import torch
from ultralytics import YOLO, settings
import cv2
import os
warnings.filterwarnings('ignore')

np.random.seed(42)
tf.random.set_seed(42)
torch.manual_seed(42)


BATCH_SIZE = 32
IMG_SIZE = 96 
EPOCHS = 10
NUM_CLASSES = 10
DATASET_DIR = 'dataset'
YOLO_DATASET_DIR = 'yolo_dataset'

def prepare_dataset():
    print("Preparing dataset...")

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    

    x_train = x_train[:10000]
    y_train = y_train[:10000]
    x_test = x_test[:2000]
    y_test = y_test[:2000]
    

    x_train = np.stack([x_train, x_train, x_train], axis=-1)
    x_test = np.stack([x_test, x_test, x_test], axis=-1)

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

    x_train_normalized = x_train_resized / 255.0
    x_test_normalized = x_test_resized / 255.0
    

    y_train_encoded = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test_encoded = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)
    
    print(f"Dataset prepared. Training samples: {len(x_train_normalized)}, Test samples: {len(x_test_normalized)}")
    

    prepare_yolo_dataset(x_train_resized, y_train, x_test_resized, y_test)
    
    return x_train_normalized, y_train_encoded, x_test_normalized, y_test_encoded, y_train, y_test

def prepare_yolo_dataset(x_train, y_train, x_test, y_test):
    print("Preparing YOLO dataset...")
    
    os.makedirs(YOLO_DATASET_DIR, exist_ok=True)
    os.makedirs(os.path.join(YOLO_DATASET_DIR,'train'), exist_ok=True)
    os.makedirs(os.path.join(YOLO_DATASET_DIR, 'val'), exist_ok=True)
    os.makedirs(os.path.join(YOLO_DATASET_DIR, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(YOLO_DATASET_DIR, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(YOLO_DATASET_DIR, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(YOLO_DATASET_DIR, 'labels', 'val'), exist_ok=True)
    for i in range(10):
        os.makedirs(os.path.join(YOLO_DATASET_DIR,'train', f'{i}'), exist_ok=True)
        os.makedirs(os.path.join(YOLO_DATASET_DIR,'val', f'{i}'), exist_ok=True)

    for i, (img, label) in enumerate(zip(x_train, y_train)):
        img_path = os.path.join(YOLO_DATASET_DIR, 'images', 'train', f'img_{i}.jpg')
        cv2.imwrite(img_path, img.astype(np.uint8))

        img_path2 = os.path.join(YOLO_DATASET_DIR, 'train', f'{label}', f'img_{i}.jpg')
        cv2.imwrite(img_path2, img.astype(np.uint8))

        label_path = os.path.join(YOLO_DATASET_DIR, 'labels', 'train', f'img_{i}.txt')
        with open(label_path, 'w') as f:
            f.write(f"{label}\n")
    
    for i, (img, label) in enumerate(zip(x_test, y_test)):
        img_path = os.path.join(YOLO_DATASET_DIR, 'images', 'val', f'img_{i}.jpg')
        cv2.imwrite(img_path, img.astype(np.uint8))
        img_path2 = os.path.join(YOLO_DATASET_DIR, 'val', f'{label}', f'img_{i}.jpg')
        cv2.imwrite(img_path2, img.astype(np.uint8))
        
        label_path = os.path.join(YOLO_DATASET_DIR, 'labels', 'val', f'img_{i}.txt')
        with open(label_path, 'w') as f:
            f.write(f"{label}\n")
    
    with open(os.path.join(YOLO_DATASET_DIR, 'dataset.yaml'), 'w') as f:
        f.write(f"path: 'C://Users/mukulkathuria/Documents/mlprac/deeplearning/yolo_dataset'\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write(f"nc: {NUM_CLASSES}\n")
        f.write("names: [")
        for i in range(NUM_CLASSES):
            if i > 0:
                f.write(", ")
            f.write(f"'class_{i}'")
        f.write("]\n")
    print("Yolo Dataset prepared")

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

def create_efficientnet_model():
    base_model = EfficientNetV2B3(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x) 
    x = Dropout(0.5)(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False
        
    return model, "EfficientNetV2B3"

def create_resnet50v2_model():
    base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x) 
    x = Dropout(0.5)(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    
    for layer in base_model.layers:
        layer.trainable = False
        
    return model, "ResNet50V2"

def create_mobilenetv3_model():
    base_model = MobileNetV3Large(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)  
    x = Dropout(0.5)(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False
        
    return model, "MobileNetV3Large"

def create_simple_cnn_model():
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = tf.keras.layers.Conv2D(32, 3, activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(128, 3, activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x) 
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model, "YOLO-like"

def train_yolo_model(x_test, y_test):
    start_time = time.time()
    
    model = YOLO('yolo11n-cls.pt') 
    print("Training YOLO model...")
    settings.update({"runs_dir": os.path.abspath(YOLO_DATASET_DIR)})
    print("Dataset looking at:", os.path.join(YOLO_DATASET_DIR, 'dataset.yaml'))
    results = model.train(
        data='C:/Users/mukulkathuria/Documents/mlprac/deeplearning/yolo_dataset',
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device='cpu' if not torch.cuda.is_available() else '0'
    )
    settings.reset()
    training_time = time.time() - start_time

    start_time = time.time()

    val_results = model.val()
    
    inference_time = time.time() - start_time
    print("Val results",type(val_results))
    accuracy = val_results.top1

    predictions = []
    y_true = y_test.tolist()

    batch_size = 100
    for i in range(0, len(x_test), batch_size):
        batch_imgs = x_test[i:i+batch_size]
        for j, img in enumerate(batch_imgs):
            img_path = os.path.join(YOLO_DATASET_DIR, 'temp_pred.jpg')
            cv2.imwrite(img_path, img.astype(np.uint8))
            
            results = model.predict(img_path, verbose=False)
            
            if len(results) > 0 and hasattr(results[0], 'probs'):
                top_class = int(results[0].probs.top1)
                predictions.append(top_class)
            else:
                predictions.append(0)

    precision = precision_score(y_true, predictions, average='weighted', zero_division=0)
    recall = recall_score(y_true, predictions, average='weighted', zero_division=0)
    f1 = f1_score(y_true, predictions, average='weighted', zero_division=0)
    cm = confusion_matrix(y_true, predictions)
    
    results = {
        'model_name': "YOLOv11n-cls",
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'training_time': training_time,
        'inference_time': inference_time,
        'history': {'accuracy': [accuracy], 'val_accuracy': [accuracy]},
        'confusion_matrix': cm
    }
    
    print(f"\nModel: {results['model_name']}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1_score']:.4f}")
    print(f"Training Time: {results['training_time']:.2f} seconds")
    print(f"Inference Time: {results['inference_time']:.4f} seconds")
    
    return results

def hyperparameter_tuning(model_func, x_train, y_train):
    print(f"Performing hyperparameter tuning for {model_func.__name__}...")
    
    learning_rates = [0.001, 0.0001]
    dropout_rates = [0.3, 0.5]
    
    best_val_accuracy = 0
    best_params = {}
    
    train_subset_size = min(2000, len(x_train))
    x_train_subset = x_train[:train_subset_size]
    y_train_subset = y_train[:train_subset_size]

    x_train_sub, x_val, y_train_sub, y_val = train_test_split(
        x_train_subset, y_train_subset, test_size=0.2, random_state=42
    )
    
    for lr in learning_rates:
        for dr in dropout_rates:
            print(f"  Testing: LR={lr}, Dropout={dr}")
            
        
            model, model_name = model_func()

            for layer in model.layers:
                if isinstance(layer, Dropout):
                    layer.rate = dr

            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            

            early_stopping = EarlyStopping(monitor='val_accuracy', patience=2, restore_best_weights=True)
            

            history = model.fit(
                x_train_sub, y_train_sub,
                batch_size=BATCH_SIZE,
                epochs=3,
                validation_data=(x_val, y_val),
                callbacks=[early_stopping],
                verbose=0
            )
            

            val_accuracy = max(history.history['val_accuracy'])
            

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_params = {
                    'learning_rate': lr,
                    'batch_size': BATCH_SIZE,
                    'dropout_rate': dr
                }
            tf.keras.backend.clear_session()
    
    print(f"Best hyperparameters for {model_func.__name__}: {best_params}")
    return best_params


def train_model(model_func, best_params, x_train, y_train, x_test, y_test):
    print(f"Training {model_func.__name__} with best hyperparameters...")
    

    model, model_name = model_func()
    

    for layer in model.layers:
        if isinstance(layer, Dropout):
            layer.rate = best_params['dropout_rate']
    

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=best_params['learning_rate']),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    

    early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6)
    

    start_time = time.time()
    

    history = model.fit(
        x_train, y_train,
        batch_size=best_params['batch_size'],
        epochs=EPOCHS,
        validation_split=0.1,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    

    training_time = time.time() - start_time

    start_time = time.time()
    y_pred_proba = model.predict(x_test)
    inference_time = time.time() - start_time
    
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(y_test, axis=1)
    

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    

    print(f"\nModel: {model_name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Training Time: {training_time:.2f} seconds")
    print(f"Inference Time: {inference_time:.4f} seconds")
    
    cm = confusion_matrix(y_true, y_pred)
    
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

def plot_learning_curves(histories_dict):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for model_name, history in histories_dict.items():
        if 'accuracy' in history['history']:
            plt.plot(history['history']['accuracy'], label=f'{model_name} Train')
            if 'val_accuracy' in history['history']:
                plt.plot(history['history']['val_accuracy'], label=f'{model_name} Val')
    
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    
    plt.subplot(1, 2, 2)
    for model_name, history in histories_dict.items():
        if 'loss' in history['history']:  # Check if history contains loss
            plt.plot(history['history']['loss'], label=f'{model_name} Train')
            if 'val_loss' in history['history']:
                plt.plot(history['history']['val_loss'], label=f'{model_name} Val')
    
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('learning_curvesmorecomparison.png')
    plt.close()

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
    plt.savefig('confusion_matrices_morecomparison.png')
    plt.close()


def plot_model_comparison(results_dict):
    models = list(results_dict.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    

    data = []
    for model in models:
        for metric in metrics:
            data.append({
                'Model': model,
                'Metric': metric.capitalize(),
                'Value': results_dict[model][metric]
            })
    
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Model', y='Value', hue='Metric', data=df)
    plt.title('Model Performance Comparison')
    plt.ylim(0, 1)
    plt.legend(loc='lower right')
    plt.savefig('model_metrics_morecomparison.png')
    plt.close()
    

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
    plt.savefig('model_time_morecomparison.png')
    plt.close()


def create_results_table(results_dict):
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
    
    df.to_csv('model_morecomparison_results.csv', index=False)
    
    print("\nModel Comparison Results:")
    print(df.to_string(index=False))
    
    return df

def main():
    print("Starting Image Classification Model Comparison")

    x_train, y_train, x_test, y_test, y_train_orig, y_test_orig = prepare_dataset()
    
    model_functions = [
        create_efficientnet_model,
        create_resnet50v2_model,
        create_mobilenetv3_model,
        create_simple_cnn_model
    ]
    
    results_dict = {}
    yolo_results = train_yolo_model(x_test, y_test_orig)
    results_dict[yolo_results['model_name']] = yolo_results
    
    for model_func in model_functions:
        best_params = hyperparameter_tuning(model_func, x_train, y_train)
        
        results = train_model(model_func, best_params, x_train, y_train, x_test, y_test)
        model_name = results['model_name']
        results_dict[model_name] = results
    plot_learning_curves(results_dict)

    plot_confusion_matrices(results_dict)
    
    plot_model_comparison(results_dict)

    results_table = create_results_table(results_dict)
    
    print("\nComparison completed. Results saved to CSV and visualizations saved as PNG files.")

if __name__ == "__main__":
    main()