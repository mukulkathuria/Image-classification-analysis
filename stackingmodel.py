import marimo

__generated_with = "0.12.6"
app = marimo.App(width="medium")


@app.cell
def _():
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
    from tensorflow.keras import regularizers  #type:ignore
    import time
    import warnings
    import torch
    import cv2
    import os
    return (
        Dense,
        Dropout,
        EarlyStopping,
        EfficientNetV2B3,
        GlobalAveragePooling2D,
        ImageDataGenerator,
        MobileNetV3Large,
        Model,
        ReduceLROnPlateau,
        ResNet50V2,
        accuracy_score,
        classification_report,
        confusion_matrix,
        cv2,
        f1_score,
        np,
        os,
        pd,
        plt,
        precision_score,
        recall_score,
        regularizers,
        sns,
        tf,
        time,
        torch,
        train_test_split,
        warnings,
    )


@app.cell
def _(np, tf):
    IMG_SIZE = 224
    NUM_CLASSES = 10
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

        return x_train_normalized, y_train_encoded, x_test_normalized, y_test_encoded, y_train, y_test
    return IMG_SIZE, NUM_CLASSES, prepare_dataset


@app.cell
def _(prepare_dataset):
    x_train, y_train, x_test, y_test, y_train_orig, y_test_orig = prepare_dataset()
    return x_test, x_train, y_test, y_test_orig, y_train, y_train_orig


@app.cell
def _(IMG_SIZE, NUM_CLASSES, tf):
    input_layer = tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    mobilenetv3model = tf.keras.applications.MobileNetV3Large(
        alpha=1.0,
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        classes=NUM_CLASSES,
        pooling=None,
        dropout_rate=0.2,
        classifier_activation="softmax",
        include_preprocessing=True,
        name="MobileNetV3Large",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    resnetv2model = tf.keras.applications.ResNet50V2(
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        classes=NUM_CLASSES,
        pooling=None,
        classifier_activation="softmax",
        name="ResNet50V2",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    efficintnetv2model = tf.keras.applications.EfficientNetV2B3(
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        classes=NUM_CLASSES,
        pooling=None,
        classifier_activation="softmax",
       input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    return efficintnetv2model, input_layer, mobilenetv3model, resnetv2model


@app.cell
def _(
    Dense,
    Dropout,
    EarlyStopping,
    GlobalAveragePooling2D,
    Model,
    efficintnetv2model,
    input_layer,
    mobilenetv3model,
    regularizers,
    resnetv2model,
    tf,
    x_test,
    x_train,
    y_test,
    y_train,
):
    x_train_sub = x_train
    y_train_sub = y_train
    x_val, y_val = x_test, y_test

    def extract_features(base_model):
        feature_extractor = Model(
            inputs=base_model.input, 
            outputs=base_model.get_layer(index=-1).output
        )
        return feature_extractor

    def ensure_4d(features):
        if len(features.shape) == 2:
            return tf.reshape(features, [-1, 7, 7, features.shape[-1]])
        return features

    for layer in mobilenetv3model.layers:
        layer.trainable = False
    for layer in resnetv2model.layers:
        layer.trainable = False
    for layer in efficintnetv2model.layers:
        layer.trainable = False

    mobilenetv3_output = mobilenetv3model(input_layer)
    mobilenetv3_output = GlobalAveragePooling2D()(mobilenetv3_output)
    resnetv2_output = resnetv2model(input_layer)
    resnetv2_output = GlobalAveragePooling2D()(resnetv2_output)
    efficintnetv2_output = efficintnetv2model(input_layer)
    efficintnetv2_output = GlobalAveragePooling2D()(efficintnetv2_output)

    mobilenetv3_output = tf.keras.layers.Flatten()(mobilenetv3_output)
    resnetv2_output = tf.keras.layers.Flatten()(resnetv2_output)
    efficintnetv2_output = tf.keras.layers.Flatten()(efficintnetv2_output)

    merged = tf.keras.layers.Concatenate(axis=-1)([mobilenetv3_output,resnetv2_output, efficintnetv2_output])

    x = Dense(512, activation='relu')(merged)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(256, activation='relu', kernel_regularizer=regularizers.L2(0.001))(x)
    x = Dropout(0.5)(x)

    output = Dense(10, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=output)
    # model = tf.keras.Sequential(
    #     [
    #         mobilenetv3model,
    #         # resnetv2model,
    #         # efficintnetv2model,
    #          # tf.keras.layers.GlobalAveragePooling2D(),
    #         tf.keras.layers.Dense(1024, activation="relu"),
    #         tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"),
    #     ]
    # )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    early_stopping = EarlyStopping(
        monitor="val_accuracy", patience=2, restore_best_weights=True
    )
    history = model.fit(
        x_train_sub,
        y_train_sub,
        batch_size=32,
        epochs=3,
        validation_data=(x_val, y_val),
        callbacks=[early_stopping],
        verbose=0,
    )
    return (
        early_stopping,
        efficintnetv2_output,
        ensure_4d,
        extract_features,
        history,
        layer,
        merged,
        mobilenetv3_output,
        model,
        output,
        resnetv2_output,
        x,
        x_train_sub,
        x_val,
        y_train_sub,
        y_val,
    )


@app.cell
def _(history):
    accuracy = history.history['accuracy'][-1]
    print("Training Accuracy : ", accuracy* 100)
    return (accuracy,)


@app.cell
def _(model, x_test):
    y_pred_proba = model.predict(x_test)
    return (y_pred_proba,)


@app.cell
def _(
    accuracy_score,
    f1_score,
    np,
    precision_score,
    recall_score,
    y_pred_proba,
    y_test,
):
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(y_test, axis=1)
    testaccuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    print(f"Accuracy: {testaccuracy * 100:.4f}")
    print(f"Precision: {precision * 100:.4f}")
    print(f"Recall: {recall * 100:.4f}")
    print(f"F1 Score: {f1 * 100:.4f}")
    return f1, precision, recall, testaccuracy, y_pred, y_true


if __name__ == "__main__":
    app.run()
