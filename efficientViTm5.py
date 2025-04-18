import tensorflow as tf
from tensorflow.keras import layers, models, optimizers # type: ignore
from tensorflow.keras.datasets import fashion_mnist # type: ignore
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore

tf.random.set_seed(42)
np.random.seed(42)

print("Loading Fashion MNIST dataset...")
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

x_train = tf.image.resize(x_train, [32, 32])
x_test = tf.image.resize(x_test, [32, 32])

y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(f"Training data shape: {x_train.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Test data shape: {x_test.shape}")
print(f"Test labels shape: {y_test.shape}")


class PositionalEncoding(layers.Layer):
    def __init__(self, embed_dim, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        
    def build(self, input_shape):
        _, height, width, _ = input_shape
        self.pos_encoding = self.add_weight(
            shape=(1, height, width, self.embed_dim),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
            trainable=True,
            name='positional_encoding'
        )
        super(PositionalEncoding, self).build(input_shape)
        
    def call(self, inputs):
        return inputs + self.pos_encoding
        
    def get_config(self):
        config = super(PositionalEncoding, self).get_config()
        config.update({
            'embed_dim': self.embed_dim
        })
        return config

class EfficientAttention(layers.Layer):
    def __init__(self, num_heads=4, key_dim=32, dropout_rate=0.2, **kwargs):
        super(EfficientAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.dropout_rate = dropout_rate
        
    def build(self, input_shape):
        self.layer_norm = layers.LayerNormalization(epsilon=1e-6)
        self.reshape_1 = layers.Reshape((-1, input_shape[-1]))
        self.mha = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.key_dim,
            dropout=self.dropout_rate
        )
        self.reshape_2 = layers.Reshape((input_shape[1], input_shape[2], input_shape[3]))
        self.add = layers.Add()
        super(EfficientAttention, self).build(input_shape)
        
    def call(self, inputs):
        x_norm = self.layer_norm(inputs)
        batch_size, height, width, channels = tf.shape(x_norm)[0], x_norm.shape[1], x_norm.shape[2], x_norm.shape[3]
        x_flat = self.reshape_1(x_norm)
        attn_output = self.mha(x_flat, x_flat)
        attn_output = self.reshape_2(attn_output)
        return self.add([inputs, attn_output])
        
    def get_config(self):
        config = super(EfficientAttention, self).get_config()
        config.update({
            'num_heads': self.num_heads,
            'key_dim': self.key_dim,
            'dropout_rate': self.dropout_rate
        })
        return config

class MLPBlock(layers.Layer):
    def __init__(self, hidden_dim, dropout_rate=0.2, **kwargs):
        super(MLPBlock, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        
    def build(self, input_shape):
        self.dense1 = layers.Dense(self.hidden_dim)
        self.activation = layers.Activation('gelu')
        self.dropout1 = layers.Dropout(self.dropout_rate)
        self.dense2 = layers.Dense(input_shape[-1])
        self.dropout2 = layers.Dropout(self.dropout_rate)
        super(MLPBlock, self).build(input_shape)
        
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.dropout2(x)
        return x
        
    def get_config(self):
        config = super(MLPBlock, self).get_config()
        config.update({
            'hidden_dim': self.hidden_dim,
            'dropout_rate': self.dropout_rate
        })
        return config

class LightweightConvBlock(layers.Layer):
    def __init__(self, kernel_size=3, expansion_factor=2, dropout_rate=0.2, **kwargs):
        super(LightweightConvBlock, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.expansion_factor = expansion_factor
        self.dropout_rate = dropout_rate
        
    def build(self, input_shape):
        input_channels = input_shape[-1]
        expanded_channels = int(input_channels * self.expansion_factor)
        
        self.conv1 = layers.Conv2D(expanded_channels, kernel_size=1, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.activation1 = layers.Activation('gelu')
        
        self.depthwise_conv = layers.DepthwiseConv2D(self.kernel_size, padding='same')
        self.bn2 = layers.BatchNormalization()
        self.activation2 = layers.Activation('gelu')
        
        self.conv2 = layers.Conv2D(input_channels, kernel_size=1, padding='same')
        self.bn3 = layers.BatchNormalization()
        self.dropout = layers.Dropout(self.dropout_rate)
        
        self.add = layers.Add()
        super(LightweightConvBlock, self).build(input_shape)
        
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.activation1(x)
        
        x = self.depthwise_conv(x)
        x = self.bn2(x)
        x = self.activation2(x)
        
        x = self.conv2(x)
        x = self.bn3(x)
        x = self.dropout(x)
        
        return self.add([inputs, x])
        
    def get_config(self):
        config = super(LightweightConvBlock, self).get_config()
        config.update({
            'kernel_size': self.kernel_size,
            'expansion_factor': self.expansion_factor,
            'dropout_rate': self.dropout_rate
        })
        return config

class EfficientViTBlock(layers.Layer):
    def __init__(self, mlp_dim, num_heads, dropout_rate=0.2, **kwargs):
        super(EfficientViTBlock, self).__init__(**kwargs)
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        
    def build(self, input_shape):
        self.layer_norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.efficient_attention = EfficientAttention(
            num_heads=self.num_heads, 
            dropout_rate=self.dropout_rate
        )
        self.add1 = layers.Add()
        
        self.layer_norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp_block = MLPBlock(self.mlp_dim, self.dropout_rate)
        self.add2 = layers.Add()
        
        self.lightweight_conv = LightweightConvBlock(dropout_rate=self.dropout_rate)
        super(EfficientViTBlock, self).build(input_shape)
        
    def call(self, inputs):
        x_norm = self.layer_norm1(inputs)
        attn_output = self.efficient_attention(x_norm)
        x = self.add1([inputs, attn_output])
        
        x_norm = self.layer_norm2(x)
        mlp_output = self.mlp_block(x_norm)
        x = self.add2([x, mlp_output])
        
        x = self.lightweight_conv(x)
        
        return x
        
    def get_config(self):
        config = super(EfficientViTBlock, self).get_config()
        config.update({
            'mlp_dim': self.mlp_dim,
            'num_heads': self.num_heads,
            'dropout_rate': self.dropout_rate
        })
        return config

def conv_block(x, filters, kernel_size=3, strides=1):
    """Standard convolutional block with batch normalization and activation"""
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('gelu')(x)
    return x

def EfficientViTM5(input_shape, num_classes, embed_dim=128, mlp_dim=512, 
                   num_heads=4, num_blocks=8, dropout_rate=0.2):
    """
    EfficientViTM5 model implementation
    
    M5 is one of the lightweight variants of EfficientViT optimized for efficiency
    """
    inputs = layers.Input(shape=input_shape)
    
    x = conv_block(inputs, filters=embed_dim//4, kernel_size=3, strides=1)
    x = conv_block(x, filters=embed_dim//2, kernel_size=3, strides=2)
    x = conv_block(x, filters=embed_dim, kernel_size=3, strides=2)
    
    x = PositionalEncoding(embed_dim)(x)
    
    for _ in range(num_blocks):
        x = EfficientViTBlock(mlp_dim=mlp_dim, num_heads=num_heads, dropout_rate=dropout_rate)(x)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Dense(mlp_dim, activation='gelu')(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    
    return model

batch_size = 128
epochs = 100
validation_split = 0.1
learning_rate = 3e-4

print("Building EfficientViTM5 model...")
model = EfficientViTM5(
    input_shape=(32, 32, 1),
    num_classes=10,
    embed_dim=96,       
    mlp_dim=384,
    num_heads=3,
    num_blocks=6,
    dropout_rate=0.2
)

optimizer = tf.keras.optimizers.AdamW(
    learning_rate=learning_rate,
    weight_decay=0.01
)

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
]

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

def apply_augmentation(images, labels):
    return data_augmentation(images, training=True), labels

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_size = int(len(x_train) * (1 - validation_split))
train_ds = train_dataset.take(train_size).batch(batch_size).map(apply_augmentation).prefetch(tf.data.AUTOTUNE)
val_ds = train_dataset.skip(train_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

print("Training the model...")
history = model.fit(
    train_ds,
    epochs=epochs,
    validation_data=val_ds,
    callbacks=callbacks,
    verbose=1
)

print("Evaluating the model...")
test_loss, test_accuracy = model.evaluate(test_ds)
print(f"Test accuracy: {test_accuracy:.4f}")
print(f"Test loss: {test_loss:.4f}")

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.tight_layout()
plt.savefig('efficientvit_m5_training_history.png')
plt.show()

predictions = model.predict(test_ds)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test, axis=1)

def plot_predictions(images, true_labels, pred_labels, class_names, n=10):
    plt.figure(figsize=(15, 3))
    for i in range(n):
        plt.subplot(1, n, i+1)
        plt.imshow(images[i].numpy().reshape(32, 32), cmap='gray')
        color = 'green' if true_labels[i] == pred_labels[i] else 'red'
        plt.title(f"T: {class_names[true_labels[i]]}\nP: {class_names[pred_labels[i]]}", color=color)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('efficientvit_m5_predictions.png')
    plt.show()

print("Displaying example predictions...")
random_indices = np.random.choice(len(x_test), 10, replace=False)
sample_images = tf.data.Dataset.from_tensor_slices(x_test[random_indices]).batch(10)
for images in sample_images:
    plot_predictions(images, true_classes[random_indices], 
                    predicted_classes[random_indices], class_names)

model.save('efficientvit_m5_fashion_mnist.keras')
print("Model saved as 'efficientvit_m5_fashion_mnist.keras'")

from sklearn.metrics import confusion_matrix
import seaborn as sns

conf_matrix = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('efficientvit_m5_confusion_matrix.png')
plt.show()

print("Complete!")