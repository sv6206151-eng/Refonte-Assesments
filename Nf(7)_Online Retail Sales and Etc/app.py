import gzip
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

def load_mnist_data(images_path, labels_path):
    """
    Loads MNIST data from gzipped IDX files.
    """
    with gzip.open(labels_path, 'rb') as lbpath:
        # Read magic number and number of items
        magic, n = np.frombuffer(lbpath.read(8), dtype=np.dtype('>i4'))
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8)

    with gzip.open(images_path, 'rb') as imgpath:
        # Read magic number, number of images, rows, and cols
        magic, num_images, rows, cols = np.frombuffer(imgpath.read(16), dtype=np.dtype('>i4'))
        images = np.frombuffer(imgpath.read(), dtype=np.uint8).reshape(len(labels), rows, cols)

    return images, labels

# Load the dataset from the provided files
train_images, train_labels = load_mnist_data('train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz')
test_images, test_labels = load_mnist_data('t10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz')

# Reshape and normalize images
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

# One-hot encode the labels
train_labels_encoded = tf.keras.utils.to_categorical(train_labels)
test_labels_encoded = tf.keras.utils.to_categorical(test_labels)

# Build the CNN model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# The following two code blocks were duplicates and have been combined
# with the corrected variable names.

# Compile and train the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model with validation split and corrected variable name
history = model.fit(train_images, train_labels_encoded,
                    epochs=10,
                    batch_size=32,
                    validation_split=0.2)

# Evaluate the model on the test dataset with corrected variable name
test_loss, test_acc = model.evaluate(test_images, test_labels_encoded)
print(f"\nTest accuracy: {test_acc:.4f}")

# Visualize training and validation accuracy/loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig('accuracy_loss_plot.png')

# Plot some test images with predictions
predictions = model.predict(test_images)
plt.figure(figsize=(10, 5))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(test_images[i].reshape(28, 28), cmap='gray')
    predicted_label = np.argmax(predictions[i])
    true_label = np.argmax(test_labels_encoded[i])
    plt.title(f"Pred: {predicted_label}\nTrue: {true_label}",
              color="green" if predicted_label == true_label else "red")
    plt.axis('off')
plt.tight_layout()
plt.savefig('predictions.png')