"""
train_model.py
----------------
CodeAlpha Machine Learning Internship - Task 3
Handwritten Digit Recognition Using Convolutional Neural Networks (CNN)

This script:
1. Loads the MNIST handwritten digit dataset
2. Preprocesses and normalizes the data
3. Builds a simple CNN model
4. Trains the model
5. Evaluates the model on test data
6. Prints test accuracy
7. Plots accuracy and loss graphs
8. Plots a confusion matrix
9. Visualizes sample predictions
10. Saves the trained model to disk

Run with:  python train_model.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow.keras import layers, models

# ---------------------------------------------------------------------------
# 0. Setup: folders for saving models and results
# ---------------------------------------------------------------------------
# Paths are relative to the project root (one level above this "src" folder)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load the MNIST dataset
# ---------------------------------------------------------------------------
# MNIST is built into Keras, so no manual download is needed.
# It contains 60,000 training images and 10,000 test images of
# handwritten digits (0-9), each 28x28 pixels, grayscale.
print("Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print(f"Training data shape: {x_train.shape}")
print(f"Test data shape: {x_test.shape}")

# ---------------------------------------------------------------------------
# 2. Data preprocessing and normalization
# ---------------------------------------------------------------------------
# Normalize pixel values from [0, 255] to [0, 1] range.
# This helps the neural network train faster and more reliably.
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Reshape data to add a "channel" dimension: (28, 28) -> (28, 28, 1)
# CNNs in Keras expect input shape: (height, width, channels)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# Convert labels to one-hot encoded vectors, e.g. 3 -> [0,0,0,1,0,0,0,0,0,0]
num_classes = 10
y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes)
y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes)

print("Data preprocessing complete.")

# ---------------------------------------------------------------------------
# 3. CNN model creation
# ---------------------------------------------------------------------------
# A simple but effective CNN architecture:
#   Conv2D -> MaxPooling -> Conv2D -> MaxPooling -> Flatten -> Dense -> Output
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),

    # First convolutional block: learns simple features (edges, curves)
    layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),

    # Second convolutional block: learns more complex features
    layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),

    # Flatten the 2D feature maps into a 1D vector
    layers.Flatten(),

    # Dropout randomly disables neurons during training to reduce overfitting
    layers.Dropout(0.5),

    # Fully connected (dense) layer
    layers.Dense(128, activation="relu"),

    # Output layer: 10 neurons (one per digit 0-9), softmax gives probabilities
    layers.Dense(num_classes, activation="softmax"),
])

# Compile the model: define optimizer, loss function, and evaluation metric
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# ---------------------------------------------------------------------------
# 4. Model training
# ---------------------------------------------------------------------------
EPOCHS = 5          # 5 epochs is enough for MNIST to reach ~99% accuracy
BATCH_SIZE = 128

print("\nStarting training...")
history = model.fit(
    x_train, y_train_cat,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.1,   # use 10% of training data for validation
    verbose=1,
)

# ---------------------------------------------------------------------------
# 5. Model evaluation
# ---------------------------------------------------------------------------
print("\nEvaluating model on test data...")
test_loss, test_accuracy = model.evaluate(x_test, y_test_cat, verbose=0)

# ---------------------------------------------------------------------------
# 6. Test accuracy
# ---------------------------------------------------------------------------
print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Save the accuracy to a text file for easy reference
with open(os.path.join(RESULTS_DIR, "test_accuracy.txt"), "w") as f:
    f.write(f"Test Loss: {test_loss:.4f}\n")
    f.write(f"Test Accuracy: {test_accuracy * 100:.2f}%\n")

# ---------------------------------------------------------------------------
# 7. Accuracy and loss graphs
# ---------------------------------------------------------------------------
plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "accuracy_loss_graph.png"))
plt.close()
print("Saved accuracy/loss graph to results/accuracy_loss_graph.png")

# ---------------------------------------------------------------------------
# 8. Confusion matrix
# ---------------------------------------------------------------------------
# Get model predictions on the test set
y_pred_probs = model.predict(x_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)   # convert probabilities to class labels

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=range(10), yticklabels=range(10))
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
plt.close()
print("Saved confusion matrix to results/confusion_matrix.png")

# ---------------------------------------------------------------------------
# 9. Sample prediction visualization
# ---------------------------------------------------------------------------
# Show 10 random test images along with predicted vs actual labels
plt.figure(figsize=(12, 5))
random_indices = np.random.choice(len(x_test), 10, replace=False)

for i, idx in enumerate(random_indices):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[idx].reshape(28, 28), cmap="gray")
    color = "green" if y_pred[idx] == y_test[idx] else "red"
    plt.title(f"Pred: {y_pred[idx]}\nTrue: {y_test[idx]}", color=color)
    plt.axis("off")

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "sample_predictions.png"))
plt.close()
print("Saved sample predictions to results/sample_predictions.png")

# ---------------------------------------------------------------------------
# 10. Saving the trained model
# ---------------------------------------------------------------------------
model_path = os.path.join(MODELS_DIR, "mnist_cnn_model.h5")
model.save(model_path)
print(f"\nModel saved to {model_path}")

print("\nAll done! Check the 'results' folder for graphs and the 'models' folder for the saved model.")
