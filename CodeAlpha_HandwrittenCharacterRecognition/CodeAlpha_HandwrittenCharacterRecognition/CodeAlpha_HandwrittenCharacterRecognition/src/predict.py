"""
predict.py
----------------
CodeAlpha Machine Learning Internship - Task 3
Simple prediction interface for the trained MNIST CNN model.

This script lets you provide the path to a handwritten digit image
(png/jpg) and get the model's predicted digit (0-9).

Usage:
    python predict.py path/to/your/image.png

If no path is given, the script will automatically pick a random
image from the MNIST test set so you can see it working immediately.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import tensorflow as tf

# ---------------------------------------------------------------------------
# Setup paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "mnist_cnn_model.h5")


def load_trained_model():
    """Load the trained CNN model from disk."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}.\n"
            "Please run 'python src/train_model.py' first to train and save the model."
        )
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess_custom_image(image_path):
    """
    Load a user-provided image and convert it into the same format
    the model was trained on: 28x28 grayscale, normalized, white
    digit on black background.
    """
    # Open image and convert to grayscale
    img = Image.open(image_path).convert("L")

    # Resize to 28x28 (same size as MNIST digits)
    img = img.resize((28, 28))

    # MNIST digits are white strokes on a black background.
    # Most photos/scans of handwriting are black strokes on a white
    # background, so we invert the colors to match MNIST's style.
    # (If your image already has a black background, you can comment
    # out the next line.)
    img = ImageOps.invert(img)

    # Convert to numpy array and normalize to [0, 1]
    img_array = np.array(img).astype("float32") / 255.0

    # Reshape to match model input: (1, 28, 28, 1)
    img_array = img_array.reshape(1, 28, 28, 1)

    return img_array, img


def predict_from_mnist_test_set(model):
    """Fallback demo: pick a random MNIST test image and predict it."""
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    idx = np.random.randint(0, len(x_test))

    image = x_test[idx].astype("float32") / 255.0
    true_label = y_test[idx]

    input_array = image.reshape(1, 28, 28, 1)
    prediction = model.predict(input_array, verbose=0)
    predicted_digit = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    plt.imshow(image, cmap="gray")
    plt.title(f"Predicted: {predicted_digit} (True: {true_label}, Confidence: {confidence:.1f}%)")
    plt.axis("off")
    plt.show()

    print(f"Predicted digit: {predicted_digit}")
    print(f"True label: {true_label}")
    print(f"Confidence: {confidence:.2f}%")


def predict_from_image_file(model, image_path):
    """Predict the digit in a user-supplied image file."""
    input_array, display_img = preprocess_custom_image(image_path)

    prediction = model.predict(input_array, verbose=0)
    predicted_digit = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    plt.imshow(display_img, cmap="gray")
    plt.title(f"Predicted: {predicted_digit} (Confidence: {confidence:.1f}%)")
    plt.axis("off")
    plt.show()

    print(f"Predicted digit: {predicted_digit}")
    print(f"Confidence: {confidence:.2f}%")


if __name__ == "__main__":
    print("Loading trained model...")
    model = load_trained_model()

    if len(sys.argv) > 1:
        # User provided an image path as a command-line argument
        image_path = sys.argv[1]
        if not os.path.exists(image_path):
            print(f"Error: file not found at '{image_path}'")
            sys.exit(1)
        print(f"Predicting digit for image: {image_path}")
        predict_from_image_file(model, image_path)
    else:
        # No image provided, demo with a random MNIST test image
        print("No image path provided. Running demo with a random MNIST test image.")
        print("Tip: run 'python predict.py path/to/your_digit.png' to test your own image.")
        predict_from_mnist_test_set(model)
