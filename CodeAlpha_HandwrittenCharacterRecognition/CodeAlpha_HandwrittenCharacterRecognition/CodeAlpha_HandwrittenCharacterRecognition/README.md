# CodeAlpha - Handwritten Digit Recognition Using CNN

**Internship:** CodeAlpha Machine Learning Internship  
**Task:** Task 3 - Handwritten Character Recognition  
**Dataset:** MNIST Handwritten Digits  
**Model:** Convolutional Neural Network (CNN)  
**Language / Framework:** Python, TensorFlow/Keras  

---

## Project Overview

This project builds a Convolutional Neural Network (CNN) that recognizes handwritten digits (0-9) using the MNIST dataset.

The project covers the complete machine learning workflow:

- Loading the MNIST dataset
- Preprocessing and normalizing image data
- Building a CNN model
- Training the model
- Evaluating model performance
- Visualizing accuracy and loss
- Generating a confusion matrix
- Generating sample predictions
- Saving the trained model
- Using the trained model to predict digits from images

The trained model achieved **99.17% test accuracy** on the MNIST test dataset.

---

## Project Structure

```text
CodeAlpha_HandwrittenCharacterRecognition/
│
├── src/
│   ├── train_model.py       # Loads data, builds, trains and evaluates the CNN
│   └── predict.py           # Loads the saved model and predicts a digit
│
├── models/
│   └── mnist_cnn_model.h5   # Trained CNN model
│
├── results/
│   ├── accuracy_loss_graph.png
│   ├── confusion_matrix.png
│   └── sample_predictions.png
│
├── requirements.txt         # Python dependencies
├── README.md
└── .gitignore
```

---

## Requirements

- Python 3.9 or later
- pip

The project uses:

- TensorFlow
- NumPy
- Matplotlib
- scikit-learn
- Pillow

---

## Installation

Open a terminal in the project's root folder.

### 1. Create a virtual environment (Recommended)

```bash
python -m venv venv
```

### 2. Activate the virtual environment

**Windows:**

```bash
venv\Scripts\activate
```

**macOS/Linux:**

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## How to Run

### 1. Train the Model

Run:

```bash
python src/train_model.py
```

This will:

- Download the MNIST dataset automatically on the first run
- Load 60,000 training images and 10,000 test images
- Normalize the pixel values
- Build the CNN architecture
- Train the model for 5 epochs
- Evaluate the model on the test dataset
- Generate accuracy and loss graphs
- Generate a confusion matrix
- Generate sample predictions
- Save the trained model

The trained model is saved as:

```text
models/mnist_cnn_model.h5
```

The generated results are saved inside:

```text
results/
```

---

### 2. Predict a Digit

The prediction script can be run without an image:

```bash
python src/predict.py
```

This runs a demonstration using a random image from the MNIST test dataset.

You can also provide an image:

```bash
python src/predict.py path/to/your_digit_image.png
```

Common image formats such as PNG, JPG, and JPEG are supported.

### Tips for Custom Images

For better results with custom images:

- Use a clear image of a single handwritten digit
- Make sure the digit is clearly visible
- Avoid excessive background or shadows
- Make the digit large enough to be recognized
- Use a simple, high-contrast image

The prediction script converts the input image to grayscale, resizes it to 28x28 pixels, and normalizes it before passing it to the CNN.

**Note:** The model is trained specifically on the MNIST dataset. Real-world photographs may produce lower accuracy because they can differ from MNIST images in background, lighting, scale, positioning, and writing style.

---

## Expected Training Output

After training, the terminal displays information similar to:

```text
Loading MNIST dataset...
Training data shape: (60000, 28, 28)
Test data shape: (10000, 28, 28)
Data preprocessing complete.

Starting training...

Epoch 1/5
...

Epoch 5/5
...

Evaluating model on test data...

Test Loss: 0.0241
Test Accuracy: 99.17%

Model saved to models/mnist_cnn_model.h5
```

The exact training values may vary slightly depending on the environment.

---

## Generated Results

The following files are generated inside the `results/` folder:

| File | Description |
|------|-------------|
| `accuracy_loss_graph.png` | Shows training and validation accuracy and loss |
| `confusion_matrix.png` | Shows correct and incorrect predictions for each digit |
| `sample_predictions.png` | Displays sample MNIST images with predicted and true labels |

---

## Trained Model

The trained model is saved inside the `models/` folder:

| File | Description |
|------|-------------|
| `mnist_cnn_model.h5` | Saved CNN model that can be loaded for predictions |

---

## How the CNN Works

A Convolutional Neural Network (CNN) is designed to recognize patterns in images.

Instead of treating every pixel independently, a CNN uses filters to detect useful visual patterns such as:

- Edges
- Curves
- Corners
- Strokes
- Shapes

This project uses the following CNN architecture:

### 1. Conv2D - 32 Filters

The first convolutional layer detects basic visual features such as edges and strokes.

### 2. MaxPooling

MaxPooling reduces the size of the feature maps while retaining important information.

### 3. Conv2D - 64 Filters

The second convolutional layer detects more complex patterns by combining simpler features.

### 4. MaxPooling

Another pooling layer reduces the feature-map size and keeps the strongest features.

### 5. Flatten

The extracted 2D feature maps are converted into a one-dimensional list of values.

### 6. Dropout

Dropout randomly disables some neurons during training. This helps reduce overfitting and improves generalization.

### 7. Dense Layer - 128 Neurons

This layer combines the extracted features to make a classification decision.

### 8. Dense Layer - 10 Neurons

The final layer contains 10 neurons, one for each digit from 0 to 9.

A softmax activation function converts the outputs into probabilities. The digit with the highest probability becomes the final prediction.

---

## Data Preprocessing

The MNIST images are grayscale images with a size of 28x28 pixels.

Before training, the pixel values are normalized from:

```text
0 - 255
```

to:

```text
0 - 1
```

This helps the neural network train more efficiently and stably.

The image data is also reshaped to include the required channel dimension for the CNN.

---

## Model Evaluation

The model is evaluated using the separate MNIST test dataset containing 10,000 images that were not used during training.

The project uses:

### Test Accuracy

Test accuracy measures the percentage of test images classified correctly.

The final test accuracy achieved in this project was:

**99.17%**

### Test Loss

Test loss measures the difference between the model's predictions and the actual labels.

Final test loss:

**0.0241**

### Confusion Matrix

The confusion matrix compares the predicted digit with the actual digit.

- Diagonal values represent correct predictions.
- Off-diagonal values represent incorrect predictions.

This helps identify which digits are most commonly confused with one another.

---

## Questions an Evaluator Might Ask

### Q1: Why did you use a CNN instead of a regular neural network?

**Answer:** CNNs are designed for image-related tasks. They use filters to detect spatial patterns such as edges, curves, and shapes, making them suitable for handwritten digit recognition.

### Q2: What does normalization do?

**Answer:** Normalization changes pixel values from the range 0-255 to the range 0-1. This helps the neural network train faster and more stably.

### Q3: What is MaxPooling?

**Answer:** MaxPooling reduces the size of feature maps by keeping the strongest value from each small region. This reduces computation while retaining important features.

### Q4: What is Dropout?

**Answer:** Dropout randomly disables some neurons during training. This reduces overfitting and helps the model generalize better to unseen data.

### Q5: What is softmax used for?

**Answer:** Softmax converts the final output values into probabilities for the ten digit classes, from 0 to 9. The class with the highest probability becomes the prediction.

### Q6: How did you evaluate the model?

**Answer:** The model was evaluated using the separate MNIST test dataset containing 10,000 images that were not used during training. Test accuracy, test loss, and a confusion matrix were used to evaluate the model.

### Q7: What is a confusion matrix?

**Answer:** A confusion matrix compares the predicted labels with the actual labels. The diagonal represents correct predictions, while off-diagonal values represent incorrect predictions.

### Q8: What accuracy did your model achieve?

**Answer:** The model achieved **99.17% test accuracy** after 5 training epochs on the MNIST test dataset.

### Q9: How could you improve the model further?

**Answer:** Possible improvements include data augmentation, hyperparameter tuning, additional convolutional layers, batch normalization, and training for more epochs.

### Q10: Can the model recognize digits from real photographs?

**Answer:** The prediction script accepts common image formats such as PNG, JPG, and JPEG. However, the model is trained on MNIST images, so its performance on real-world photographs may be lower because photographs can differ in background, lighting, scale, positioning, and writing style.

---

## Limitations

Although the model achieved high accuracy on the MNIST dataset, there are some limitations:

- The model is trained specifically on MNIST-style handwritten digits.
- Real-world photographs may have different backgrounds and lighting.
- Digit positioning and scale can affect predictions.
- Different handwriting styles may affect recognition.
- The current model recognizes digits from 0 to 9 and does not recognize alphabetic characters.

---

## Future Improvements

The project could be improved by:

- Adding better preprocessing for real-world photographs
- Automatically detecting and cropping the handwritten digit
- Applying data augmentation
- Training on additional handwriting datasets
- Adding a graphical user interface
- Supporting handwritten alphabet characters
- Deploying the model as a web or mobile application

---

## Project Results

The final CNN achieved:

```text
Test Accuracy: 99.17%
Test Loss: 0.0241
Training Epochs: 5
```

The project also generates:

- Accuracy and loss curves
- Confusion matrix
- Sample predictions
- Saved trained CNN model

These results demonstrate that the CNN can effectively classify handwritten digits from the MNIST dataset.

---

## CodeAlpha Submission Checklist

- [ ] Code runs without errors from a fresh environment
- [ ] `train_model.py` trains the CNN and prints test accuracy
- [ ] `results/` contains the accuracy/loss graph
- [ ] `results/` contains the confusion matrix
- [ ] `results/` contains sample predictions
- [ ] `models/mnist_cnn_model.h5` is generated after training
- [ ] `predict.py` successfully predicts MNIST test images
- [ ] Code contains clear comments
- [ ] README.md is complete and professional
- [ ] Project uploaded to a public GitHub repository
- [ ] GitHub repository link is ready for submission

---

## Creating the GitHub Repository

When the project is ready to upload:

1. Go to GitHub and log in.
2. Click the **+** icon in the top-right corner.
3. Select **New repository**.
4. Name the repository:

```text
CodeAlpha_HandwrittenCharacterRecognition
```

5. Set the repository visibility to **Public** if required by the internship.
6. Do not initialize the repository with another README because this project already contains one.
7. Create the repository.

---

## Git Commands to Upload the Project

Run these commands from the project's root folder:

```bash
git init
git add .
git commit -m "Initial commit: Handwritten Digit Recognition CNN project"
git branch -M main
git remote add origin https://github.com/<your-username>/CodeAlpha_HandwrittenCharacterRecognition.git
git push -u origin main
```

Replace `<your-username>` with your GitHub username.

---

## Author

**Drisya M.**

Completed as part of the **CodeAlpha Machine Learning Internship**.