# Leaf Classification Using Machine Learning and Deep Learning

## 📌 Project Overview

This project explores automated leaf image classification using two machine learning approaches: a traditional machine learning pipeline based on handcrafted image features and Support Vector Machine (SVM), and a deep learning approach using MobileNetV2 transfer learning.

The objective is to compare the performance of these approaches on a leaf image dataset and explore confidence-based prediction for images that may not belong to the model's known classes.

## 🎯 Objectives

- Classify leaf images using visual features.
- Implement a traditional machine learning model using HOG features, RGB colour histograms, and SVM.
- Build a deep learning classifier using MobileNetV2 transfer learning.
- Compare the models using accuracy, macro-F1 score, macro recall, classification reports, and confusion matrices.
- Inspect misclassified images and evaluate predictions on new images.
- Explore confidence-based detection of potentially unknown classes.
- Save trained models and evaluation outputs for future use.

## 🧠 Models Used

### 1. HOG + RGB Colour Histograms + SVM

The traditional machine learning pipeline extracts several features from each image:

- **RGB colour histograms:** Capture the distribution of red, green, and blue pixel intensities.
- **Histogram of Oriented Gradients (HOG):** Captures local image gradients, edges, and shape-related information.
- **Grayscale statistics:** Use mean and standard deviation to describe image intensity.

The extracted features are standardised using `StandardScaler` and classified using an RBF-kernel Support Vector Classifier with probability estimation and balanced class weights.

### 2. MobileNetV2 Transfer Learning

MobileNetV2 is used as a pretrained convolutional neural network with ImageNet weights.

The notebook:

- Resizes images to 160 × 160 pixels.
- Applies MobileNetV2-specific input preprocessing.
- Freezes the pretrained convolutional base.
- Adds global average pooling, dropout, and a dense softmax classification layer.
- Trains the classification head for up to 10 epochs, with early stopping based on validation loss.

This approach uses pretrained visual representations to classify leaf images without training the entire network from scratch.

## 📂 Dataset and Preprocessing

The notebook supports image datasets organised into class-specific folders, with images in formats such as JPG, JPEG, PNG, BMP, and WebP.

The workflow includes:

1. Extracting ZIP archives.
2. Detecting folders containing images.
3. Inspecting class counts, image dimensions, colour information, and sample images.
4. Removing unreadable images.
5. Creating stratified training, validation, and test sets using a fixed random seed.

The split is approximately:

- **70% — Training**
- **15% — Validation**
- **15% — Testing**

The notebook selects up to five known classes automatically by folder size unless `KNOWN_CLASSES` is explicitly configured. Other discovered image folders are assigned the `unknown` label. For reproducible experiments, configure `KNOWN_CLASSES` with the exact intended species names.

## 📊 Model Evaluation

Both models are evaluated on the test set using:

- Accuracy
- Macro-F1 score
- Macro recall
- Classification reports
- Confusion matrices

The notebook also compares incorrect predictions and displays example images where the models make mistakes.

Actual evaluation scores should be taken from the outputs generated when the notebook is executed with the intended dataset.

## 🔎 Unknown-Leaf Prediction

The project includes a basic confidence-based prediction mechanism for new images.

A prediction is labelled as a low-confidence unknown when its highest class probability falls below a threshold estimated from validation predictions.

**Limitation:** This is a heuristic, not a dedicated unknown-species classifier. A high-confidence prediction can still be wrong, and a low-confidence prediction can belong to a known species. Reliable unknown-class evaluation requires representative unknown leaves and, ideally, other out-of-distribution images that were not used for training.

## 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Pillow (PIL)
- scikit-image
- scikit-learn
- TensorFlow / Keras
- Joblib
- Jupyter Notebook

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/nooby30/Leaf_Classification_Final.git
cd Leaf_Classification_Final
```

### 2. Install the dependencies

```bash
pip install numpy pandas matplotlib seaborn pillow scikit-image scikit-learn joblib tensorflow
```

### 3. Prepare the dataset

Place the leaf dataset ZIP file in the working directory, or upload it to the location used by your notebook.

The notebook contains both local-path and Google Colab-style paths. Update the ZIP and extraction paths to match your environment before running the relevant cells.

### 4. Run the notebook

Open `Data_Science_project.ipynb` in Jupyter Notebook or Google Colab and execute the cells in order.

The first MobileNetV2 run may require an internet connection to obtain pretrained ImageNet weights.

### 5. Evaluate and save the models

The notebook generates evaluation metrics and visualisations and saves model artefacts in a `project_outputs/` directory.

## 💾 Saved Outputs

The notebook is configured to save the following files inside `project_outputs/`:

| File | Description |
|---|---|
| `hog_colour_svm.joblib` | Trained SVM pipeline |
| `mobilenetv2_leaf_classifier.keras` | Trained MobileNetV2 classifier |
| `model_notes.json` | Class labels, confidence thresholds, image sizes, and notes |
| `model_comparison.csv` | Comparison of evaluation metrics |
| `leaf_metadata.csv` | Image paths and class metadata |

It also saves training, validation, and test metadata CSV files in the notebook's working directory.

These files are generated when the relevant notebook cells run successfully; they are not necessarily included in the repository.

## ⚠️ Limitations

- Performance depends on dataset quality, class balance, and image diversity.
- Images from the same source or near-duplicates across different splits can inflate evaluation scores.
- Performance on the test set does not guarantee similar results on real-world photographs.
- Confidence thresholds do not guarantee reliable unknown-species detection.
- The automatic selection of the five largest folders should be checked before training.

## 🔮 Future Improvements

- Fine-tune selected MobileNetV2 layers.
- Add more leaf species and diverse image conditions.
- Check for duplicate or near-duplicate images across dataset splits.
- Evaluate on genuinely independent images from different sources.
- Improve unknown-class detection using dedicated unknown examples and calibrated thresholds.
- Build a simple interface for uploading a leaf image and viewing predictions.

