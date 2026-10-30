# 🍃 Leaf Classification Using Machine Learning and Deep Learning

### A Comparative Study of Traditional Machine Learning and Deep Learning Approaches for Leaf Image Classification

---

## 1. Project Overview

Plant identification through leaf morphology is an important application of computer vision and image processing. Characteristics such as leaf colour, shape, texture, and structural patterns can provide useful information for distinguishing plant categories.

This project focuses on developing an automated leaf classification system using two different approaches:

1. **Traditional Machine Learning:** HOG (Histogram of Oriented Gradients), RGB Colour Histograms, and Support Vector Machine (SVM).
2. **Deep Learning:** MobileNetV2-based Transfer Learning using pretrained ImageNet weights.

The primary objective is to compare both approaches and investigate their effectiveness in classifying leaf images. The project also explores confidence-based prediction to identify uncertain or potentially unfamiliar images.

## 2. Project Objectives

- Develop an automated system for leaf image classification.
- Extract meaningful visual features using colour histograms and HOG.
- Implement a traditional machine learning classifier using SVM.
- Apply transfer learning using MobileNetV2.
- Compare the performance of traditional machine learning and deep learning models.
- Evaluate models using accuracy, precision, recall, and F1-score.
- Visualise classification performance using confusion matrices.
- Explore confidence-based prediction for unknown or unfamiliar images.

## 3. Dataset Description

The dataset consists of **327 RGB leaf images** distributed across six categories, including five known categories and one unknown category.

### Dataset Distribution

| Category | Number of Images |
|---|---:|
| Mango selected | 62 |
| Lembu | 60 |
| Coccinia grandis selected | 54 |
| Codiaeum variegatum selected | 51 |
| Herbarium zip | 50 |
| Unknown | 50 |
| **Total** | **327** |

### Dataset Characteristics

| Property | Description |
|---|---|
| Total images | 327 |
| Number of categories | 6 |
| Known categories | 5 |
| Unknown category | 1 |
| Image dimensions | 512 × 512 pixels |
| Image format | RGB |

All images were checked for readability before model development.

### Dataset Split

Stratified sampling was used to maintain the distribution of categories across the training, validation, and testing sets.

| Dataset | Images | Percentage |
|---|---:|---:|
| Training | 228 | 69.72% |
| Validation | 49 | 14.98% |
| Testing | 50 | 15.30% |
| **Total** | **327** | **100%** |

A fixed random seed of 42 was used for reproducibility.

## 4. Data Preprocessing and Exploration

The dataset was processed using Python libraries such as Pillow, NumPy, and Pandas.

The preprocessing workflow included:

- Extracting image datasets from ZIP archives.
- Automatically identifying class folders.
- Assigning labels to known and unknown categories.
- Checking image readability.
- Analysing image dimensions and RGB colour information.
- Removing unreadable images.
- Creating stratified training, validation, and testing datasets.

### Important Code: Dataset Preparation

```python
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def direct_images(folder):
    return [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]

# Identify folders containing images
class_dirs = []

for folder in EXTRACT_DIR.rglob("*"):
    if folder.is_dir() and direct_images(folder):
        class_dirs.append(folder)

# Select five known categories
ranked = sorted(
    normal_dirs,
    key=lambda p: len(direct_images(p)),
    reverse=True
)

known_dirs = ranked[:5]
```

The notebook automatically selected the five largest non-unknown image folders as known categories. Other folders were mapped to the unknown label.

### Important Code: Stratified Train-Test Split

```python
train_df, temp_df = train_test_split(
    metadata,
    test_size=0.30,
    random_state=42,
    stratify=metadata["label"]
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["label"]
)

print("Training:", len(train_df))
print("Validation:", len(val_df))
print("Testing:", len(test_df))
```

This resulted in 228 training images, 49 validation images, and 50 testing images.

## 5. Traditional Machine Learning Approach: HOG + RGB + SVM

The first classification approach uses handcrafted image features to represent the visual characteristics of leaves.

### 5.1 Feature Extraction

Three types of features were extracted from each image:

**RGB Colour Histograms**

- Capture the distribution of red, green, and blue intensities.
- Use 16 bins for each colour channel.
- Generate 48 colour features.

**Histogram of Oriented Gradients (HOG)**

- Extracts local gradient and edge information.
- Captures structural patterns and leaf outlines.
- Uses 9 orientations, 8 × 8 pixels per cell, and 2 × 2 cells per block.

**Grayscale Statistics**

- Mean grayscale intensity.
- Standard deviation of grayscale intensity.

The combined feature vector contains **8,150 features per image**.

### Important Code: HOG and RGB Feature Extraction

```python
IMG_SIZE = (128, 128)

def extract_features(image_path):

    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMG_SIZE)

    rgb = np.asarray(image, dtype=np.uint8)
    features = []

    # RGB colour histograms
    for channel in range(3):

        hist, _ = np.histogram(
            rgb[:, :, channel],
            bins=16,
            range=(0, 256)
        )

        hist = hist.astype(np.float32)
        hist = hist / max(hist.sum(), 1)

        features.extend(hist)

    # HOG feature extraction
    gray = rgb2gray(rgb)

    hog_features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True
    )

    features.extend(hog_features)

    # Grayscale statistics
    features.extend([
        gray.mean(),
        gray.std()
    ])

    return np.asarray(features, dtype=np.float32)
```

### 5.2 Support Vector Machine Classification

The extracted features were standardised using `StandardScaler` and classified using a Support Vector Machine with an RBF kernel.

Balanced class weights were used to account for differences in class representation.

### Important Code: SVM Model

```python
svm_model = Pipeline([
    ("scaler", StandardScaler()),

    ("classifier", SVC(
        C=3.0,
        kernel="rbf",
        probability=True,
        class_weight="balanced",
        random_state=42
    ))
])

# Train the model
svm_model.fit(X_train_feat, y_train)

# Generate validation predictions
val_pred_svm = svm_model.predict(X_val_feat)

print("Validation Accuracy:",
      accuracy_score(y_val, val_pred_svm))
```

The SVM model was evaluated using accuracy and macro-F1 score.

## 6. Deep Learning Approach: MobileNetV2 Transfer Learning

The second approach uses MobileNetV2, a convolutional neural network pretrained on ImageNet.

Transfer learning allows the model to utilise previously learned visual representations rather than learning all image features from scratch.

### 6.1 Image Preprocessing

- Images are resized to 160 × 160 pixels.
- RGB channels are retained.
- MobileNetV2-specific preprocessing is applied.
- Images and labels are converted into TensorFlow datasets.
- Training data is shuffled and batched.

### 6.2 Model Architecture

The architecture consists of:

1. Input layer: 160 × 160 × 3.
2. Pretrained MobileNetV2 convolutional base.
3. Global Average Pooling layer.
4. Dropout layer.
5. Dense classification layer with six neurons.
6. Softmax activation for multiclass prediction.

### Important Code: MobileNetV2 Model

```python
IMG_SHAPE = (160, 160)

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(160, 160, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze pretrained layers
base_model.trainable = False

inputs = keras.Input(shape=(160, 160, 3))

x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.25)(x)

outputs = layers.Dense(
    len(class_names),
    activation="softmax"
)(x)

cnn_model = keras.Model(inputs, outputs)

cnn_model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
```

The pretrained MobileNetV2 backbone was kept frozen, while the newly added classification head was trained for the six categories.

### 6.3 Model Training and Early Stopping

The model was trained for a maximum of 10 epochs. Early stopping monitored validation loss and restored the best-performing weights.

### Important Code: Training

```python
early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

history = cnn_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=[early_stop],
    verbose=1
)
```

## 7. Model Evaluation

Both models were evaluated using the same held-out test dataset containing 50 images.

The following evaluation metrics were considered:

- **Accuracy:** Proportion of correctly classified images.
- **Precision:** Proportion of predicted class instances that are correct.
- **Recall:** Proportion of actual class instances correctly identified.
- **F1-score:** Harmonic mean of precision and recall.
- **Macro-F1:** Average F1-score across all categories, giving equal importance to each class.

### Important Code: Model Comparison

```python
# SVM predictions
pred_svm = svm_model.predict(X_test_feat)

# MobileNetV2 predictions
prob_cnn = cnn_model.predict(test_ds, verbose=0)

cnn_ids = np.argmax(prob_cnn, axis=1)
pred_cnn = np.array(class_names)[cnn_ids]

# Compare evaluation metrics
results = pd.DataFrame([
    {
        "Model": "HOG + RGB + SVM",
        "Accuracy": accuracy_score(y_test, pred_svm),
        "Macro-F1": f1_score(
            y_test, pred_svm,
            average="macro",
            zero_division=0
        )
    },
    {
        "Model": "MobileNetV2",
        "Accuracy": accuracy_score(y_test, pred_cnn),
        "Macro-F1": f1_score(
            y_test, pred_cnn,
            average="macro",
            zero_division=0
        )
    }
])

print(results.round(4))
```

### Overall Performance Comparison

| Evaluation Metric | HOG + RGB + SVM | MobileNetV2 |
|---|---:|---:|
| Accuracy | 90.00% | **96.00%** |
| Macro-F1 Score | 89.82% | **96.14%** |
| Macro Recall | 89.70% | **96.06%** |

### Test Accuracy

- HOG + RGB + SVM: 90% — 45 out of 50 images correctly classified.
- MobileNetV2: 96% — 48 out of 50 images correctly classified.

### Macro-F1 Comparison

| Model | Macro-F1 |
|---|---:|
| HOG + RGB + SVM | 89.82% |
| MobileNetV2 | **96.14%** |

MobileNetV2 outperformed the traditional SVM model across all three aggregate evaluation metrics.

## 8. Class-wise Performance Analysis

Classification reports were generated to evaluate the performance of both models across individual categories.

### HOG + RGB + SVM

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Codiaeum variegatum selected | 1.00 | 1.00 | 1.00 |
| Herbarium zip | 0.86 | 0.86 | 0.86 |
| Lembu | 0.82 | 1.00 | 0.90 |
| Mango selected | 0.90 | 0.90 | 0.90 |
| Coccinia grandis selected | 0.88 | 0.88 | 0.88 |
| Unknown | 1.00 | 0.75 | 0.86 |

### MobileNetV2

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Codiaeum variegatum selected | 1.00 | 1.00 | 1.00 |
| Herbarium zip | 1.00 | 1.00 | 1.00 |
| Lembu | 1.00 | 0.89 | 0.94 |
| Mango selected | 0.91 | 1.00 | 0.95 |
| Coccinia grandis selected | 1.00 | 1.00 | 1.00 |
| Unknown | 0.88 | 0.88 | 0.88 |

MobileNetV2 achieved an F1-score of 1.00 for three categories, demonstrating strong classification performance on this test split.

### Confusion Matrix Analysis

Confusion matrices were generated to visualise the number of correct and incorrect predictions for each category.

The diagonal elements represent correctly classified images, while off-diagonal elements indicate misclassifications.

```python
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax, pred, title in [
    (axes[0], pred_svm, "SVM Confusion Matrix"),
    (axes[1], pred_cnn, "MobileNetV2 Confusion Matrix")
]:

    cm = confusion_matrix(
        y_test,
        pred,
        labels=class_names
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    disp.plot(
        ax=ax,
        xticks_rotation=90,
        colorbar=False,
        values_format="d"
    )

    ax.set_title(title)

plt.tight_layout()
plt.show()
```

## 9. Training Performance

MobileNetV2 training and validation accuracy were monitored across 10 epochs.

| Epoch | Training Accuracy | Validation Accuracy |
|---:|---:|---:|
| 1 | 41.23% | 77.55% |
| 2 | 82.46% | 89.80% |
| 3 | 91.67% | 91.84% |
| 4 | 97.37% | 91.84% |
| 5 | 95.61% | 95.92% |
| 6 | 98.25% | 95.92% |
| 7 | 98.25% | 95.92% |
| 8 | 99.12% | 100.00% |
| 9 | 97.81% | 97.96% |
| 10 | 99.56% | 100.00% |

The final recorded validation accuracy was 100%, with a validation loss of 0.0818.

The relatively small validation set should be considered when interpreting this result.

## 10. Unknown Leaf Detection

An additional component of the project explores confidence-based prediction for unfamiliar or uncertain images.

The maximum prediction probability is calculated for each image. A confidence threshold is estimated using the 10th percentile of maximum validation probabilities.

| Model | Confidence Threshold |
|---|---:|
| SVM | 0.390 |
| MobileNetV2 | 0.808 |

### Important Code: Confidence-Based Prediction

```python
# Calculate validation probabilities
val_prob_svm = svm_model.predict_proba(X_val_feat)
val_prob_cnn = cnn_model.predict(val_ds, verbose=0)

# Estimate confidence thresholds
svm_threshold = np.quantile(
    val_prob_svm.max(axis=1), 0.10
)

cnn_threshold = np.quantile(
    val_prob_cnn.max(axis=1), 0.10
)

def predict_with_unknown_cnn(image_path):

    img = Image.open(image_path).convert("RGB")
    img = img.resize(IMG_SHAPE)

    arr = np.asarray(img, dtype=np.float32)

    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)

    probs = cnn_model.predict(
        arr[None, ...], verbose=0
    )[0]

    idx = np.argmax(probs)
    confidence = float(probs[idx])

    if confidence >= cnn_threshold:
        label = class_names[idx]
    else:
        label = "unknown (low confidence)"

    return label, confidence
```

This approach provides a basic confidence-based rejection mechanism. However, it is a heuristic rather than a dedicated open-set recognition model. A high-confidence prediction can still be incorrect for an unfamiliar species or a non-leaf image.

## 11. Model Saving and Export

The trained models and evaluation outputs are saved for future use.

### Important Code: Saving Models

```python
OUTPUT_DIR = Path("project_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Save SVM model
joblib.dump(
    svm_model,
    OUTPUT_DIR / "hog_colour_svm.joblib"
)

# Save MobileNetV2 model
cnn_model.save(
    OUTPUT_DIR / "mobilenetv2_leaf_classifier.keras"
)

# Save evaluation results
results.to_csv(
    OUTPUT_DIR / "model_comparison.csv",
    index=False
)
```

### Generated Files

| File | Description |
|---|---|
| hog_colour_svm.joblib | Trained SVM pipeline |
| mobilenetv2_leaf_classifier.keras | Trained MobileNetV2 model |
| model_notes.json | Model configuration and confidence thresholds |
| model_comparison.csv | Performance comparison |
| leaf_metadata.csv | Dataset image paths and labels |

## 12. Technologies and Libraries Used

| Technology | Application |
|---|---|
| Python | Core programming language |
| NumPy | Numerical operations |
| Pandas | Dataset handling |
| Matplotlib | Data visualisation |
| Seaborn | Statistical visualisation |
| Pillow | Image loading and processing |
| Scikit-image | HOG feature extraction |
| Scikit-learn | SVM and model evaluation |
| TensorFlow / Keras | Deep learning and transfer learning |
| Joblib | Model saving |
| Jupyter Notebook | Development environment |

## 13. How to Run the Project

### Clone the repository

```bash
git clone https://github.com/nooby30/Leaf_Classification_Final.git
cd Leaf_Classification_Final
```

### Install the required libraries

```bash
pip install numpy pandas matplotlib seaborn pillow scikit-image scikit-learn joblib tensorflow
```

### Prepare the dataset

Place the dataset ZIP files in the working directory expected by the notebook.

The current notebook uses `/content` for its ZIP extraction workflow. When running locally in Jupyter Notebook, update the dataset paths to match your computer.

### Run the notebook

Open `Data_Science_project.ipynb` using Jupyter Notebook or Google Colab and execute the cells sequentially.

An internet connection is required during the first MobileNetV2 run to download pretrained ImageNet weights.

## 14. Limitations and Future Scope

Although the project demonstrates promising classification performance, several improvements can be explored:

- Increasing the dataset size and diversity.
- Testing images collected under different lighting and background conditions.
- Checking for duplicate and near-duplicate images across dataset splits.
- Fine-tuning selected MobileNetV2 layers.
- Exploring alternative CNN architectures such as EfficientNet and ResNet.
- Developing more reliable open-set recognition techniques.
- Building an interface for uploading leaf images and obtaining predictions.
- Evaluating the models using larger, independently collected datasets.

## 15. Conclusion

This project presents a comparative study of traditional machine learning and deep learning approaches for leaf image classification.

The HOG and RGB colour histogram-based SVM model achieved 90% test accuracy, demonstrating that handcrafted visual features can provide a useful baseline for image classification.

MobileNetV2 transfer learning achieved 96% test accuracy and a macro-F1 score of 96.14%, outperforming the traditional machine learning approach on the evaluated test dataset.

The results demonstrate the effectiveness of pretrained convolutional neural networks in extracting useful visual representations for leaf classification.

The project also explores confidence-based prediction for uncertain images, providing a foundation for further investigation into unfamiliar species recognition.

Overall, this work highlights the potential of combining computer vision, machine learning, and deep learning techniques for automated plant image classification.

---

## Author

**Soumyadip Barik,**
**Bakshish Kaur,**
**Khushi Malhotra**

GitHub: [@nooby30](https://github.com/nooby30)

Project Repository: [Leaf Classification Using Machine Learning and Deep Learning](https://github.com/nooby30/Leaf_Classification_Final)
