# **Skin Lesion Classification: Hybrid CNN and Tree-Based Model**

This project builds a hybrid machine learning pipeline for classifying skin lesions as benign or malignant. The pipeline combines convolutional neural networks (CNNs) and tree-based models to achieve competitive performance. The final LightGBM model achieves a **partial AUC (pAUC) of 0.1768**, surpassing the Kaggle leaderboard score of 0.1762.

---

## Overview

The project is organized into three key notebooks:

1. **KFoldTraining**: Training and generating out-of-fold (OOF) predictions using 5 CNN models.
2. **Trees**: Building and tuning tree-based models (LightGBM and XGBoost) using the CNN-generated OOF predictions and metadata.
3. **FullData**: Extending the hybrid approach to the full training dataset for final model optimization.

---

## Dataset Details

- **Original Dataset**:
  - Was too big to be added onto GitHub directly. It can be accessed through this Drive Link: https://drive.google.com/drive/folders/1DU-pHs5fyZ2MszKWATh1lH1uProMpik1?usp=share_link/
  - Highly imbalanced with **~400,000 benign samples** and **~400 malignant samples**.
  - Severe class imbalance poses significant challenges for model training and evaluation.
  
- **Resampled Dataset**:
  - **Negative Samples**: Downsampled to **40,000**.
  - **Positive Samples**: Upsampled to **4,000** using data augmentation.
  - This balanced dataset (40,000 vs. 4,000) is used for CNN training, enabling the model to better learn features of malignant lesions.

---

## Model Architecture

The base model used for feature extraction is **MobileNetV0**, a lightweight, efficient architecture pretrained on ImageNet. The model is fine-tuned with additional layers to adapt it for binary classification:

1. **Global Average Pooling Layer**:
   - Reduces the spatial dimensionality of the feature maps.

2. **Fully Connected Layers**:
   - Dense Layer 1: 
     - 512 units, ReLU activation, He normal initializer.
     - Batch normalization and dropout (rate: 0.4).
   - Dense Layer 2:
     - 256 units, ReLU activation, He normal initializer.
     - Batch normalization and dropout (rate: 0.5).

3. **Output Layer**:
   - 1 unit, sigmoid activation for binary classification.

---

## Custom Data Generators

The project uses **custom data generators** for efficient data loading and augmentation:

1. **Training Data Generator**:
   - Balances the dataset by applying heavy augmentation to malignant samples.
   - Augmentation techniques include:
     - **Rotations**: Up to ±30 degrees.
     - **Horizontal/Vertical Flips**.
     - **Random Cropping** and **Scaling**.
   - Augmentation ensures the model is exposed to diverse malignant patterns, improving generalization.

2. **Validation Data Generator**:
   - Uses the original class distribution without augmentation to provide an unbiased evaluation of model performance.

These generators handle large datasets (400,000+ samples) efficiently and are implemented to dynamically load and preprocess images in batches, avoiding memory bottlenecks.

---

## Notebooks

### 1. **KFoldTraining.ipynb**

This notebook implements the training of 5 CNN models using **Stratified K-Fold Cross-Validation**:
- **Objective**: Generate unbiased OOF predictions for each training sample.
- **Highlights**:
  - **Custom Sampling**: Creates a balanced dataset (40,000 vs. 4,000) for training.
  - **CNN Fine-Tuning**:
    - Initial training of fully connected layers.
    - Gradual unfreezing of MobileNetV0 layers in multiple phases.
    - Cosine annealing learning rate schedule with warm restarts.
  - **OOF Predictions**: Ensures predictions are unbiased by training each model on folds 1-4 and predicting on fold 5.
  - **pAUC Scores**: Partial AUC scores for each model are logged for comparison.

---

### 2. **Trees.ipynb**

This notebook focuses on building and tuning two tree-based models:
- **Objective**: Train LightGBM and XGBoost models on metadata and CNN-generated OOF predictions.
- **Highlights**:
  - **Input Features**:
    - Metadata (age, sex, anatomical site, etc.).
    - CNN OOF predictions (with Gaussian noise added to prevent overfitting).
  - **Model Architecture**:
    - Each tree model is optimized using a voting classifier.
  - **Optuna Optimization**: 
    - Hyperparameter tuning for both LightGBM and XGBoost.
    - Efficient search for optimal parameters like learning rate, max depth, and regularization.
  - **pAUC Scores**:
    - **LightGBM**: ~0.17.
    - **XGBoost**: Similar performance (~0.17).

---

### 3. **FullData.ipynb**

This notebook extends the hybrid approach to the entire dataset:
- **Objective**: Train a final hybrid model using all available data for maximum generalization.
- **Highlights**:
  - **CNN Predictions**:
    - Predictions for all samples (~400,000 benign and ~400 malignant) are generated using the 5 trained CNN models.
    - Averaging predictions across all 5 models reduces bias.
  - **Tree Model Training**:
    - Conducts a cross-validation study on the complete dataset to train tree-based models.
    - LightGBM and XGBoost are re-optimized for this larger dataset.
  - **Final Scores**:
    - **LightGBM**: pAUC = **0.1768**.
    - **XGBoost**: pAUC = **0.1728**.
  - **Comparison**: Final LightGBM model exceeds the Kaggle leaderboard score of **0.1762**.

---

## Key Features of the Pipeline

1. **Hybrid Approach**: Combines CNNs for feature extraction with tree-based models for decision-making.
2. **Data Imbalance Handling**: Custom sampling and augmentation to create a balanced dataset for training.
3. **Custom Data Generators**: Efficient handling of large datasets with on-the-fly augmentation.
4. **OOF Predictions**: Ensures unbiased predictions for tree-based models.
5. **Tree Model Optimization**: Uses Optuna for hyperparameter tuning.
6. **Model Ensembling**: Averages predictions from 5 CNN models for robustness.

---

## Installation and Usage

### Prerequisites
- Python 3.11.5
- TensorFlow (2.12.0)
- LightGBM
- XGBoost
- Optuna
- Scikit-learn
- Pandas, NumPy, Matplotlib

### Steps to Reproduce
1. Clone this repository and navigate to the project folder:
   ```bash
   git clone <repo_url>
   cd skin-lesion-classification
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the notebooks in order:
   1. `KFoldTraining.ipynb` to train CNN models and generate OOF predictions.
   2. `Trees.ipynb` to build and tune tree-based models using the OOF predictions.
   3. `FullData.ipynb` to extend the approach to the entire dataset.

---

## Results

| Model        | Partial AUC (Validation) | Notes                               |
|--------------|---------------------------|-------------------------------------|
| MobileNetV0  | Per-fold pAUC ≈ 0.12     | 5-fold stratified cross-validation |
| LightGBM     | **0.1768**                | Final model, full dataset          |
| XGBoost      | 0.1728                    | Final model, full dataset          |

---

## Future Work

1. **Testing on Private Kaggle Dataset**: Validate the final model on unseen test data.
2. **Augmentation Refinements**: Experiment with additional augmentation strategies for malignant samples.
3. **Additional Metadata Features**: Explore inclusion of new features for tree-based models.

---

## Authors and Acknowledgments

- **Author**: Anshul Srivastava 
- **Acknowledgments**: Thanks to the Kaggle Skin Lesion Classification Challenge for providing the dataset and baseline metrics.
