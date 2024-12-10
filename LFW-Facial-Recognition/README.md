# Optimizing Traditional Machine Learning Methods for Facial Recognition

## Project Overview
This project explores facial recognition using traditional machine learning methods on the **Labeled Faces in the Wild (LFW)** dataset. The goal is to identify the optimal machine learning model for this multi-class classification task by balancing performance and computational efficiency. The project emphasizes preprocessing, hyperparameter optimization, and comparative analysis of different classifiers.

## Key Objectives
1. Preprocess the LFW dataset to prepare features for classification.
2. Address challenges such as high dimensionality and class imbalance.
3. Evaluate and compare the performance of four traditional machine learning methods:
   - Support Vector Machines (SVM)
   - Logistic Regression (LR)
   - K-Nearest Neighbors (KNN)
   - Linear Discriminant Analysis (LDA)
4. Identify the best model based on accuracy and training time.

## Methodology

### 1. **Data Preprocessing**
- **Feature Extraction**: Features were extracted from the dataset.
- **Dimensionality Reduction**: Principal Component Analysis (PCA) was applied to reduce dimensionality while retaining 95% of the variance.
- **Class Imbalance**: Strategies such as weighted classifiers and data augmentation were implemented to handle class imbalance effectively.

### 2. **Model Training and Tuning**
- Each model was trained using hyperparameter optimization techniques, including **Grid Search** and **Randomized Search**.
- Validation curves were used to analyze the impact of key hyperparameters on performance.
- Training and evaluation were conducted using stratified cross-validation to ensure balanced sampling across classes.

### 3. **Performance Metrics**
- **Accuracy**: Best cross-validation accuracy was used to compare the models.
- **Training Time**: Average training time was measured to assess computational efficiency.

### 4. **Model Comparison**
The models were compared based on their cross-validation accuracy and training time. The plots revealed:
- **KNN** performed the worst in terms of accuracy.
- **SVM**, **LR**, and **LDA** achieved similar accuracy, with LDA slightly outperforming.
- **LDA** significantly outperformed the others in training time, making it the optimal choice.

### 5. **Final Model Selection**
LDA was chosen as the final model due to its combination of high accuracy and minimal training time, making it well-suited for this task.

## Results
### Validation and Learning Curves
- **Validation Curves**: Explored the impact of hyperparameters such as `tol` and `shrinkage` on LDA performance.
- **Learning Curves**: Demonstrated that the LDA model generalizes better with larger datasets and effectively reduces overfitting as more data is added.

### Final Model Performance
- LDA achieved the best balance between cross-validation accuracy and computational efficiency, making it the most practical choice for facial recognition on the LFW dataset.


## Key Libraries
- **Scikit-learn**: For machine learning models and preprocessing.
- **Matplotlib & Seaborn**: For visualizing results.
- **NumPy**: For numerical operations.

## Conclusion
This project demonstrates the utility of traditional machine learning methods for facial recognition tasks. The results emphasize the importance of balancing performance and computational efficiency when selecting models. LDA emerged as the optimal choice, providing a strong baseline for similar classification problems.

## Future Work
- Explore additional methods to further address class imbalance.
- Experiment with advanced dimensionality reduction techniques.
- Extend the analysis to include deep learning approaches for comparison. 
