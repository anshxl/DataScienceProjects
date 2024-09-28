# LFW Facial Recognition 

#### *Anshul Srivastava*

## Introduction

In today's digital age, the demand for robust and efficient facial recognition systems has surged across various domains, including security, authentication, surveillance, and human-computer interaction. The ability to accurately identify and verify individuals from images or video streams holds immense potential for enhancing security protocols, streamlining authentication processes, and improving user experiences in various applications.

The goal of this project is to develop a facial recognition system using Support Vector Machines (SVM), a powerful machine learning algorithm known for its effectiveness in classification tasks. We leverage the Labeled Faces in the Wild (LFW) dataset, a widely used benchmark dataset in the field of face recognition, containing a diverse set of facial images collected from the web.

Our approach involves several key steps, including data preprocessing, feature extraction using Principal Component Analysis (PCA) for dimensionality reduction, model building using SVM with hyperparameter tuning, and comprehensive evaluation of the model's performance.

Through this project, we aim to demonstrate the efficacy of SVM-based facial recognition systems and explore the impact of various factors, such as hyperparameters and feature representation, on the system's performance. Additionally, we seek to gain insights into the challenges and opportunities associated with building and deploying facial recognition systems in real-world scenarios.

## Dataset

The Labeled Faces in the Wild (LFW) dataset, accessible through the scikit-learn library, is a widely used benchmark dataset for face recognition tasks. It consists of a collection of facial images extracted from various online sources, such as news articles, Google Images, and celebrity websites. The dataset contains images of faces belonging to different individuals, captured under a variety of conditions, including varying lighting, pose, and facial expressions.

With over 13,000 images of approximately 5,000 individuals, LFW offers a diverse and challenging dataset for evaluating the performance of facial recognition algorithms. Each image in the dataset is labeled with the corresponding identity of the individual depicted, providing ground truth for training and evaluation purposes. This rich and expansive dataset serves as a valuable resource for researchers and practitioners working in the field of computer vision and machine learning, facilitating advancements in facial recognition technology.

For the purpose of this project, we only import those persons who have at least 50 unique images. This ensures our model has enough data to train on to classify a particular individual. This is ensured by the *min_faces_per_person* argument. Furthermore, the images are resized to 40% of their original size to maximize efficiency and speed.

## Methodology

1. **Data Preprocessing**: Import the LFW dataset and filter it to include only individuals with at least 50 images. Resize the images to 40% of their original size.
2. **Feature Extraction**: Use Principal Component Analysis (PCA) for dimensionality reduction.
3. **Model Building**: Train a Support Vector Machine (SVM) classifier with hyperparameter tuning using GridSearchCV.
4. **Evaluation**: Evaluate the model's performance using metrics such as confusion matrix and classification report.

## Results

The results section will include the performance metrics of the SVM model, such as accuracy, precision, recall, and F1-score. Additionally, visualizations such as confusion matrices and sample predictions will be provided to illustrate the model's effectiveness.

## Conclusion

This project demonstrates the potential of SVM-based facial recognition systems using the LFW dataset. The results indicate that SVM, combined with PCA for feature extraction, can achieve high accuracy in classifying individuals. Future work could explore the impact of different feature extraction techniques, alternative machine learning algorithms, and the integration of additional datasets to further enhance the system's performance.

## Installation and Usage

1. **Clone the repository**:
   ```sh
   git clone https://github.com/anshxl/DataScienceProjects.git
   cd your-repo-name
   ```
2. Install the required libraries:
  ```sh
  pip install -r requirements.txt
  ```
3. Run the Jupyter Notebook:
  ```sh
  jupyter notebook PROJECT_V2.ipynb
  ```
4. Follow the instructions in the notebook to execute the code cells and visualize the results.

## License
This project is licensed under the MIT License - see the LICENSE file for details.



