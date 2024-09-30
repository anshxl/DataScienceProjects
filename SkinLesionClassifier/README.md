# Skin Lesion Classification Using AI

## Overview
This project aims to develop an AI-based binary classification model capable of identifying histologically confirmed malignant and benign skin lesions from high-resolution images. By leveraging a dataset of over 45,000 skin lesion images, alongside metadata describing their attributes, the project focuses on creating a robust model suitable for deployment in a smartphone app, thereby facilitating early skin cancer detection in non-clinical settings.

## Project Motivation
Skin cancer is a prevalent and potentially deadly disease, but early detection significantly improves patient outcomes. Many populations lack access to specialized dermatologic care, making it crucial to develop AI solutions that can operate in primary care or telehealth settings. This project strives to extend the benefits of automated skin cancer detection to a broader audience by building a model that evaluates lower-quality images, akin to those taken with smartphones, to assist in triaging and early diagnosis.

## Dataset
The dataset consists of:
- **45,000 skin lesion images**: These images are high-resolution and resemble those captured by a smartphone.
- **Metadata**: Each image is accompanied by additional attributes such as age, lesion size, location, etc., providing valuable context that could enhance classification accuracy.

The dataset is based on 3D total body photos (TBP), with single-lesion crops sourced from thousands of patients across multiple continents.

## Goals
- **Binary Classification**: Develop a model that can differentiate between malignant and benign skin lesions.
- **Smartphone Compatibility**: Create a model that performs effectively with lower-quality images typically captured by smartphones for telehealth purposes.
- **Incorporate Metadata**: Utilize metadata to improve classification accuracy and develop a unified model that leverages both image and non-image data.

## Approach
1. **Image-Based Model**:
   - Building a convolutional neural network (CNN) to process the skin lesion images, taking advantage of pre-trained architectures (e.g., ResNet, EfficientNet) for transfer learning.
   - Fine-tuning the model to accurately differentiate between malignant and benign lesions.

2. **Metadata Integration**:
   - Training a separate classifier using metadata attributes to predict lesion malignancy.
   - Using knowledge distillation techniques to incorporate the insights from the metadata model into the image-only model.

3. **Knowledge Distillation**:
   - Training a "teacher" model using both image and metadata inputs.
   - Training a "student" model that uses only image data but mimics the teacher’s predictions, allowing it to benefit from the metadata's influence without requiring it during deployment.

## Expected Outcomes
- An AI model that achieves high accuracy in classifying skin lesions as malignant or benign, suitable for use in a smartphone app.
- A lightweight, efficient model that can perform inference on lower-quality images, extending its application to underserved populations and telehealth environments.

## Potential Impact
- Enhancing early skin cancer detection, particularly in non-clinical or resource-limited settings.
- Improving triage and diagnosis efficiency, leading to better long-term outcomes for patients.

## Tools & Technologies
- **Deep Learning Framework**: PyTorch / TensorFlow / Keras
- **CNN Architectures**: ResNet, EfficientNet, or MobileNet
- **Data Processing**: OpenCV, Pandas, and Scikit-learn
- **Visualization**: Matplotlib, Seaborn

## Future Work
- Incorporate more advanced techniques like data augmentation and ensemble learning to further improve model accuracy.
- Explore integration with smartphone camera APIs for real-time lesion analysis.
