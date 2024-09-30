# Disaster Tweet Classification Using NLP and PyTorch

## Overview

This project focuses on building a Natural Language Processing (NLP) model to classify tweets as related to real disasters or not. Using PyTorch, we implemented a BERT-based model to analyze tweet text, allowing for accurate classification of disaster-related tweets.

The goal is to help organizations quickly identify relevant tweets during disaster events for better response and resource allocation.

## Dataset

The dataset contains tweet data with the following columns:
- `id`: Unique identifier for each tweet
- `keyword`: A keyword from the tweet (can be `NaN`)
- `location`: The location the tweet refers to (can be `NaN`)
- `text`: The tweet content
- `target`: Target label (1 = disaster-related, 0 = not disaster-related)

## Approach

### 1. **Preprocessing**

- **Text Tokenization**: We used the BERT tokenizer to preprocess tweets, converting them into input tensors that the BERT model can understand.
- **Handling Missing Values**: Missing values in the `keyword` and `location` columns were ignored since they were not needed for text-based classification.

### 2. **Model Selection**

We chose a BERT-based model for its strong performance in NLP tasks:
- **BERT (Bidirectional Encoder Representations from Transformers)** is a state-of-the-art language model capable of understanding context in text by considering words bidirectionally.

**Why BERT?**
- BERT excels in text classification tasks and captures context better than simpler models.
- It performs well even with relatively small datasets due to transfer learning.

### 3. **Training**

- **Loss Function**: Cross-entropy loss was used, suitable for multi-class classification problems.
- **Optimizer**: Adam optimizer with weight decay was chosen for efficient convergence.
- **Training Time**: We trained the model on a GPU-enabled laptop, with training times optimized using a batch size of 16 and 2-3 epochs for effective learning.

### 4. **Tracking Progress**

- Used the `tqdm` library to monitor training and prediction progress.
- Utilized `TensorBoard` for visualizing training and validation metrics over epochs.

### 5. **Prediction on Validation Set**

- Predictions were made on the validation set without labels using the trained model.
- A DataLoader was modified to handle unlabeled data efficiently.

### 6. **Evaluation and Output**

- The final model predictions were saved in a CSV file with columns `id` and `target` for easy interpretation.

## Requirements

- Python 3.7+
- PyTorch
- Transformers (Hugging Face)
- tqdm
- pandas
- numpy

Install the dependencies using:
```bash
pip install -r requirements.txt
```

## Future Work

•	Experiment with other transformer-based models like RoBERTa or DistilBERT for potentially faster inference.
•	Implement techniques for handling imbalanced data (e.g., data augmentation, weighted loss).
