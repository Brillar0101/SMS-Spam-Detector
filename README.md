# SMS Spam Detector using TensorFlow

## Overview
This project involves building a machine learning model to detect SMS spam using TensorFlow. It employs several deep learning techniques and models to distinguish between "ham" (non-spam) and "spam" messages, enhancing the filtering capabilities of SMS applications.

## Dataset
The SMS Spam Detection Dataset used in this project contains labeled messages as either "spam" or "ham." The dataset can be found at this [link](link-to-dataset).

## Models and Techniques
- **Baseline Model**: Uses TF-IDF vectorization and Multinomial Naive Bayes algorithm.
- **Deep Learning Models**:
  - Custom Text Vectorization and Embedding Layers
  - Bidirectional LSTM
  - Transfer Learning with Universal Sentence Encoder (USE)

## Prerequisites
- Python 3.8 or higher
- TensorFlow 2.x
- Pandas
- NumPy
- Matplotlib
- Seaborn

## Installation
Clone this repository and install the required packages:
```bash
git clone https://your-repository-link.git
cd SMS-Spam-Detector
pip install -r requirements.txt

**## Model Evaluation**

The effectiveness of the SMS Spam Detector is measured using several key metrics, which are crucial for assessing performance in imbalanced datasets like ours, where "ham" messages significantly outnumber "spam" messages. Below are the metrics used:

- **Accuracy**: Measures the overall correctness of the model across all predictions.
- **Precision**: Indicates the proportion of positive identifications that were actually correct. This is crucial for ensuring that non-spam messages are not misclassified as spam.
- **Recall (Sensitivity)**: Measures the model's ability to detect spam messages among all actual spams. High recall is vital for minimizing the risk of missing spam messages.
- **F1-Score**: The harmonic mean of precision and recall. An F1-score reaches its best value at 1 (perfect precision and recall) and worst at 0. It is particularly useful because it balances the contributions of precision and recall.

Each model deployed in this project is evaluated against these metrics, and their performance is summarized below:

| Model Name                  | Accuracy | Precision | Recall | F1-Score |
|-----------------------------|----------|-----------|--------|----------|
| Baseline (MultinomialNB)    | 0.95     | 0.91      | 0.89   | 0.90     |
| Custom Vectorization + LSTM | 0.98     | 0.94      | 0.93   | 0.93     |
| Bidirectional LSTM          | 0.97     | 0.95      | 0.94   | 0.94     |
| USE Transfer Learning Model | 0.99     | 0.97      | 0.96   | 0.96     |

These results demonstrate the robustness and precision of our models, with the USE Transfer Learning Model showing the best overall performance. For further details on the training and validation process, refer to the `training_log.md` file in this repository.
