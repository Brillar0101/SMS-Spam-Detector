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
