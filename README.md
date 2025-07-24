# Toxic Comment Classification

This project implements a deep learning model to classify toxic comments from the Jigsaw Toxic Comment Classification Challenge dataset. The model is built using TensorFlow and Keras and includes a user-friendly interface created with Gradio for real-time comment scoring.

## 📜 Description

The primary objective of this project is to build a robust model capable of identifying and categorizing various types of toxic language in online comments. This includes comments that are toxic, severely toxic, obscene, threatening, insulting, or contain identity-based hate. By accurately classifying such comments, we can help create a safer and more inclusive online environment.

The project covers the entire machine learning pipeline, from data preprocessing and model training to evaluation and deployment with a Gradio web interface.

## 💾 Dataset

The model is trained on the [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) dataset. This dataset contains a large number of Wikipedia comments that have been labeled by human raters for their level of toxicity.

The dataset includes the following toxicity categories:
*   `toxic`
*   `severe_toxic`
*   `obscene`
*   `threat`
*   `insult`
*   `identity_hate`

## ⚙️ Installation

To run this project, you need to have Python 3 installed. You can clone the repository and install the necessary dependencies using pip:

```bash
git clone https://github.com/your-username/toxic-comment-classification.git
cd toxic-comment-classification
pip install -r requirements.txt
```

**Dependencies**
This project relies on the following Python libraries:
*   `tensorflow`
*   `pandas`
*   `numpy`
*   `gradio`
*   `jinja2`

## 🚀 Usage

You can run the Jupyter Notebook `Comment_Toxisity.ipynb` to see the entire process of data loading, preprocessing, model training, and evaluation.

To launch the interactive Gradio interface for scoring comments, run the final cells of the notebook. This will create a public URL that you can use to interact with the model in your browser.

## 🤖 Model Architecture

The model is a Sequential model built with Keras and consists of the following layers:

1.  **Embedding Layer**: Converts input text into dense vectors of fixed size. It takes `MAX_FEATURES + 1` (200001) as the input dimension and produces embeddings of dimension 32.
2.  **Bidirectional LSTM Layer**: A recurrent neural network layer that processes the sequence in both forward and backward directions, capturing contextual information from both past and future. It has 32 units and uses the `tanh` activation function.
3.  **Dropout Layer**: A regularization layer to prevent overfitting by randomly setting 20% of the input units to 0 during training.
4.  **Dense Layers**: A set of fully connected layers with ReLU activation functions (128, 256, and 128 units) to learn complex patterns from the features extracted by the LSTM layer.
5.  **Output Layer**: A final Dense layer with 6 units and a sigmoid activation function to output a probability score for each of the 6 toxicity categories.

The model is compiled with the `binary_crossentropy` loss function and the `adam` optimizer.

## 📊 Evaluation

The model's performance is evaluated on the test set using the following metrics:

*   **Precision**: 0.929
*   **Recall**: 0.888
*   **Accuracy**: 0.517

These metrics indicate that the model is effective at identifying toxic comments, with a high precision in its predictions.
